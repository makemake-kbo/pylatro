"""CPU evaluation environments with central policy inference and bounded workers."""

from __future__ import annotations

import multiprocessing as mp
import time
import traceback
from contextlib import suppress
from multiprocessing.connection import wait

import numpy as np

from ..constants import NUM_ACTIONS
from ..env import BalatroEnv


def _pack_observation(obs):
    # Environment legality masks are binary int8 arrays. Pack only the wire
    # representation; policy inputs retain their original values and dtype.
    return {**obs, "action_mask": np.packbits(obs["action_mask"])}


def _unpack_observation(obs):
    return {**obs, "action_mask": np.unpackbits(obs["action_mask"], count=NUM_ACTIONS).view(np.int8)}


class _Environments:
    def __init__(self, kwargs, *, compact_observations=False):
        self.compact_observations = compact_observations
        self.kwargs = kwargs
        self.envs = {}

    def reset(self, seed):
        env = BalatroEnv(seed=seed, **self.kwargs)
        self.envs[seed] = env
        obs, _ = env.reset(seed=seed)
        return (_pack_observation(obs) if self.compact_observations else obs), str(env._sub_phase)

    def step(self, actions):
        results = []
        for seed, action in actions:
            env = self.envs[seed]
            started = time.perf_counter()
            obs, _, terminated, truncated, info = env.step(action)
            seconds = time.perf_counter() - started
            # Avoid sending reward diagnostics and other training-only payloads
            # through pipes. These are all the fields evaluation consumes.
            compact = {
                key: info[key]
                for key in (
                    "ante",
                    "won",
                    "round_score",
                    "stalled",
                    "dollars",
                    "blind_on_deck",
                    "boss_key",
                )
                if key in info
            }
            compact["phase"] = str(env._sub_phase)
            if terminated or truncated:
                compact["joker_keys"] = list(env.state.joker_keys)
                compact["cleared_antes"] = sorted(
                    env._cleared_boss_antes | ({env.state.win_ante} if info.get("won") else set())
                )
                env.close()
                del self.envs[seed]
            if self.compact_observations:
                obs = _pack_observation(obs)
            results.append((seed, (obs, terminated, truncated, compact, seconds)))
        return results

    def close(self):
        for env in self.envs.values():
            env.close()
        self.envs.clear()


def _worker(connection):
    # A worker owns CPU engines only. Never copy the model or initialize CUDA.
    import torch

    torch.set_num_threads(1)
    envs = _Environments(connection.recv(), compact_observations=True)
    try:
        while True:
            command, payload = connection.recv()
            if command == "close":
                return
            result = getattr(envs, command)(payload)
            connection.send((True, result))
    except EOFError:
        pass
    except BaseException:
        with suppress(BrokenPipeError, EOFError, OSError):
            connection.send((False, traceback.format_exc()))
    finally:
        envs.close()
        connection.close()


class EvaluationEnvs:
    """Advance disjoint games in parallel; return rows in caller order.

    Workers retain only live environments, and each owns several batch slots.
    Spawn avoids inheriting a trainer's CUDA context or thread pools. Zero
    workers runs the same protocol locally, useful for small panels and tests.
    """

    def __init__(self, workers: int, **kwargs):
        if workers < 0:
            raise ValueError("eval_workers must be non-negative")
        self.local = _Environments(kwargs) if workers == 0 else None
        self.connections = []
        self.processes = []
        self.owners = {}
        self.loads = [0] * workers
        try:
            context = mp.get_context("spawn")
            for _ in range(workers):
                parent, child = context.Pipe()
                process = context.Process(target=_worker, args=(child,), daemon=True)
                try:
                    process.start()
                except BaseException:
                    parent.close()
                    child.close()
                    raise
                child.close()
                self.connections.append(parent)
                self.processes.append(process)
            # Start every interpreter before sending game data. Passing the
            # large data object in Process args blocks start() on each child
            # import, accidentally serializing interpreter startup.
            for connection in self.connections:
                connection.send(kwargs)
        except BaseException:
            self.close()
            raise

    def _receive(self, index):
        connection = self.connections[index]
        process = self.processes[index]
        ready = wait([connection, process.sentinel])
        if connection not in ready:
            raise RuntimeError(f"Evaluation worker {index} exited with code {process.exitcode}")
        try:
            ok, result = connection.recv()
        except (EOFError, OSError) as error:
            raise RuntimeError(f"Evaluation worker {index} disconnected") from error
        if not ok:
            raise RuntimeError(f"Evaluation worker {index} failed:\n{result}")
        return result

    def reset(self, seed):
        if self.local is not None:
            return self.local.reset(seed)
        index = min(range(len(self.loads)), key=self.loads.__getitem__)
        self.connections[index].send(("reset", seed))
        result = self._receive(index)
        self.owners[seed] = index
        self.loads[index] += 1
        obs, phase = result
        return _unpack_observation(obs), phase

    def step(self, actions):
        if self.local is not None:
            return dict(self.local.step(actions))
        shards = [[] for _ in self.processes]
        for seed, action in actions:
            shards[self.owners[seed]].append((seed, action))
        # Dispatch every shard before receiving, so engines actually overlap.
        for index, shard in enumerate(shards):
            if shard:
                self.connections[index].send(("step", shard))
        results = {}
        for index, shard in enumerate(shards):
            if shard:
                for seed, result in self._receive(index):
                    obs, terminated, truncated, info, seconds = result
                    results[seed] = (_unpack_observation(obs), terminated, truncated, info, seconds)
                    if result[1] or result[2]:
                        del self.owners[seed]
                        self.loads[index] -= 1
        return results

    def close(self):
        if self.local is not None:
            self.local.close()
        # Also handles callback failures, partial startup, and crashed workers.
        for connection in self.connections:
            with suppress(BrokenPipeError, EOFError, OSError):
                connection.send(("close", None))
            connection.close()
        for process in self.processes:
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join()
            process.close()
        self.connections.clear()
        self.processes.clear()
