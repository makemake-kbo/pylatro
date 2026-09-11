"""Parquet metadata, lossless observation loading, and bounded-memory training."""
import json
from dataclasses import replace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pylatro_agent.constants import TOKENIZER_VERSION
from pylatro_agent.reward import DEFAULT_REWARD_CONFIG
from pylatro_agent.training.model_generate import load_records, save_records
from pylatro_agent.training.parquet_records import ParquetRecords


def sample_records(count=4):
    return [dict(obs={
        "tokens": np.arange(12, dtype=np.int16).reshape(3, 4) + i,
        "scalars": np.array([i / 7, -0.5], dtype=np.float32),
        "history_cards": np.arange(24, dtype=np.int16).reshape(2, 2, 2, 3),
        "action_mask": np.array([1, 0, 1], dtype=np.int8),
    }, action=0, seed=100 + i, won=i % 2 == 0, reward=i / 3,
        return_target=-i / 7, max_ante=8, tokenizer_version=TOKENIZER_VERSION,
        heuristic_blind_rollout=None) for i in range(count)]


def assert_record_equal(expected, actual):
    assert expected.keys() == actual.keys()
    for key in expected:
        if key == 'obs':
            assert expected[key].keys() == actual[key].keys()
            for name, array in expected[key].items():
                np.testing.assert_array_equal(array, actual[key][name], strict=True)
        else:
            assert expected[key] == actual[key]


def test_parquet_roundtrip_and_indexing(tmp_path):
    original = sample_records(600)
    path = tmp_path / 'records.parquet'
    save_records(original, path, reward_config=DEFAULT_REWARD_CONFIG)
    loaded = load_records(path, reward_config=DEFAULT_REWARD_CONFIG)
    assert isinstance(loaded, ParquetRecords)
    assert len(loaded) == 600
    assert len(loaded._cache) == 0
    for expected, actual in zip(original, loaded, strict=True):
        assert_record_equal(expected, actual)
    for i in [0, 255, 256, 599, -1]:
        assert_record_equal(original[i], loaded[i])
    assert len(loaded._cache) <= 2
    assert len(loaded[2:10:2]) == 4
    assert sorted(loaded.shuffled_indices()) == list(range(600))
    with pytest.raises(IndexError):
        loaded[600]
    with pytest.raises(IndexError):
        loaded[-601]


def test_parquet_metadata_rejects_wrong_rewards(tmp_path):
    path = tmp_path / 'records.parquet'
    save_records(sample_records(), path, reward_config=DEFAULT_REWARD_CONFIG)
    with pytest.raises(ValueError, match='reward fingerprint mismatch'):
        load_records(path, reward_config=replace(DEFAULT_REWARD_CONFIG, gamma=.5))


@pytest.mark.parametrize('key,value,match', [
    ('tokenizer_version', -1, 'TOKENIZER_VERSION'),
    ('tokenizer_semantics', 'old', 'current semantics'),
    ('reward_fingerprint', None, 'predates reward metadata'),
])
def test_parquet_metadata_rejects_incompatible_schema(tmp_path, key, value, match):
    path = tmp_path / 'records.parquet'
    save_records(sample_records(), path, reward_config=DEFAULT_REWARD_CONFIG)
    table = pq.read_table(path)
    metadata = json.loads(table.schema.metadata[b'pylatro'])
    metadata[key] = value
    pq.write_table(table.replace_schema_metadata({b'pylatro': json.dumps(metadata).encode()}), path)
    with pytest.raises(ValueError, match=match):
        load_records(path, reward_config=DEFAULT_REWARD_CONFIG)


def test_published_layout_with_sidecar_and_step(tmp_path):
    path = tmp_path / 'data' / 'train-00000.parquet'
    original = sample_records()
    save_records(original, path, reward_config=DEFAULT_REWARD_CONFIG)
    table = pq.read_table(path)
    metadata = json.loads(table.schema.metadata[b'pylatro'])
    table = table.append_column('step', pa.array(range(len(table))))
    pq.write_table(table.replace_schema_metadata(None), path)
    with pytest.raises(ValueError, match='require embedded metadata'):
        load_records(path, reward_config=DEFAULT_REWARD_CONFIG)
    (tmp_path / 'metadata').mkdir()
    (tmp_path / 'metadata/schema.json').write_text(json.dumps(metadata))
    for source in [path, path.parent, tmp_path]:
        loaded = load_records(source, reward_config=DEFAULT_REWARD_CONFIG)
        for expected, actual in zip(original, loaded, strict=True):
            assert_record_equal(expected, actual)


def test_parquet_directory_and_empty_input(tmp_path):
    for i in range(2):
        save_records(sample_records(), tmp_path / f'{i}.parquet', reward_config=DEFAULT_REWARD_CONFIG)
    assert len(load_records(tmp_path, reward_config=DEFAULT_REWARD_CONFIG)) == 8
    with pytest.raises(ValueError, match='empty records'):
        save_records([], tmp_path / 'empty.parquet', reward_config=DEFAULT_REWARD_CONFIG)
    (tmp_path / 'empty').mkdir()
    with pytest.raises(ValueError, match='No Parquet files'):
        load_records(tmp_path / 'empty', reward_config=DEFAULT_REWARD_CONFIG)


def test_pickle_compatibility_warns(tmp_path):
    path = tmp_path / 'legacy.pkl'
    with pytest.warns(FutureWarning, match='deprecated'):
        save_records(sample_records(), path, reward_config=DEFAULT_REWARD_CONFIG)
    with pytest.warns(FutureWarning, match='deprecated'):
        loaded = load_records(path, reward_config=DEFAULT_REWARD_CONFIG)
    assert_record_equal(sample_records()[0], loaded[0])


def test_supervised_uses_parquet_without_generation(tmp_path, monkeypatch):
    from pylatro import load_game_data
    from pylatro_agent.agent import AgentConfig
    from pylatro_agent.tokenizer import Tokenizer
    from pylatro_agent.training import supervised
    from pylatro_agent.training.fast_generate import _build_obs
    from pylatro_agent.training.fast_runner import FastRunner
    from pylatro_agent.vocab import build_vocab

    data = load_game_data()
    runner = FastRunner(0, data)
    obs = _build_obs(runner, Tokenizer(vocab=build_vocab(data)))
    record = dict(obs=obs, action=0, seed=0, won=True, max_ante=8,
                  return_target=0.0, terminal_outcome_target=8, terminal_outcome_mask=1.0,
                  tokenizer_version=TOKENIZER_VERSION)
    path = tmp_path / 'records.parquet'
    save_records([record], path, reward_config=DEFAULT_REWARD_CONFIG)

    def fail(*args, **kwargs):
        raise AssertionError('Pretraining should use the supplied dataset')

    monkeypatch.setattr(supervised, 'generate_training_data', fail)
    supervised.train_supervised(
        supervised.SupervisedConfig(data_path=str(path), device='cpu', max_epochs=1,
                                    batch_size=1, num_games=1,
                                    save_dir=str(tmp_path / 'checkpoints'), log_dir=str(tmp_path / 'logs')),
        AgentConfig(d_model=16, n_layers=1, n_heads=2, d_ff=32), data=data,
    )
    assert (tmp_path / 'checkpoints/supervised_epoch1.pt').is_file()
