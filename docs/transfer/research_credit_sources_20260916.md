# Credit assignment research notes, 2026-09-16

These notes connect primary literature to the stopped complete-cohort run. They contain no new training result. Production code, checkpoints, and the MPS ledger were not changed. Mathematical counterexamples below establish possible mechanisms, not the cause of the observed gameplay plateau.

## The quantities that must be kept separate

The current objective is expected terminal reward: `10` for a win, `0` otherwise, with gamma one. For the policy that actually generates the continuation,

\[
V^\pi(h_t)=10\,P_\pi(\text{win}\mid h_t),\quad
A^\pi(h_t,a_t)=Q^\pi(h_t,a_t)-V^\pi(h_t).
\]

Here `h` denotes all information available before acting; a sufficient Markov observation can replace it. These are expectations over future play. An individual loss does not make a forecast of 40% wrong. Likewise, forecasting the average win rate correctly does not show that the critic distinguishes stronger states or provides accurate action advantages.

For any predictor `v(h)` and return `Y`, our elementary conditional-variance decomposition is

\[
\mathbb E[(Y-v)^2\mid h]
=\operatorname{Var}(Y\mid h)+(\mathbb E[Y\mid h]-v)^2.
\]

A perfect 40%-win predictor has return MSE `100 × .4 × .6 = 24`. Raw MSE around 19 is therefore not evidence, on its own, of a catastrophically inaccurate conditional mean. For a head `v=10p`, return MSE equals `100 × win Brier score` exactly; these are not independent confirmations.

At the recent +60 sampled checkpoint, the recorded game-weighted outcome-head Brier is approximately `.18456`. A constant forecast fitted to that panel's pooled win frequency `.41` has Brier `.2419`. The first is about 23.7% lower. This is a descriptive comparison against a fitted base rate, not a preregistered held-out model comparison. It also mixes visited states from two policies. It shows predictive information beyond a constant on those visited states, but does not establish learning progress or an advantage over an ante/phase-only baseline.

Forecast verification has long separated reliability, resolution, and event uncertainty. Murphy's original decomposition is the relevant precedent; a constant base-rate forecast can be calibrated while having no resolution. Finite reliability bins add estimation error, so report their game-clustered uncertainty and counts. [Murphy, 1973](https://doi.org/10.1175/1520-0450(1973)012%3C0595:ANVPOT%3E2.0.CO;2).

## What GAE and rollout cuts actually do

The original GAE paper makes the distinction explicit: with an exact value function, short TD-based advantages can be correct; with an approximate value function, lambda below one trades bias for variance. Full-episode lambda one telescopes to realized return minus the initial baseline. Therefore a small coefficient on the literal terminal reward is not a proof that an early action receives no useful information. The information can arrive through intermediate value differences. [Schulman et al., GAE, Sections 2–4 and equations 14–18](https://arxiv.org/html/1506.02438v6).

The implementation in `training/cohort.py::fragment_gae` computes

\[
\delta_t=r_t+\widehat V_{t+1}-\widehat V_t,\qquad
\widehat A_t=\sum_{k=0}^{n-1}\lambda^k\delta_{t+k},
\]

where `n` stops at either true termination or the next logical 256-step cut. True termination has value zero; an artificial cut retains the next state's bootstrap. This is recognizable truncated GAE, not accidentally treating every fragment as death. The original PPO paper itself describes fixed-length fragments with bootstrapping. Its existence establishes correctness as an estimator choice, not suitability for our environment. [Schulman et al., PPO, equations 10–12](https://arxiv.org/pdf/1707.06347).

**Our algebraic consequence for the current implementation.** Write value error as `e_t = Vhat_t − Vπ_t`. At gamma one the difference between GAE using the approximate and exact values is

\[
\Delta\widehat A_t
=-e_t+(1-\lambda)\sum_{j=1}^{n-1}\lambda^{j-1}e_{t+j}
+\lambda^{n-1}e_{t+n}.
\]

The last error is zero at a true terminal state. This identifies the relevant problem more precisely than “the horizon is too long”: bootstrapped **future value errors** enter the action estimator. The initial `−e_t` is a state baseline; in the unmodified on-policy score-function gradient it integrates to zero. State-dependent errors at future states can depend on the current action through its consequences.

Consider a constant additive error `c` over an entirely nonterminal fragment. All TD differences cancel it, so `ΔA=0`; however the regression target `Vhat+Ahat` retains the offset `c`. Local critic regression can have small loss without correcting the offset. If the fragment reaches a terminal state, the target corrects an otherwise constant offset by only `c λ^(n−1)`. With lambda `.97` and `n=101`, this is about `.04755c`. A cut before termination excludes this anchoring term entirely from that target. This is an exact illustrative mechanism for slow correction, not evidence that the measured positive residual has that particular origin.

The audit's roughly 0.7% mean direct terminal coefficient for decisions over 100 actions before termination measures estimator dependence on bootstrapping. It does not measure the fraction of true action credit retained. Removing cuts while retaining `.97`, increasing lambda while retaining cuts, and using full-episode Monte Carlo are three different experiments.

## Why full-episode Monte Carlo is a diagnostic, not a promised fix

For the collection policy, the score-function identity allows

\[
\nabla J(\theta)
=\mathbb E\left[\sum_t\nabla\log\pi_\theta(a_t\mid h_t)
\left(Y-b(h_t)\right)\right].
\]

The baseline may be inaccurate without introducing bias provided it is independent of the sampled action conditional on the pre-action information. Freeze it from before collection, or cross-fit it: aggressively fitting and reusing a baseline on the same trajectories can violate the simple independence argument. The identity does not make PPO's clipped, reused, normalized finite-batch update unbiased. It describes the underlying estimator at the collection policy. [Williams, 1992, original paper](https://gwern.net/doc/reinforcement-learning/model-free/1992-williams.pdf).

Monte Carlo uses the completed outcome at every decision and avoids future value bootstraps. Its weakness is variance: one bad early choice can coexist with a win, and a good choice can precede a later blunder. More decisions bearing the same outcome are not independent outcome observations. In a 64-game cohort, drawing 8192 decisions twice does not create 16384 independent games. Measure gradient signal-to-noise across whole games as well as decision count.

A baseline that best predicts returns need not minimize policy-gradient variance. The gradient's own magnitudes and correlations matter, which is why good outcome NLL and useful policy credit are different evaluation questions. Greensmith et al. derive optimal control variates and show that usual baseline choices need not be variance-optimal. Their analysis uses an average-reward reactive-policy setting, so its exact bounds should not be transplanted to episodic PPO. [Greensmith, Bartlett and Baxter, 2004](https://jmlr.org/papers/v5/greensmith04a.html).

**Partial-observation inference.** If the encoded observation merges distinct histories, even its ideal conditional outcome predictor need not obey a Markov Bellman recursion across encoded observations. Full-history or belief-state values do. Monte Carlo with a pre-action baseline does not require assuming that a compressed observation is Markov. This is another reason it is a useful diagnostic; it is not evidence that observation aliasing is our demonstrated bottleneck.

## Critic targets do not have to equal actor targets

Two original OpenAI implementations make different legitimate choices:

| Implementation | Actor estimate | Value target and optimizer |
| --- | --- | --- |
| [Spinning Up PPO](https://raw.githubusercontent.com/openai/spinningup/master/spinup/algos/pytorch/ppo/ppo.py) | GAE | Bootstrapped reward-to-go, squared error, separate value optimizer; value fitting proceeds after actor KL early stopping. See `finish_path`, `compute_loss_v`, and `update`. |
| [Baselines PPO2](https://raw.githubusercontent.com/openai/baselines/master/baselines/ppo2/runner.py) | GAE | `returns = advantages + values`, a lambda-return target. See `Runner.run`. |

The new complete cohorts permit a target-only critic investigation: retain the actor estimator while evaluating critic fitting to complete outcomes or Monte Carlo returns. Changing critic supervision and actor credit simultaneously would obscure which part helped.

Our `ValueHead` emits `10P(win)+residual`. In the current pure terminal, undiscounted task, a correct outcome predictor already specifies the entire expected return. The residual can compensate for imperfect fitting, but is not mathematically required by dense rewards or discounting here. It is separately trained to the bootstrapped lambda target, so improved outcome probabilities do not establish that the value consumed by GAE improved.

## Loss geometry can change what “value” means

`value_head.py::return_huber_loss` fits the residual using Huber delta one. Fixed-threshold Huber regression does not generally elicit a conditional mean. Feng and Wu distinguish Huber-risk consistency from mean-regression consistency and analyze the role of the threshold. [A Statistical Learning Assessment of Huber Regression](https://arxiv.org/abs/2009.12755).

**Our direct example.** For `Y∈{0,10}` with win probability `p`, solving `E[clip(v−Y,−1,1)]=0` yields

\[
v^*_{\rm Huber}=\begin{cases}
p/(1-p),&p<1/2,\\
\text{any point in }[1,9],&p=1/2,\\
10-(1-p)/p,&p>1/2.
\end{cases}
\]

At `p=.2`, the optimum is `.25`, whereas the mean is `2`. At `.4`, it is about `.667`, whereas the mean is `4`. Adding a fixed terminal prediction and optimizing an unconstrained residual does not remove this property: Huber still acts on total prediction error. The current target is bootstrapped GAE, not a binary Monte Carlo outcome; these numbers are not predictions of its live bias. They are a reason not to blindly replace its target with Monte Carlo while retaining Huber.

A large primary empirical study found MSE better than Huber for critic learning, and separate policy/value networks better in four of five continuous-control environments. Those results motivate bounded tests, not universal architecture prescriptions; continuous-control rewards, observations, and action spaces differ substantially from Balatro. The same paper emphasizes interacting hyperparameters and reports evidence that stale advantages can matter. [Andrychowicz et al., Sections 3.2, 3.4–3.5](https://arxiv.org/pdf/2006.05990).

## Shared representation and optimization exposure

Phasic Policy Gradient separates actor and critic optimization, then distills useful features subject to a policy-preservation term. Its motivation includes both gradient interference and the greater sample reuse value fitting can tolerate. This is directly relevant if our critic is underfit after only two passes or if critic fitting changes actor behavior through the shared trunk. It does not establish either condition in this run. [Cobbe et al., 2021](https://proceedings.mlr.press/v139/cobbe21a.html); [original implementation](https://github.com/openai/phasic-policy-gradient).

Measure distinct things before choosing a remedy: critic gradient magnitude, actual parameter change, fixed-set loss before/after fitting, held-out-by-game loss, actor-only versus critic-only gradient cosine, and policy KL induced by critic-only steps. A low aggregate loss can hide early-state errors. A low global learning rate can limit learning but is not diagnosed by its numerical value alone. A narrow frozen-head fit can distinguish head capacity from shared-representation or optimization limitations.

## More ambitious credit methods and why they come later

| Primary work | Mechanism and relevance | Preconditions and limits here |
| --- | --- | --- |
| [RUDDER, Arjona-Medina et al., 2019](https://proceedings.neurips.cc/paper/2019/hash/16105fb9cc614fc29e1bda00dab60d41-Abstract.html); [paper](https://papers.neurips.cc/paper/2019/file/16105fb9cc614fc29e1bda00dab60d41-Paper.pdf); [code](https://github.com/ml-jku/rudder) | Learns sequence return prediction, uses changes in predictions to redistribute delayed rewards, and includes a correction to preserve return. Directly addresses sparse terminal supervision. | A better sequence predictor and contribution analysis are required. Naively paying for increases in our imperfect `P(win)` is not equivalent to implementing RUDDER. Predicted contributions are not automatically causal effects. Must verify return equivalence and compare to MC and value-only controls. |
| [Temporal Value Transport, Hung et al., 2019](https://www.nature.com/articles/s41467-019-13073-w) | Uses memory retrieval to transport value to distant relevant events; demonstrates tasks with long temporal delays. | Requires a new memory/retrieval mechanism. The authors explicitly acknowledge heuristic elements and that transport can bias the objective and hurt short-credit tasks. It is evidence that specialized credit methods can help, not a ready-made correctness repair. |
| [Hindsight Credit Assignment, Harutyunyan et al., 2019](https://papers.nips.cc/paper/2019/file/195f15384c2a79cedf293e4a847ce85c-Paper.pdf) | Learns relationships between actions and subsequent events to improve assignment of delayed outcomes. | Introduces additional estimated conditional distributions. Need evidence that MC variance or causal attribution, rather than basic fitting, is limiting progress before taking on those approximation errors. |
| [Counterfactual Credit Assignment, Mesnard et al., 2021](https://proceedings.mlr.press/v139/mesnard21a/mesnard21a.pdf) | Conditions baselines on future information to distinguish action effects from luck. Derives conditions for low-variance unbiased policy-gradient estimation. | Future information must be constrained to avoid leaking action information and introducing bias. For diagnosis, our simulator already supports more interpretable repeated continuations with a single forced action, subject to verified state/RNG isolation. That is not an implementation of the paper's learned counterfactual baseline. |

## What existing experiments already rule against claiming

The repository's [previous outcome PPO report](outcome_ppo_results.md) must accompany any new proposal:

- The original transferred critic forecast about 83% wins against 16% observed in its student validation setting. A teacher-policy outcome predictor is not automatically a student-policy value function. Private-head fitting corrected that transfer calibration, but both PPO arms used it, so its separate gameplay effect was not measured.
- Lambda one has already been tried **with fragment bootstrapping retained**. The direct sampled advantage of lambda `.97` was `+1.08 [−1.67,+3.92]` points, so neither setting was established as superior. This did not test boundary-free full-episode Monte Carlo; replica-zero's early restart also limits the clean interpretation.
- The prior complete-critic treatment bundled zero residual, frozen critic during actor updates, changed label source, gradient scope, and replay exposure. Its sampled difference was `−.42 [−3.67,+2.75]` points and greedy difference `−4.25 [−8.25,−.25]`. It provides no positive evidence for the bundle and cannot identify residual removal alone.
- A previous frozen gradient probe found weighted critic/actor backbone norm ratio `.145` and cosine `−.070`. This does not support “the critic overwhelms the actor.” A new hypothesis about current interference needs a current, stratified measurement.
- Huber's binary-target mismatch was already demonstrated locally. Treat it as established loss geometry, not a newly discovered gameplay cause.
- The new complete-cohort run removed the original missing-outcome-label selection mechanism, but kept actor GAE and residual bootstrapping. Label coverage is a resolved implementation property; actor credit quality is not thereby resolved.

## Research gates before another long run

1. Define the estimand for every head and metric: continuation policy, information set, game/decision weighting, and whether the target is a realized outcome or bootstrapped estimate.
2. Recheck the stopped run's legality failure independently. No statistical explanation for slow learning excuses an invalid sampled action.
3. On frozen checkpoints and fixed complete episodes, characterize outcome discrimination/calibration, residual target bias by distance to termination, and per-objective gradients. Split fitting and assessment by whole game.
4. Test whether the critic can fit and generalize when the actor is held fixed. Separate loss choice, target choice, fitting exposure, and representation scope. Avoid treating an aggressively overfit training subset as proof of useful values.
5. Compare local action advantages against repeated continuations from identical snapshots, reporting uncertainty and intervening on the action rather than calling every disagreement an error. Use independent future policy draws; common randomness can reduce variance but is not proof that different actions consume simulator randomness identically.
6. Only then select a small matched actor experiment: fragment removal at fixed lambda, full-episode Monte Carlo with frozen baseline, or improved critic with the original actor estimator. Keep the established negative and inconclusive controls visible.

The literature supports investigating our target definitions and measurement assumptions before extending PPO. It does not yet justify declaring long horizons, critic interference, residual architecture, or GAE itself to be the single established cause.
