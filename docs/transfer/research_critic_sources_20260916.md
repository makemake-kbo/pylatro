# Critic research: what is being learned, and how to measure it

Research notes dated 2026-09-16. Sources below are original papers or official documentation. This is a literature-to-code analysis, not a new training experiment. No MPS work, production edits, or policy changes were performed. The stopped run's safety failure remains a separate prerequisite for further rollouts.

## What the existing evidence supports

The new checkpoint audit establishes that the critic is updating: both critic towers advanced 1,560 Adam steps over the matched 60 updates ([audit](data/critic_update_audit_20260916.json)). Every committed complete-cohort update has full outcome labels and the prescribed optimizer exposure. Training losses change. However, there is no resolved improvement in evaluation forecast quality on the small development panel. Measurements on successive policies also change both the visited states and the policy whose future outcomes the critic should predict. See the [existing artifact audit](../../eval/complete_cohort_long_20260915/analysis_20260916/README.md).

The code has two distinct estimation problems:

- `outcome_nll` in [value_head.py](../../src/pylatro_agent/value_head.py) learns a categorical terminal distribution from realized complete-game labels.
- `return_huber_loss` fits a residual to `GAE_return - stop_gradient(terminal_value)`, with Huber threshold 1. PPO consumes their sum. The target is not the realized complete-game return.

The ordinary PPO update jointly differentiates policy, outcome, and residual losses through shared model parameters in [ppo_optimization.py](../../src/pylatro_agent/training/ppo_optimization.py). Detaching the terminal value inside the residual target prevents a direct residual-loss gradient into that terminal prediction; it does not isolate the shared backbone from residual updates.

For the current reward definition, with zero intermediate rewards, gamma 1, and terminal reward 10 for a win and 0 otherwise, the desired conditional mean is exactly `V_pi(x) = 10 P_pi(win | x)`, where `x` must specify the information available to the predictor and the policy. A residual is not mathematically needed if that probability is correct. With approximation errors, it can help or hurt, which requires measurement.

## Ranked source findings and their limits

| Priority | Source and result relevant here | Application and boundary |
| --- | --- | --- |
| 1 | [Ilyas et al., ICLR 2020, §2.2](https://arxiv.org/pdf/1811.02553) measured PPO/TRPO value networks that fit GAE-derived supervised targets while remaining inaccurate against policy returns. They also measured gradient variance under different baselines. | Falling residual Huber loss need not mean better expected-return prediction. Evaluate target fitting, true-return prediction, and actor gradient quality separately. Their MuJoCo findings do not establish our cause. |
| 2 | [Cobbe et al., Phasic Policy Gradient, ICML 2021](https://proceedings.mlr.press/v139/cobbe21a/cobbe21a.pdf) separates policy and value optimization, allowing different sample reuse and reducing shared-objective interference. Its experiments find value learning can benefit from more reuse than policy learning. | Our two joint epochs do not establish that the critic is adequately fitted. A frozen-policy critic fit and gradient attribution are justified. PPG's particular architecture, epoch count, and Procgen improvements are not a prescription for Balatro. |
| 3 | [Feng and Wu, Journal of Approximation Theory 2022](https://arxiv.org/pdf/2009.12755) shows that fixed-threshold Huber risk consistency need not imply learning the conditional mean under asymmetric noise. | Our threshold-1 residual regression warrants an estimand audit. The theorem does not establish that it explains the observed positive residual, because our targets are bootstrapped GAE returns. |
| 4 | [Gneiting and Raftery, JASA 2007](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf) provides the proper-scoring foundation for evaluating predictive distributions; [Murphy, 1973](https://journals.ametsoc.org/view/journals/apme/12/4/1520-0450_1973_012_0595_anvpot_2_0_co_2.xml) separates Brier score into uncertainty, reliability, and resolution. | Matching average wins is useful evidence against a global offset, but a constant base-rate predictor can do that without distinguishing states. Report NLL/Brier, calibration by strata, and resolution relative to a base-rate predictor. |
| 5 | [Baisero and Amato, AAMAS 2022](https://www.ifaamas.org/Proceedings/aamas2022/pdfs/p44.pdf) analyzes bias caused by mismatching history-based policies with state-only critics under partial observability, and develops a history-state formulation. | Specify what the critic conditions on before calling a simulator continuation the ground truth. Our particular observation/history representation must be audited; this paper does not prove it is insufficient. |
| 6 | [Schulman et al., GAE, ICLR 2016, §3](https://arxiv.org/pdf/1506.02438) distinguishes an action-independent baseline from bootstrapping, and makes the approximation-dependent bias/variance tradeoff explicit. | A bad Monte Carlo baseline and bad bootstrap values have different effects. Full-episode Monte Carlo and short traces are useful diagnostic endpoints; neither is automatically the best training estimator. |
| 7 | [van Hasselt et al., 2018](https://arxiv.org/pdf/1812.02648) studies the interaction of approximation, bootstrapping, and off-policy sampling, primarily with replay-based DQN. | Do not diagnose the deadly triad from a plateau. Our newly collected cohorts, bounded outcome probabilities, and absence of runaway values do not resemble a demonstrated divergence case. The scalar residual still merits a fixed-point audit. |
| 8 | [Patterson, Liao, and White, TPAMI 2023](https://arxiv.org/pdf/2205.08464v2) distinguishes robust Bellman-error objectives from heuristic error clipping and derives algorithms for the former. | Robust loss is not inherently wrong, but a robust Bellman-error algorithm is not interchangeable with our supervised Huber fit to sampled GAE targets. |
| 9 | [Official explained-variance documentation](https://scikit-learn.org/1.6/modules/model_evaluation.html#explained-variance-score) notes that explained variance ignores systematic prediction offsets. | Report bias and MSE alongside explained variance, and state whether its target is GAE return or actual outcome. High explained variance against a self-generated target cannot settle calibration. |

## Direct implications of the loss definitions

### Long horizon does not inherently suppress the current outcome-NLL gradient

For a winning episode, the hazard parameterization gives `p_win = product_k sigmoid(z_k)` across active antes. Away from numerical clamps,

```
-log(p_win) = sum_k -log(sigmoid(z_k))
d NLL / d z_k = sigmoid(z_k) - 1
```

Thus the win product does not itself introduce exponentially small gradients through the logarithm. The complete outcome label directly supervises the prediction made at an early state. It is not propagated back through hundreds of simulated actions. Long horizons can still make the prediction harder through stochastic outcomes, hidden information, distant consequences, changing policies, and weak representations.

The actual implementation clamps selected outcome probabilities at `1e-7`. A CPU audit of stored sampled evaluation forecasts at +0/+20/+40/+60, both replicas, found **0 of 65,952 realized-outcome probabilities at or below that floor**. The endpoint minima were 0.008784, 0.011415, 0.003053, and 0.009090. This excludes that selected-probability clamp on these forecasts; it says nothing definitive about training minibatches or individual pre-sigmoid hazard saturation. The [machine-readable audit](data/critic_clamp_audit_20260916.json) preserves all eight input-file hashes and source hashes. Recreate it with `.venv/bin/python tools/research_critic_clamp_audit.py --output /tmp/critic_clamp_audit.json`; the [script](../../tools/research_critic_clamp_audit.py) validates the seed panel and rejects duplicate or incomplete game outcomes.

Reproduce that bounded audit:

```python
import json
from pathlib import Path

root = Path("eval/complete_cohort_long_20260915/evaluations")
for endpoint in (0, 20, 40, 60):
    count = clipped = 0
    minimum = 1.0
    for replica in (0, 1):
        path = root / f"update{endpoint}" / f"seed{replica}" / "sampled" / "outcomes.jsonl"
        for line in path.read_text().splitlines():
            row = json.loads(line)
            target = 8 if row["won"] else int(row["final_ante"]) - 1
            assert 0 <= target <= 8
            for forecast in row["critic_forecasts"]:
                probability = forecast["outcome_probabilities"][target]
                count += 1
                clipped += probability <= 1e-7
                minimum = min(minimum, probability)
    print(endpoint, count, clipped, minimum)
```

### Huber can optimize the wrong statistic for expected return

The following is an analytic example, not an explanation established on our run. Let `G = 10Y` and `P(Y=1)=0.4`. The expected return is 4. If a scalar prediction `v` minimizes threshold-1 Huber error against these complete returns, then for `0 <= v <= 1` its expected derivative is

```
0.6 v - 0.4 = 0
```

so its optimum is `v = 2/3`, rather than 4. With an exact terminal prediction of 4 and Huber-trained additive residual, the equivalent residual optimum is `-10/3`. This follows directly from the loss definition and illustrates the mean-versus-robust-location distinction. It is not a prediction of our observed residual's sign: actual residual targets are GAE returns, states differ in win probability, and shared approximation changes the solution.

Suitable isolation is a frozen-data comparison of the current residual objective, mean-squared regression to Monte Carlo returns, and the terminal probability alone. The first criterion is conditional-mean accuracy; gameplay requires a separate controlled intervention.

### Large outcome MSE is not entirely critic error

For a predictor `v(x)` and true win probability `p(x)`,

```
E[(10Y - v(x))^2 | x]
  = 100 p(x)(1-p(x)) + (10p(x) - v(x))^2
```

The first term is irreducible outcome noise given `x`. At a true probability of 0.4, even the perfect value 4 has expected MSE 24. An aggregate observed MSE around 20 does not, on its own, show a disastrously inaccurate conditional mean. The second term is what a better predictor can remove. The same distinction appears in binary Brier scores without the factor of 100.

Conversely, matching the average outcome is not enough. If every opening receives a forecast of 0.35 and 35% of games win, calibration in the large is perfect; the forecast still gives no information about differences among openings. Stratified calibration and resolution are required, with uncertainty reflecting distinct games rather than treating hundreds of correlated forecasts as independent labels.

## Diagnostics that distinguish the hypotheses

### First: a fixed-policy, fixed-data critic experiment

Freeze one actor including sampling temperature, mixture, legal-action distribution, observation version, and dropout mode. Keep its policy inference in an immutable model instance. Store complete episodes under that policy and split by game seed before selecting rows. No teacher labels are needed.

Use a separate critic copy to compare:

1. Existing features with only the critic heads trainable.
2. A critic copy whose representation can train, while the actor used to generate data remains frozen.
3. Matched target choices: complete categorical outcome NLL, scalar Monte Carlo MSE, and current GAE-residual Huber.

Keep sample ordering, distinct episodes, optimization counts, and validation states fixed for the comparisons. First establish that each objective can decrease on a small training subset. Then measure generalization to held-out game seeds. Report parameter deltas, per-head gradient norms, sigmoid saturation, and each loss separately; joint loss magnitudes alone do not quantify gradient interference.

If head-only fitting fails but the trainable critic copy succeeds, representation capacity or suitability is implicated. If both fit training outcomes but fail held-out seeds, generalization/data coverage is implicated. If neither fits a small subset, inspect objective/gradient/optimizer implementation before proposing larger architectures. If the fixed-policy critic learns well but the online critic does not, policy drift, optimization exposure, and interference become stronger candidates.

For interference, compute outcome-, residual-, and actor-gradient norms and pairwise cosines on the shared parameters before clipping. Measure how a critic-only step changes policy probabilities, and how an actor-only step changes predictions on a fixed bank. Negative cosine is a local diagnostic, not proof that separating networks improves gameplay. Account for the global gradient clip, Adam state, and shared learning rate.

### Second: repeated continuations, with the conditioning made explicit

Select snapshots by prespecified ante/phase and prediction bins, not by whether the original rollout eventually won. Preserve model-visible history, simulator state, RNG state, and checkpoint hash. Check snapshot isolation before using the result.

Repeating a frozen policy with different policy-sampling streams from one fixed simulator snapshot estimates outcome probability **conditional on that full hidden state and fixed simulator RNG**. It does not generally estimate the probability conditional only on the model's observation/history. A future deck order hidden from the model can make these quantities differ even if the observation-conditioned predictor is perfect.

Use these continuations for two purposes, labeled separately:

- Estimate within-snapshot action/rollout noise and identify large predictive discrepancies conditional on the full snapshot.
- Compare candidate first actions followed by the same frozen policy, with paired continuation streams and confidence intervals. This estimates action-value differences, which a scalar average calibration check cannot provide.

To claim irreducible error for model-visible information, average over a valid distribution of hidden states consistent with that information or use grouped held-out forecasts across independently sampled games. Arbitrarily replacing simulator RNG in the middle of a game is not necessarily a valid draw: hidden deck order, consumed random streams, and revealed information impose constraints. Preserve the distinction between missing-information error and approximation error if conditional hidden-state sampling is unavailable.

A small number of continuations produces wide binomial intervals near 50% win probability. Report that uncertainty, and cluster action-comparison results by snapshot and ancestor game. More policy repeats on one snapshot do not replace coverage across distinct game states.

### Third: assess the signal reaching the actor

On the same frozen trajectories, compare current fragmented GAE, full-episode GAE, and Monte Carlo return minus a detached baseline. Inspect differences by horizon, phase, ante, action family, and value-error strata. One realized winning or losing continuation is not an oracle advantage; sign disagreement with it is not an action-error rate.

Where repeated action continuations resolve action-value differences, assess whether the estimated advantages rank those actions correctly. Measure gradient agreement across independent episode subsets and against a higher-sample reference where feasible. This tests what PPO needs from the critic, beyond whether an opening forecast has the right average.

Only after separating target mismatch, fit capacity, uncertainty, and actor-signal quality should the next training intervention be selected. These notes support this research sequence; they do not validate residual removal, full Monte Carlo training, a larger critic, PPG, or renewed long PPO training.
