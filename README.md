# routed-cl-lab

Routed continual learning for LLM agents: reproduce and extend
Philip et al., "Routed Continual Learning" (arXiv:2610.38796, 28 Oct 2026),
on a synthetic heterogeneous task stream. No LLM calls, no GPU, NumPy only.

The paper's question: when agents deployed at scale keep meeting related
problems, how should they retain what they learn? A single global history
preserves everything but gets expensive and mixes together experience that
does not belong together; compressing it into one fixed representation is
cheap but throws away detail. Routed Continual Learning treats this as a
routing problem: past trajectories are grouped into clusters, each cluster
keeps state in one memory system chosen from a fixed library, and a router
decides per instance whether to reuse a cluster or open a new one.

This repo implements that architecture end to end on a controlled
testbed — the five memory systems, the router, the four staged
conditions plus a privileged ground-truth diagnostic — and measures
reward and API cost on matched interleaved streams. It then extends the
paper with the experiment the paper leaves open: several agents sharing
one routed memory concurrently, reading through stale snapshots.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
python scripts/demo.py            # one rollout, all conditions
python scripts/run_experiment.py  # full grid (5 rollouts) -> results/
python scripts/crossfoot.py       # recompute every README number
```

## The stream

398 instances per rollout: 301 from six analytic task families
(text-to-SQL, poker, sales prediction, codebase adaptation, cohort
analysis, spectrum) plus 97 multi-turn banking-policy dialogues, in the
same 301 + 97 proportions as the paper. Instances are interleaved in
random order while preserving each family's canonical order. Each
instance has a latent skill vector, a difficulty, and a schema affinity
— the memory system the paper's ground-truth assignment pairs with that
family (SQL/code -> ICL, poker/banking -> ReasoningBank,
sales -> hierarchical regression, spectrum -> discrete-event state,
cohort -> persistent-entity state).

The solver is a simulated agent: success probability rises with
*relevant* retrieved mass (experience matching the instance, weighted by
the memory system's fidelity for the instance's schema) and falls with
*irrelevant* mass (other families' experience visible in the same
context). That dilution is the mechanism behind the headline result.

## The five memory systems (paper Table 2)

| System | Retained state | Cost per instance | Best on |
| --- | --- | --- | --- |
| ICL | ordered visible trajectories (100K-token budget) | up to 100K tokens, mostly cached | SQL, code |
| ReasoningBank | up to 3 strategy items per trajectory, retrieved by similarity | 3 x 60 tokens | poker, banking |
| Hierarchical regression | keyed scalar outcomes | 40 tokens | sales |
| Discrete-event state | event-free curves over horizons | 40 tokens | spectrum |
| Persistent-entity state | noisy measurement sets | 40 tokens | cohort |

Costs use a frozen price schedule (2026-11-01): $3.00 / 1M input
tokens, $0.30 / 1M cached input tokens, $15.00 / 1M output tokens.
Every instance pays the same 800-token prompt and 400-token response;
routed conditions add one router call. See `src/routed_cl_lab/costs.py`.

## Measured results

Five matched rollouts, interleaved streams, 398 instances each.
Normalized reward maps the external stateless reference to 0 and the
task maximum to 1; stateless-reference gain compares each method with
Stateless ICL on the same rollout. All numbers are simulation numbers
from this testbed, cross-footed from `results/experiment.json` by
`scripts/crossfoot.py`. Paper numbers are quoted separately below for
comparison — they are not these numbers.

| Condition | Norm. reward | Stateless gain | Cost / stream | vs Global ICL |
| --- | --- | --- | --- | --- |
| Stateless ICL | -0.024 | 0.000 | $3.34 | 72% cheaper |
| Global ICL | -0.092 | -0.065 | $11.95 | — |
| Global RB | 0.472 | 0.487 | $3.44 | 71% cheaper |
| Routed ICL | 0.348 | 0.362 | $6.32 | 47% cheaper |
| Routed RB | 0.471 | 0.487 | $4.45 | 63% cheaper |
| **Routed CL** | **0.604** | **0.613** | **$4.84** | **60% cheaper** |
| Ground truth (privileged) | 0.614 | 0.623 | $3.79 | 68% cheaper |

Reading the headline: Routed CL reaches 0.604 mean normalized reward
against -0.092 for Global ICL, at $4.84 per stream against $11.95 for
Global ICL — a 60% cost reduction. Its stateless-reference gain is
0.613 against -0.065 for Global ICL: a single global history gains
essentially nothing (slightly negative here) once tasks are interleaved,
while routing recovers the gain. Routed CL sits 0.010 below the
privileged ground-truth assignment (0.614), so the router leaves almost
no headroom on this testbed. The router agrees with ground truth on the
system choice 99.5% of the time, the partition 98.9%, and both 98.8% —
higher than the paper's 86% / 91% / 79%, because the synthetic skill
space is cleaner than real problem statements (see "What is simulated"
below).

Two findings reproduce the paper's subtler results:

- **Routing ReasoningBank adds nothing.** Routed RB (0.471) is
  indistinguishable from Global RB (0.472) and costs more ($4.45 vs
  $3.44). RB already retrieves selectively by similarity, so
  partitioning its store buys nothing — the paper's Appendix C.2 result.
- **System selection is where the extra gain lives.** Routed CL (0.604)
  beats Routed ICL (0.348) by 0.256, and the gap concentrates on the
  three structured-schema families: sales 0.66 vs 0.24, cohort 0.69 vs
  0.03, spectrum 0.61 vs 0.39. ICL alone cannot represent keyed scalars,
  event curves, or entity measurements.

## Ablations

**Blocked vs interleaved.** When families arrive in homogeneous blocks
instead of interleaved, Global ICL's gain rises from -0.065 to 0.046 —
blocking helps, but the history still accumulates cross-family
contamination, so it stays far below Routed CL (0.610). Routing makes
order nearly irrelevant: Routed CL gains 0.613 interleaved, 0.610 blocked.

**Isolated streams (the CL-Bench control).** Each family evaluated
separately with a fresh history, Global ICL's gain is 0.490 — naive ICL
is strong on homogeneous streams, exactly the paper's starting
observation. Interleaving is what breaks it (gain -0.065).

**Router noise.** The router's similarity and fit estimates get
Gaussian noise with standard deviation sigma:

| sigma | Norm. reward | Gain | Router agreement (both) | Clusters |
| --- | --- | --- | --- | --- |
| 0.00 | 0.615 | 0.624 | 0.974 | 6.8 |
| 0.10 | 0.604 | 0.613 | 0.988 | 7.0 |
| 0.20 | 0.437 | 0.451 | 0.725 | 9.6 |
| 0.40 | 0.197 | 0.214 | 0.292 | 9.2 |

Reward collapses once the router can no longer tell families apart:
at sigma 0.40 it opens spurious clusters and misroutes a third of the
stream. Router quality is the binding constraint on the whole
architecture.

## Extension: concurrent agents sharing one routed memory

The paper evaluates sequential rollouts only and notes that
simultaneous updates and stale reads are untested. Here four agents run
separate interleaved streams against one shared routed-cl memory,
reading it through local snapshots refreshed every `staleness` steps
(0 = always fresh). Sequential Routed CL is the baseline: 0.604
normalized reward at $4.84 per agent-stream.

| Staleness (steps) | Norm. reward | Cost / agent |
| --- | --- | --- |
| 0 (fresh) | 0.671 | $16.40 |
| 1 | 0.671 | $16.40 |
| 4 | 0.664 | $7.99 |
| 16 | 0.658 | $6.02 |

Sharing pools experience, so concurrent agents beat the sequential
baseline (0.671 vs 0.604) — but the shared ICL histories grow fourfold,
which is what makes staleness 0 expensive ($16.40). Staleness is a
cheap knob: at 16 steps, reward falls from 0.671 to 0.658 while cost
falls 63% to $6.02, because stale snapshots still contain most of the useful
history and the compact structured state barely goes stale at all. The
cost figures are per agent over a full 398-instance stream.

## How this compares with the paper

| Check | Paper (arXiv:2610.38796) | This simulation |
| --- | --- | --- |
| Global ICL gain, interleaved | +0.015 | -0.065 |
| Routed ICL gain, interleaved | +0.222 | 0.362 |
| Routed CL gain, interleaved | +0.280 | 0.613 |
| Routed CL cost vs Global ICL | -65% to -72% | -60% |
| Routed CL reward, Table 1 | 0.213–0.464 across models | 0.604 |
| Ground truth above Routed CL | +0.008 to +0.071 | +0.010 |
| Routing RB vs Global RB | no consistent gain, +8–29% cost | no gain, +29% cost |
| Router agreement (system / partition / both) | 86% / 91% / 79% | 99.5% / 98.9% / 98.8% |

The ordering matches on every check; the magnitudes differ. That is
expected: the simulated solver's gain (2.2 logit) and interference (1.5
logit) constants are modelling choices, not fitted to the paper, and the
synthetic skill space separates families more cleanly than real problem
statements do, so the router agrees with ground truth more often. The
value of the testbed is the mechanism — dilution, selective retrieval,
system selection, router quality — not the absolute numbers.

## What is simulated, and what is not

- The solver is a closed-form success probability, not a language
  model. There are no LLM calls anywhere; the "router" is a
  similarity scorer with calibrated noise, not an LLM judge.
- Skill vectors, difficulties, and outcomes are synthetic; no real
  benchmark items are used. Family names mirror the paper's source
  tasks (CL-Bench's six tasks plus tau-bench banking) so the structure
  is comparable, but no CL-Bench or tau-bench data is involved.
- Token counts are modelled (400 per trajectory, 60 per RB item,
  40 per structured retrieval) under a frozen generic price schedule,
  not measured from a provider bill.
- The concurrency extension models stale reads via snapshot refresh;
  it does not model write conflicts (appends and count updates
  commute here) or network latency.

## Scale-up path

The simulated solver is isolated behind `agent.solve` and the
`MemorySystem` interface: to run the same experiment with a real model,
replace `solve` with an LLM call that receives the retrieved context,
and replace the router's similarity scorer with an LLM router call
(the paper uses the evaluated model for every model-mediated role).
The structured systems already depend only on typed records, as in the
paper's Context Memory Adapter. Expect the absolute numbers to change;
the checks in "How this compares with the paper" are the ones to
re-run. A GPU is not needed for this testbed; the real-model version
wants one API key, not a training run.

## Layout

```
src/routed_cl_lab/
  stream.py      heterogeneous streams, families, stateless reference
  agent.py       simulated solver, fidelity matrix
  memory.py      the five memory systems + clusters
  router.py      similarity router + ground-truth diagnostic
  costs.py       frozen price schedule, token accounting
  metrics.py     normalized reward, gains, Pareto frontier
  experiment.py  staged conditions, rollouts, concurrency extension
scripts/
  demo.py            one rollout, printed
  run_experiment.py  full grid -> results/measured.json
  crossfoot.py       recompute aggregates, check the README
docs/RESEARCH.md     trending check, sources, design decisions
results/
  measured.json      aggregates (single source of truth for the README)
  experiment.json    every per-rollout condition result
```

## Sources

- Philip, Balakrishnan, Geraghty, Mao, Venkat, Mironov, "Routed
  Continual Learning", arXiv:2610.38796, 28 Oct 2026.
  https://arxiv.org/abs/2610.38796
- Asawa et al., "CL-Bench" (continual learning benchmark, six tasks),
  and Shi et al., "tau-bench" (banking domain), as cited therein.
- Ouyang et al., "ReasoningBank", as cited therein.
