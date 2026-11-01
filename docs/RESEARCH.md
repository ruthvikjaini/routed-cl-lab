# Research notes — 2026-11-01 (Day 8)

## Trending-first check (before any paper research)

GitHub trending, daily and weekly, checked 2026-11-01 ~00:00 ET:

- Daily top: thedotmack/claude-mem (persistent agent context),
  mattpocock/skills, Donchitos/Claude-Code-Game-Studios, pingdotgg/t3code,
  pbakaus/impeccable, obra/superpowers, lightpanda-io/browser,
  HKUDS/DeepTutor (lifelong personalized tutoring), Orkas-AI/Orkas.
- Weekly top: Nutlope/hallmark, cloudflare/security-audit-skill
  (55,456 stars, 3,614 forks, 11,908 stars this week),
  AgriciDaniel/claude-seo, pbakaus/impeccable, obra/superpowers,
  TauricResearch/TradingAgents, thedotmack/claude-mem, anthropics/skills,
  affaan-m/ECC.
- Acceleration report (whatstrending.ai, 2026-11-01): morluto/rea fastest
  at 5,259 stars/day (+36,813 in a week); thedotmack/claude-mem biggest
  acceleration (254 -> 882 stars/day).

Verdict: same as Days 5–7. The trending list is dominated by
prompt/skill packs, agent harnesses, and full platforms (DeepTutor,
TradingAgents, OpenBot). None has a benchmarkable code weakness worth
optimizing: skill packs are prose, not code; the platforms are whole
products, not a measurable component. No trending repo was picked.
Fell back to the paper-based original project, per the standing rule.

## Primary source (verified)

Philip, Balakrishnan, Geraghty, Mao, Venkat, Mironov, "Routed Continual
Learning", arXiv:2610.38796, submitted 28 Oct 2026, FAIR at Meta.
Verified on the arXiv abstract page and full HTML on 2026-11-01.

Key facts used (all from the paper, not secondary summaries):

- Architecture: past trajectories grouped into clusters; each cluster
  maintains state with one memory system from a fixed library of five;
  before each run an LLM router reuses an existing cluster or creates a
  new one, using the problem statement, memory-system descriptions, and
  cluster summaries. Context Memory Adapter mediates between the memory
  system and the agentic loop.
- Five memory systems (Table 2): ICL (ordered visible trajectories);
  ReasoningBank (strategies induced from self-judged successful/failed
  trajectories); Hierarchical Regression (repeated keyed scalar
  outcomes); Hierarchical Discrete-Event State (monotone event-free
  curves over ordered horizons); Persistent-Entity State (noisy
  measurement sets with uncertain entity correspondence).
- Evaluation: 398-instance heterogeneous stream — 301 instances from the
  six CL-Bench tasks + 97 banking instances from tau-bench — interleaved
  preserving per-task order; four language models (Gemini 3.7 Flash,
  GPT-5.6 Sol, Opus 4.7, Opus 5); five matched rollouts.
- Staged conditions: Stateless ICL, Global ICL, Routed ICL, Routed CL,
  plus Ground-truth assignment (privileged diagnostic). ICL histories
  use a 100K-token context budget, stream-wide for Global ICL and per
  cluster for the routed conditions.
- Results: stateless-reference gain 0.015 (Global ICL) -> 0.222 (Routed
  ICL) -> 0.280 (Routed CL); normalized reward +15–30% of the maximum
  attainable over Global ICL; estimated API cost -65–72% vs Global ICL;
  Routed CL on the reward–cost Pareto frontier for every model.
  Table 1 normalized rewards: Gemini -0.030 / 0.211 / 0.213;
  GPT-5.6 Sol 0.168 / 0.387 / 0.464; Opus 4.7 0.092 / 0.209 / 0.243;
  Opus 5 0.169 / 0.340 / 0.401 (Global / Routed ICL / Routed CL).
  Ground truth exceeds Routed CL by 0.008–0.071. Router agreement with
  ground truth: 86% same system, 91% same partition, 79% both.
- Appendix C.2: routing ReasoningBank gives no consistent reward gain
  and raises mean cost 8–29%; ICL benefits from routing because it
  exposes an entire cluster history, while RB already retrieves
  selectively by request similarity.
- Open question (Section 7 / future work): runs are sequential; agents
  updating the same memory cluster at once, stale reads, and cross-model
  sharing are untested. This is what the concurrency extension in this
  repo tests.

## Other candidates considered (checked 2026-11-01)

- Zhu et al., "Don't Lose the Thread: Empowering Long-Horizon LLM
  Agents with Cognitive Resource Self-Allocation" (CORAL,
  arXiv:2610.40350, 29 Oct 2026, COLM 2026): checkpoint verified facts,
  reset cluttered context; GAIA 42.8% with an 8B model. Rejected as the
  primary build: its headline result depends on SFT + a multi-episode
  ARPO training loop, which cannot be reproduced honestly in one night
  on CPU; the checkpoint/reset mechanism alone is a thinner build than
  the routed-memory architecture, and it overlaps the Day 2–3
  controller/harness theme. Held as a possible later night.
- TraceQuery (diffusion LMs, arXiv surfacing 31 Oct 2026), RAQ-MLA
  (quantization for MLA caches), FG-ConvNO, GeoLift: noted, not
  buildable as CPU-only agent-learning testbeds in one night.
- LRMoE (MoE training system, 31 Oct 2026): needs real Qwen3-30B /
  GLM4.5-scale training runs; rejected on the CPU-runnability rule.
- Barron softmax fix (arXiv:2610.31642, held since Day 6): still held;
  needs a bf16 training harness, out of scope for one night.

## Design decisions and assumptions (all documented in code)

- Simulation, not a real model: the solver is a closed-form success
  probability; the router is a similarity scorer with Gaussian noise.
  No LLM calls anywhere. The paper's numbers are quoted in the README
  for comparison only; every measured number in the repo comes from
  this testbed.
- Gain/interference constants (agent.py): GAIN_LOGIT 2.2,
  INTERFERENCE_LOGIT 1.5. Chosen so memory can move a middling instance
  from ~0.4 to ~0.8 and a fully diluted context can pull it below its
  stateless level — the qualitative regime the paper reports. Not fitted
  to the paper's magnitudes.
- Fidelity matrix (agent.py): generic systems (ICL, RB) 0.75 on their
  home schema, 0.50 elsewhere; structured systems 0.95 on their home
  schema, 0.05 elsewhere. The paper's premise: no single system is
  optimal across applications.
- Token sizes (memory.py): 400 per ICL trajectory, 60 per RB item,
  40 per structured retrieval; ICL budget 100K tokens per history, as
  in the paper. Price schedule frozen 2026-11-01: $3.00 / 1M input,
  $0.30 / 1M cached, $15.00 / 1M output.
- Router: score = 0.7 * cosine(instance, cluster summary) + 0.3 *
  noisy system-fit estimate + noise; new cluster below threshold 0.58;
  max 14 clusters. Cluster summaries are the mean of member skill
  vectors.
- Structured-system maturity: 1 - exp(-n/8) over same-schema
  observations in the cluster (see "Bugs found" below for why it is
  not per-key).
- Stateless reference: expected stateless reward per family by
  fixed-seed Monte Carlo (seed 12345, n=20000) over the difficulty
  distribution — so the stateless condition's normalized reward is ~0
  by construction, as in the paper.
- Concurrency: four agents, separate streams, one shared routed-cl
  memory; local snapshot views refreshed every `staleness` steps.
  Appends and count updates commute, so write conflicts are not
  modelled; stale reads are.

## Bugs found and fixed during the night

1. Router under-clustering (threshold 0.45 + summary from system
   state). First run: routed conditions formed only 3–5 clusters for
   7 families; partition agreement 0.71; Routed ICL gain 0.151 and
   Routed CL 0.358, with Global RB (0.472) beating Routed CL — a
   qualitative divergence from the paper. Root causes: (a) the create
   threshold sat below the cross-family similarity floor, so new
   families were merged into existing clusters; (b) cluster summaries
   came from the memory system's stored state, which is empty until
   the system has observed its schema, so early routing compared
   against a neutral 0.5. Fixes: threshold 0.45 -> 0.58 (between the
   cross-family ceiling ~0.45 and same-family floor ~0.75) and
   summaries from member skill vectors. After the fix: 7 clusters,
   partition agreement 0.989, Routed ICL gain 0.362, Routed CL 0.613.
   The pre-fix numbers were discarded, not published.
2. Structured maturity was per-key (1 - exp(-n_key/4)). With 12 keys
   per family, each key was seen ~4 times, so specialists capped at
   ~0.5 relevant mass and lost to generic ICL/RB even on their home
   schema — contradicting the paper's premise that the structured
   systems are better there. A regression/curve/entity model improves
   with every same-schema observation, not only repeats of one key, so
   maturity is now 1 - exp(-n_schema/8). After the fix, the structured
   families are where Routed CL gains over Routed ICL (sales 0.66 vs
   0.24, cohort 0.69 vs 0.03).
3. Test bug (not code): the PES test probed a random cohort instance
   whose entity key had not been observed; fixed by picking the most
   common key and then asserting the unseen-key/seen-key contrast.
4. Test bug (not code): the blocked-stream test asserted an absolute
   Global ICL gain > 0.10; the honest control for the paper's CL-Bench
   observation is separate homogeneous streams (fresh history per
   family), not one blocked stream — a blocked stream still
   accumulates cross-family history. Added `run_isolated` and kept the
   blocked test as a relative comparison (blocked > interleaved).

5. Nondeterministic tie-break in router agreement (found by the
   reproducibility check, not by tests). `_record_agreement` picked the
   majority family with `max(set(...), key=count)`; set iteration order
   depends on per-process hash randomization, so agreement rates
   differed in the third decimal between identical runs. Fixed with
   `Counter.most_common`, which breaks ties by first-seen order. After
   the fix, both result files are byte-identical across separate
   processes. The fix moved a few agreement rates by <= 0.01 and no
   reward or cost figure.

## Reproducibility

- Every RNG is seeded; the full grid reruns byte-identically
  (verified by rerunning run_experiment.py and diffing measured.json).
- 48/48 pytest pass in a fresh venv from a clean copy.
- `scripts/crossfoot.py` recomputes every aggregate in measured.json
  from the raw per-rollout rows in experiment.json and checks the
  README's headline figures against measured.json.
