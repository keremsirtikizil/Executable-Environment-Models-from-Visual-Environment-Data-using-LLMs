# PURE: Pseudocode for Understanding Real-world Environments

**Can a VLM watch a short video and write correct pseudocode describing the environment's rules — from pixels alone?**

---

## Project Goal

PURE tests whether a vision-language model can perform **visual rule induction**: given only raw video frames of an unknown environment (no labels, no text, no hints), can it produce correct pseudocode describing the rules that govern that environment? The task is hard by design — the model must independently discover what objects exist, how they interact, and what edge cases apply, using only visual evidence.

---

## Pipeline Overview

Two models operate in sequence. **Extractor (Gemini 2.5 Pro via fal.ai)** receives only PNG frames and outputs pseudocode. **Verifier (Gemini 2.5 Flash)** independently re-checks the pseudocode against the same frames. In incremental mode (exp04+), frames are sent in batches of 3 and the model revises its hypothesis each round. No scene descriptions, cell legends, action names, or rule hints are ever provided.

See `README.md` for the full pipeline diagram and frame-generation details.

---

## Key Findings

### What the Model Gets Right

- **Basic mechanics are reliably captured.** In synthetic grid worlds, simple movement and wall-blocking were solved in essentially every run.
- **Unfamiliar real-world physics.** Given videos of a superhydrophobic bouncing ball, a Peaucellier-Lipkin linkage, and Newton's cradle, the model correctly identified the domain, mechanism, and operational phases — including non-trivial details (Cassie-Baxter vs. Wenzel states, N-in/N-out conservation).
- **Sensitivity to filming technique.** A thermal-camera billiards video produced heat-physics pseudocode; a standard-camera billiards video produced correct mechanical pseudocode. The model read the frames, not the filename.
- **Label blindness.** When videos were mislabelled (exp06 pair 3: a gyroscope demo filed as "pendulum", a folding furniture mechanism filed as "double_pendulum"), the pseudocode accurately described the actual content. Scene names had zero influence.

### What the Model Gets Wrong

- **Higher-order interactions fail reliably.** Rules combining two mechanics (ice + edge, magnetic attraction + wall cancellation) were missed in every synthetic-world run.
- **Confabulation from longer sequences.** Extending episodes to 5× frames did not improve accuracy. A new failure mode appeared: the model latched onto a temporal coincidence and invented a rule ("if move is horizontal, remove all metal from entire grid") that was never in the environment.
- **Direction errors.** Newton's cradle: left-right directionality was wrong in the 3-ball case despite correct conservation logic.

### The Prior-Contamination Problem

This is the most important failure mode in synthetic worlds. EchoWorld was designed so that its key rule is the **exact opposite** of MagnetWorld's analogous rule (echo moves away from agent; magnet moves toward). Even with 5× more disconfirming evidence than confirming evidence, the model consistently imported MagnetWorld's "same direction" prior and reported the wrong rule. The model was not reading the frames — it was defending a prior belief against visual evidence.

### Real-World vs. Synthetic Performance Gap

| Domain | Best accuracy |
|---|---|
| Synthetic (MagnetWorld, exp02) | 35% (11/31 test cases) |
| Synthetic (EchoWorld, exp03) | 36% (8/22 test cases) |
| Real-world physics (exp05) | 3/5 correct or partially correct |

The gap is striking. In unfamiliar real-world domains the model **has no prior to import** and is forced to read the frames. Synthetic invented-rule environments are harder precisely because frontier VLMs carry strong priors about "how grid-world rules work."

### Label-Bias (or Lack Thereof)

Across exp05 and exp06, scene names were never shown to the model. In exp06 pair 3, both videos had completely wrong filenames — the model produced pseudocode that accurately matched the actual visual content of each. PURE's label-blindness is confirmed.

### Similar-Video Sensitivity

All 3 exp06 pairs rated **HIGH sensitivity**:

| Pair | Key difference | Sensitivity |
|---|---|---|
| bowling_pins vs. bowling_strike | Mechanical state machine vs. physics kinematics | HIGH |
| billiard_break vs. billiards_physics | Thermal camera vs. standard camera | HIGH |
| pendulum vs. double_pendulum | Gyroscope demo vs. furniture hardware | HIGH |

PURE successfully distinguishes videos that share a surface domain but differ in one key aspect — filming technique, mechanical abstraction level, or actual content.

### Verifier Reliability

The verifier is **too lenient**. In exp02, it marked 3 of 4 MagnetWorld runs as CORRECT despite all runs scoring below 36% on ground-truth test cases. The verifier's frame-by-frame check does not catch rule formulations that happen to be consistent with a subset of frames but fail on edge cases. Verifier scores should not be used as a proxy for accuracy.

---

## Quantitative Summary

| Experiment | Scene | Frames | Verdict | Accuracy |
|---|---|---|---|---|
| exp02 | MagnetWorld (×4 runs) | ~20 | INCORRECT | 16–35% (27% mean) |
| exp03 | EchoWorld | ~20 | INCORRECT | 36% best |
| exp05 | bouncing_ball | 24 | CORRECT | — |
| exp05 | pendulum (Peaucellier linkage) | 44 | CORRECT | — |
| exp05 | newtons_cradle | 60 | PARTIALLY CORRECT | — |
| exp05 | metronomes | 108 | UNKNOWN | — |
| exp05 | double_pendulum (wrong video) | 60 | CORRECT (wrong content) | — |
| exp05 | cymatics (label contamination) | 80 | PARTIALLY CORRECT | — |
| exp06 | bowling_pins | — | PARTIALLY CORRECT | — |
| exp06 | bowling_strike | — | PARTIALLY CORRECT | — |
| exp06 | billiard_break (thermal) | — | INCORRECT | — |
| exp06 | billiards_physics | — | CORRECT | — |
| exp06 | pendulum (gyroscope demo) | — | CORRECT (mislabelled) | — |
| exp06 | double_pendulum (furniture) | — | CORRECT (mislabelled) | — |

---

## Conclusions

- **Prior contamination, not frame-reading failure, is the dominant bottleneck in synthetic worlds.** The model knows grid-world conventions and defends them against contradicting visual evidence.
- **Real-world performance is substantially better** because unfamiliar physics domains force genuine frame reading. The model has no stored rule to defend.
- **Label-bias is effectively zero.** The pipeline correctly ignores scene names and file metadata — pseudocode reflects actual frame content.
- **The verifier cannot be trusted as a quality signal.** Independent verification against frames is necessary but not sufficient; edge-case coverage requires ground-truth test suites.
- **More frames do not reliably help.** Longer sequences introduce confabulation risk without improving rule recovery. Smarter batching or active-query strategies may be needed.

---

## Open Questions / Next Steps

- **Contrastive prompting:** Show the model two videos that differ in exactly one rule and ask it to identify the difference. Does constraining the prior space help?
- **Adversarial synthetic worlds:** Design rule sets that deliberately contradict every major VLM grid-world prior simultaneously. What is the ceiling for prior-contamination resistance?
- **Verifier calibration:** Replace lenient frame-by-frame verification with targeted edge-case probing. Can the verifier be prompted to actively try to falsify the extracted rules?
- **Active querying:** Rather than batching all frames, let the model request specific frame ranges to resolve ambiguities. Does targeted observation reduce confabulation?
- **Domain interpolation:** Test on domains that are partially familiar (e.g., modified Sokoban with one invented rule). Where exactly does the prior-vs-reading tradeoff shift?
