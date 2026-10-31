# Multilingual LLM-as-a-Judge Reliability: thesis code

Code for the bachelor's thesis *Multilingual LLM-as-a-Judge Reliability: Diagnosis, Selective Verification, and Targeted Adaptation* by Çağla Ece Azizoğlu (Technical University of Munich, 2026). Package identifier **TC-20261025-v1**.

The thesis evaluates Gemma 3 4B as a judge of factual attribution against mFACE human ratings in 45 languages, tests claim-level verification combined with the holistic score, and studies two Korean adaptations (DPO preference training and a supervised NLI verifier).

## Repository structure

| Path | Contents |
|---|---|
| `supplementary_controls/` | CPU analysis of the saved scores: post-hoc controls and trained-verifier controls, with their protocol and saved summary results |
| `verify_reproduction.py` | Checks reproduced values against the evidence archive and the thesis tables |
| `historical/` | Scripts of the original GPU experiments (development judges, Korean DPO, transfer checks, supervised NLI verifier, bridge, additional-article evaluation) |
| `readable/` | Formatted reading copies of the original Python files |
| `prompts/` | Exact runtime prompts |
| `figures/` | Figure generator |
| `CODE_MAP.json`, `CODE_MANIFEST.json`, `FORMATTING_MAP.json` | Provenance and checksums |

## Distinguish the deliverables

- This code package contains selected recovered scripts, protocols and exact prompts, without research datasets, model weights or credentials.
- **Thesis_Evidence.zip**, identifier **TE-20261024-v1** (it extracts to the folder `Thesis_Evidence_2026-10-24_v1`), is the separate evidence archive containing saved outputs, selected inputs, reference scores and a hash manifest. It was submitted with the thesis and is not included in this repository. The reproduction commands below need it.
- The thesis source/PDF carries the scientific argument. Original endpoints, prior exposure and post-hoc status are not changed by publishing code.

## Tested CPU reproduction

Extract the evidence archive alongside this directory. The saved results were computed with Python 3.12.14 and requirements_cpu.txt; on 28 October 2026 all commands below were rerun with Python 3.13.17, NumPy 2.5.3, pandas 3.0.5 and SciPy 1.18.1, and every reported value reproduced exactly (only the recorded version stamps differ). From this code-package root:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements_cpu.txt
.venv/bin/python supplementary_controls/analyze_controls.py --data-root ../Thesis_Evidence_2026-10-24_v1/supplementary_controls/data --out runs/reproduction_NEW
```

Choose a new output directory to preserve existing results. This reconstructs H, SV, SV-NLI and matched-base scores, fits the additive offset only on development data, and reproduces the post-hoc controls, original primary ranking intervals, multiplicity sensitivity and within-language summaries. The protocol hash and input hashes are checked. This is saved-output reproduction, not fresh GPU inference or independent semantic validation.

For full table/figure verification, also install matplotlib and PyMuPDF and extract the companion manuscript source. Then run:

```sh
.venv/bin/python verify_reproduction.py --evidence-root ../Thesis_Evidence_2026-10-24_v1 --reproduced runs/reproduction_NEW --manuscript ../Thesis_Final_Overleaf --out runs/verification_NEW
```

The verifier checks all manifest-listed evidence files, reproduced metrics, the additional-sample table, both panels of the control table, the failure-handling table, the within-language table and the contrast figure (Figure 6). Software version stamps are reported but not required to match. Table numbers can change; stable source labels identify them. PDF plot timestamps are ignored; rendered pixels are compared.

### Trained-verifier controls (added 26 October 2026)

Post-hoc controls for SV-NLI: unconditional averaging with the trained verifier and a development-only additive offset for SV-NLI and for that average (thesis Table 19, lower panel, Section 7.4). Extract `results_v1/FINAL_ROWS.jsonl` from `archival/Trigger_Bridge_Verified_Results_2026-10-20.zip` in the evidence archive, then run:

```sh
.venv/bin/python supplementary_controls/analyze_nli_controls.py \
  --additional-rows ../Thesis_Evidence_2026-10-24_v1/supplementary_controls/data/additional_rows \
  --additional-labels ../Thesis_Evidence_2026-10-24_v1/supplementary_controls/data/additional_labels.csv \
  --development ../Thesis_Evidence_2026-10-24_v1/supplementary_controls/data/development900.csv \
  --bridge-rows path/to/FINAL_ROWS.jsonl \
  --out runs/nli_controls_NEW
```

The saved summary is `supplementary_controls/results_nli_v1/RESULTS.json`. The per-row score file is not included here because it contains mFACE human ratings; the command above recreates it. The script first checks that it reproduces the saved development SV-NLI scores. It uses the same language-cluster bootstrap (2,000 draws, seed 20261024) as the other controls.

### Failure-handling and scale checks (added 28 October 2026)

Two exploratory checks added after all other results were known (thesis Section 7.4 and Table 20; Section 4.3). The first recomputes the four main Spearman contrasts when failed pipeline outputs are not scored as zero, with the same bootstrap and seeds as the original analyses; the second tabulates how much of the holistic over-scoring falls at each judge score:

```sh
.venv/bin/python supplementary_controls/failure_handling_sensitivity.py --evidence-root ../Thesis_Evidence_2026-10-24_v1 --out runs/failure_NEW
.venv/bin/python supplementary_controls/scale_mismatch_check.py --evidence-root ../Thesis_Evidence_2026-10-24_v1 --out runs/scale_NEW
```

Saved summaries: `supplementary_controls/results_failure_v1/RESULTS.json` and `supplementary_controls/results_scale_v1/RESULTS.json`.

### Released NLI comparator and computation summary (added 30 October 2026)

Two further post-hoc analyses (thesis Section 7.2). The first compares the rankings with the NLI-based factuality score that the mFACE release provides for every example; it needs your copy of the release spreadsheet, which is not redistributed here, and checks every selected row against the archived records. The second counts generation calls, generated characters and recorded wall time from the archived per-row records of the additional run:

```sh
.venv/bin/python supplementary_controls/released_nli_comparator.py --release <mface_annotated_data.csv> --evidence-root ../Thesis_Evidence_2026-10-24_v1 --out runs/released_nli_NEW
.venv/bin/python supplementary_controls/computation_cost_summary.py --evidence-root ../Thesis_Evidence_2026-10-24_v1 --out runs/cost_NEW
```

Saved summaries: `supplementary_controls/results_released_nli_v1/RESULTS.json` and `supplementary_controls/results_cost_v1/RESULTS.json`.

### Result figures and supplementary documentation (added 29 October 2026)

`figures/make_result_figures.py` draws thesis Figures 2, 3 and 5 from the saved records of the evidence archive:

```sh
.venv/bin/python figures/make_result_figures.py --evidence-root ../Thesis_Evidence_2026-10-24_v1 --out runs/figures_NEW
```

`Supplementary_Technical_Documentation.pdf` reproduces, unchanged, the full technical appendix of the thesis before it was condensed: runtime and parser details, generation-limit diagnostics, the complete Korean preference-experiment tables, the verifier, bridge and repair protocols, and the qualitative case records.

## Readable source copies

For reviewing the implementation, start with readable/, which mirrors the 40 original Python files with consistent formatting. The later `supplementary_controls/analyze_nli_controls.py` is already written in readable form and has no separate copy. The originals are unchanged and remain the execution entry points. FORMATTING_MAP.json records original/formatted hashes and full AST-equality checks. The copies deliberately do not duplicate research data or frozen protocols.

## Historical GPU code

The frozen experiment plans (`EXPERIMENT_PLAN.md`) are part of the evidence archive submitted with the thesis and are not included in this repository. The `PROTOCOL.json` files still list them by checksum, so the historical runners need the archived packages to run.

historical/ preserves recovered development judges, Korean DPO final train/evaluation, Korean transfer, supervised NLI, downstream bridge and additional-article scripts. CODE_MAP.json records exact source locations and hashes, plus syntax-only checks. GPU execution is **not** certified by a syntax check. Original cluster paths, expected data layouts, gated Gemma access, adapter weights and recorded environments must be supplied/configured. Do not launch GPU inference on a cluster login node. The initial DPO and subsequent supervised NLI adapters are distinct. Some original complete runner chains or exploratory scripts were not recovered; no completeness claim is made.

Historical scripts retain their original terminology and path strings. In particular, archived wording 'pre-registered' is not proof of external registration. The manuscript's qualified interpretation and protocols govern the claims. Do not change frozen scoring choices or reinterpret a rerun as a new confirmatory result.

## Models, AI assistance and source attribution

The experimental model is Google's google/gemma-3-4b-it. The early runs did not pin a model revision, so their exact revision is unknown. The follow-up runs pin revision 093f9f388b31de276ce2de164bdc2081324b9767, and a re-verification under this pinned revision gives nearly identical scores.

Section 3.1 of the thesis describes the AI-assisted steps of the method, and Appendix I declares the AI tools and their purposes. ChatGPT (OpenAI; GPT-5.6) proposed refinements to the error taxonomy, gave a second reading of audit cases, and helped construct and check the Korean controlled preference pairs; these constructed labels are kept separate from the published human KLUE annotations. Claude (Anthropic) helped with the translation audit, the English-likeness heuristic and the translation-repair procedure.

The author specified the analyses, including the metrics, statistical tests and comparisons to be performed. Following these specifications, Claude and Codex (OpenAI) wrote and debugged the Python code, and ChatGPT provided programming assistance for the KLUE transfer test. For the KLUE transfer test, ChatGPT also suggested prompts and design choices, which the author reviewed before deciding on them. The author reviewed the code and checked its outputs, interpreted the results and decided on the conclusions.

ChatGPT used GPT-5.6 and Codex used GPT-5.6 Sol. Claude was used as Claude Sonnet in earlier phases and as Claude Opus 5.5 in the final phase. The author directed the research, made the final decisions, and reviewed and revised all adopted text, code and documentation.

The thesis uses mFACE (Aharoni et al., 2023), XL-Sum (Hasan et al., 2021) and KLUE (Park et al., 2021). The original publications and releases keep their attribution and licences, and model weights and third-party components retain their original terms.

## Licence

The code is released under the MIT licence (see `LICENSE`). Data and third-party material keep their own licences; see `LICENSING.md`.

## Methods and sources

Each script implements methods from the publications and software below. Full references for the publications are in the thesis bibliography. The Python `hashlib` and matplotlib entries refer to software documentation and are not cited in the thesis.

| Script(s) | What it implements | Sources |
|---|---|---|
| `historical/development/run_gemma_fullsource_q1.py`, `_q2.py`, `_q3.py` | Criterion-specific LLM-as-a-judge scoring of mFACE summaries with Gemma 3 4B and its native chat template | Gemma Team (2025); Wolf et al. (2020); Liu et al. (2023); Zheng et al. (2023); Aharoni et al. (2023) |
| `historical/development/run_gemma_fullsource_q2_selfeval.py`, `_taxaudit.py` | Prompt interventions: self-evaluation and taxonomy-aware prompts | Pagnoni et al. (2021); Hada et al. (2024) |
| `historical/development/run_gemma_fullsource_q2_hybrid.py` | Earlier combination of holistic and claim scores, compared with the final selective rule | Fabbri et al. (2022) |
| `historical/development/run_structured_claim_verification_dev10.py`, `..._translated_dev10.py` | Claim extraction and per-claim verification against the source, in the original language and after translation to English | Min et al. (2023); Fabbri et al. (2022); Artetxe et al. (2023) |
| `historical/development/annotate_q2_error50.py` | Interface for the manual audit of severe over-scoring; categories compared with FRANK | Pagnoni et al. (2021) |
| `historical/korean_dpo/train_korean_xlsum_FINAL_DPO.py` | DPO with LoRA adapters, AdamW optimizer, on Korean XL-Sum near-miss pairs | Rafailov et al. (2023); Hu et al. (2022); Loshchilov and Hutter (2019); Paszke et al. (2019); Hasan et al. (2021) |
| `historical/korean_dpo/eval_korean_xlsum_FINAL_TEST50.py` | Strict generation and label-logprob evaluation in both candidate orders; exact McNemar test; paired bootstrap | McNemar (1947); Efron and Tibshirani (1993); Zheng et al. (2023) |
| `historical/korean_extension/*`, `historical/Korean_Extension_Verified_Results_2026-10-19/*` | Chat-serialization robustness and non-Korean transfer of the DPO adapter; paired language-cluster bootstrap | Efron and Tibshirani (1993); Spearman (1904) |
| `historical/KLUE_Verified_Results_2026-10-19/*` | Pairwise transfer test on human-labelled KLUE-NLI with a premise-omission control; exact McNemar test; paired bootstrap | Park et al. (2021); McNemar (1947); Efron and Tibshirani (1993) |
| `historical/Korean_Verifier_Verified_Results_2026-10-19/*` (incl. `verifier.py`) | LoRA classifier on KLUE-NLI (supervised three-label NLI) with AdamW; premise-cluster bootstrap | Park et al. (2021); Hu et al. (2022); Hugging Face PEFT documentation; Loshchilov and Hutter (2019); Efron and Tibshirani (1993) |
| `historical/Trigger_Bridge_Verified_Results_2026-10-20/*` | NLI-based claim verification inside the selective 75/50 rule; language-cluster bootstrap | Laban et al. (2022); Efron and Tibshirani (1993) |
| `historical/additional_articles/*` | Fixed-system evaluation on additional mFACE articles: sampling, translation, extraction, both verifiers and analysis | Aharoni et al. (2023); Hasan et al. (2021); Min et al. (2023); Efron and Tibshirani (1993) |
| `supplementary_controls/analyze_controls.py` | Post-hoc controls (unconditional mean, development-fitted offset), Spearman/Kendall/Pearson, language-cluster percentile bootstrap, Bonferroni-style sensitivity | Spearman (1904); Kendall (1938, 1945); Pearson (1895); Efron and Tibshirani (1993); Virtanen et al. (2020); Harris et al. (2020); McKinney (2010) |
| `supplementary_controls/analyze_nli_controls.py` | Trained-verifier controls: unconditional averaging vs the selective trigger, development-only additive calibration with clipping, language-cluster percentile bootstrap | Efron and Tibshirani (1993); Spearman (1904); Virtanen et al. (2020); Harris et al. (2020); McKinney (2010) |
| `verify_reproduction.py` | Recomputation of reported values and SHA-256 integrity checks | Python `hashlib` documentation (software) |
| `figures/make_figures.py` | Pipeline diagram and contrast plot | matplotlib (software) |

All model scripts use PyTorch (Paszke et al., 2019) and Hugging Face Transformers (Wolf et al., 2020).

## Consistency with the manuscript

The controlled Korean DPO study and the later supervised three-label NLI verifier are separate adaptations. The DPO result does not establish general multilingual scoring improvement. In the additional-article evaluation, SV is the original selective four-label pipeline, SV-NLI replaces its verifier with the supervised adapter, and SV-NLI-base uses the untrained base in the same three-label branch. Historical names such as original4_selective and trained3_selective refer to these implementations; they do not name additional methods. The global offset control is fitted only on the recorded development scores. The supplementary controls remain post hoc, and the additional sample is not described as wholly unexposed.

Documentation update: 27 October 2026. The two SLURM scripts in `historical/korean_dpo/` differ from the archived originals only in their working-directory line, which now uses `$HOME` instead of a personal cluster path; `CODE_MAP.json` records both checksums. All other scripts, prompts and protocols are unchanged.

## Claim-score aggregation and parser edge cases

The thesis reports the proportional and capped claim scores (Chapter 5) and the parsing fallbacks that actually occurred (Appendix B.3). The development verification scripts implement further behaviour that no reported result depends on:

- **Strict aggregation (implemented, not used in the thesis).** Rules apply in order: an empty claim list gives 0; if every claim is fully supported, 100; if every claim is unsupported or contradicted, 0; otherwise any contradiction gives 25, then any unsupported claim 50, then any partially supported claim 75, and anything else 0.
- **Extraction edge cases.** An extraction output that cannot be parsed, has no claims field, or has a claims field that is not a list falls back to an empty list. Object types are not fully validated, so a parsed value that is not an object can raise an exception.
- **Verification edge cases.** An unparseable response becomes `unsupported` with empty evidence and a reason marking the JSON as invalid; a missing label also becomes `unsupported`. Labels are stripped and lowercased before aggregation. An unrecognized label receives zero weight but does not necessarily trigger the caps tied to the named labels. The reported development outputs contain no missing or unrecognized labels.

## Historical rerun cautions

Some historical runners require more than data and weights: protocol manifests and prompt files must also be restored at their original runtime-relative locations. For example, historical/additional_articles expects data/HOLDOUT_MANIFEST.json and prompts/nli_ko.txt, translation.txt, claim_extraction.txt and claim_verification.txt beside its runner. Supervised NLI expects DATA_MANIFEST.json and its split files beside verifier.py. Use the archived experiment package and its protocol hashes to restore the exact resources; do not substitute similarly named prompts.

The historical sampling script writes fixed output filenames without an overwrite refusal. Run it only in a new working directory with the recorded input paths. Existing frozen scripts are preserved rather than silently patched. Historical parser type assumptions and the manuscript's declared fallback policies remain unchanged.
