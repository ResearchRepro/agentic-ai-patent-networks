# Agentic AI Patent Networks

Supporting data and processing outputs for the manuscript:
"A Dual-Layer Semantic and Organizational Patent-Mining Framework for
Mapping Patent-Documented Innovation around Agentic Artificial
Intelligence" (Machine Learning with Applications, submission
MLWA-D-26-00551).

## Repository contents

| Folder | Contents |
|---|---|
| `00_input_snapshot` | [توضیح دقیق: چه چیزی، از کجا، چه تاریخی] |
| `01_raw` | Raw Lens.org export fields retained for this analysis |
| `02_normalized` | Deduplicated / normalized patent-family records |
| `03_tokens_pos` | POS-tagged tokenized text used for keyword extraction |
| `04_ngrams` | Candidate unigram–trigram terms per document |
| `05_top15` | Final top-15 document-level keyword lists |
| `06_edges` | Weighted edge lists for the keyword co-occurrence network and the applicant co-application network |
| `07_communities` | Louvain community membership per node (keyword network and organizational network), resolution = [مقدار] |
| `08_query` | Exact executable Lens.org Boolean search expression (matches Appendix A of the manuscript) and retrieval filters/date range |
| `09_scripts` | Processing scripts used to generate the above outputs (see "Reproducibility" below) |

## Data source and license

Underlying patent records were retrieved from Lens.org
(https://www.lens.org) on [تاریخ بازیابی، مثلاً August 2026]. Redistribution
of raw Lens.org records is subject to Lens.org's terms of use; this
repository therefore shares [derived/processed outputs only — بگویید دقیقاً چه چیزی
از خام قابل‌انتشار است و چه چیزی نیست].

## What this repository does NOT yet include

- An independent manual relevance-validation sample (planned; see
  manuscript Section 5.5).
- An embedding-based semantic baseline comparison (planned; see
  manuscript Section 5.5).
- A logged, independently reviewed entity-resolution adjudication set
  (planned; see manuscript Section 5.5).
