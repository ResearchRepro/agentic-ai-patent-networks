# Agentic AI patent networks

Supporting data and independently reproducible network diagnostics for manuscript MLWA-D-26-00551.

## Completed data reconstruction

The supplied keyword spreadsheet contains reciprocal representations of an undirected graph. Restricting it to the supplied seven-community membership list produces **5,957 nodes and 194,172 unique undirected edges**. Reciprocal weights are checked for equality and counted once. Literal keyword values such as `null` must be retained as strings when reading the input; default missing-value conversion silently changes the graph.

The organizational spreadsheet produces **864 nodes and 1,347 unique undirected edges**. Its **262 connected components** and independently optimized weighted modularity **0.972482**, with **265 communities**, reproduce the reported fragmented organization network. Independently computed community labels are not the historical Gephi label IDs. The eight selected historical groups contain **134 nodes (15.51% of the 864 connected nodes)** and **498 edges (36.97% of 1,347 edges)**. This is node/edge coverage, not patent-family coverage.

## Completed organizational validation

The largest connected component contains **41 nodes and 125 edges**, accounting for **4.745%** of connected applicant nodes. Its independently optimized weighted modularity is **0.575737**. Thus the historical claim of 159 nodes / 18.4% must be reconciled before submission.

Both the complete organization graph and its largest component have been compared with **100 binary degree-preserving randomized graphs**. Each graph uses 20 accepted double-edge swaps per edge; five identical Louvain seeds are used for observed and randomized graphs. For the complete binary graph, observed Q is **0.969958**, versus null mean **0.623873**. For the largest-component binary graph, observed Q is **0.595136**, versus null mean **0.286304**. The finite Monte Carlo upper-tail probability is **1/101** in each comparison. The null preserves binary degrees and edge count; it does not preserve component structure or weighted strengths. These experiments support topology diagnostics, not semantic accuracy, entity matching accuracy, or superiority to other methods.

## Keyword diagnostic reconciliation

The supplied seven-community partition has weighted modularity **0.174525**, not 0.751. The reconstructed selected graph has **1 connected component**, average weighted degree **105.091153**, mean finite unweighted path length **2.477218**, and mean local clustering **0.712497**. The organization graph's average weighted degree is **3.8125**, and mean local clustering is **0.499413**. These values use one copy of each undirected edge, and the clustering mean includes degree-zero/one nodes as zero. Keep these documented definitions when updating the manuscript.

## Directory contents

- `00_input_snapshot`: the original Lens CSV. This is an input snapshot, not an asserted final 7,584-family manifest.
- `01_raw` to `05_top15`: archived processing outputs, copied without modification.
- `06_edges`: one row per undirected edge, with checked weights.
- `07_communities`: final keyword memberships, selected historical organization memberships, and clearly identified independently reconstructed organizational memberships.
- `08_query`: the query specification transcribed from manuscript Appendix A; it is not an API retrieval implementation.
- `09_scripts`: diagnostic scripts and a prospective Colab notebook.
- `10_validation`: actual completed diagnostics, null replicates, and an unlabelled prospective entity-pair review sample.
- `11_manual_audit`: author-confirmed aggregates for the 200-record manual audit. The completed record-level ledger still needs to be supplied before it is claimed as deposited.

## Reproduce completed diagnostics

Install `09_scripts/requirements_analysis.txt`, then run:

```sh
python 09_scripts/org_network_audit.py --source 06_edges/org_edges.csv --out reproduced_org --replicates 100
```

The implementation is an independent two-phase weighted Louvain implementation at resolution 1.0, based on Blondel et al. (2008), https://arxiv.org/abs/0803.0476. Double-edge rewiring is described at https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.swap.double_edge_swap.html. Randomized graphs are diagnostic samples, not asserted uniform draws from every graph with the given degrees. Input hashes, parameter choices, and full replicate outputs are retained.

## Remaining inputs and execution

1. Exact retained-family manifest: `07_communities/retained_families.csv`, columns `family_id,row_index`, 7,584 unique families and representative source-row indices. The archive alone does not establish which records were excluded or consolidated. No alternative cohort has been substituted.
2. Complete record-level manual relevance ledger matching the author-confirmed 200-case counts (strict 58/87/55; broad 114/74/12).
3. Human labels for `10_validation/entity_audit/entity_pairs_to_review.csv`. Enter YES, NO, or UNCERTAIN and a reason. The sample uses raw applicant names, explicit normalized Levenshtein similarity, and two threshold strata. It is prospective and is not the historical normalization decision log.
4. Original processing code / historical identity-decision log, if available. Reconstructed scripts are labelled as such rather than represented as the original pipeline.

SBERT execution and fractional IPC/CPC tables require the exact retained-family manifest. The Colab notebook checks this prerequisite before analysis, records the model revision and token truncation, and exports its metrics and all class-table rows. No semantic-baseline results or entity-resolution precision have been reported as completed.
