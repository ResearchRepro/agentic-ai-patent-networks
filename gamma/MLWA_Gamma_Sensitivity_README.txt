MLWA R4: Reproducible Louvain resolution-sensitivity analysis

SOURCE
  GitHub_Update_Final.zip (author-supplied archival snapshot)
  12_completed_validation/repo_inputs/keyword_memberships.csv
  12_completed_validation/repo_inputs/per_doc_keywords.csv
  12_completed_validation/repo_inputs/org_edges.csv

RECONSTRUCTION
  Undirected keyword edges are pairs of distinct terms appearing within each
  patent row; edge weights count patent rows containing each pair. This exactly
  reproduces 5,957 nodes and 194,172 weighted edges from the release.
  Organization uses the 864-node, 1,347-edge released positive-degree graph.

PARAMETERS
  NetworkX weighted Louvain, fixed seed=0; gamma = 0.5, 0.75, 1, 1.25, 1.5.
  Optimized modularity is reported at each gamma (not directly comparable across
  different gamma values); a common-reference gamma=1 score is also reported.
  NMI is measured against this run's gamma=1 partition, on the same node set.
  Silhouette metric: cosine distance between weighted adjacency-row profiles.
  Keyword silhouette: the same 1,000 nodes, sampled without replacement with
  NumPy default_rng(20261112), for each gamma.
  Organization silhouette: all 864 positive-degree nodes.
  Original released keyword partition Q(gamma=1) = 0.1745254; the reoptimized
  one-seed gamma=1 result is Q=0.1744562 (separate partitions).

CAUTIONS
  The keyword silhouette coefficients are near zero and keyword membership
  changes materially across gamma; do not claim resolution-invariant keyword
  communities. Organization appears stable but is highly disconnected (262
  components), so its silhouette is not external semantic validation.
  This is resolution sensitivity under one fixed random seed; it does not
  estimate the full distribution of Louvain stochastic variability.
  Patent-text embedding silhouette coefficients in the manuscript use a
  different observation/feature space and are not directly comparable here.

REPRODUCE
  python -m pip install pandas numpy scipy scikit-learn networkx
  python MLWA_gamma_sensitivity_reproducible.py --archive GitHub_Update_Final.zip --outdir .

FILES
  MLWA_gamma_sensitivity_results.csv: five gamma values on each of two graphs
  MLWA_gamma_sensitivity_seed_runs.csv: the seed-0 runs
  MLWA_gamma_sensitivity_reproducible.py: self-contained reproduction code

The supplementary files need to be uploaded to the public repository or supplied
with the journal revision. Preparing them does not update GitHub automatically.
