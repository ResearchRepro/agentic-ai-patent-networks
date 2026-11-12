"""Reproducible sensitivity of weighted patent keyword and applicant networks.
Inputs: author-supplied GitHub_Update_Final.zip
No fabricated crosswalk, HHI, or open-source-overlap observations.
"""
from __future__ import annotations
import csv, io, itertools, zipfile
from collections import Counter
from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import normalized_mutual_info_score, silhouette_score

import argparse
parser = argparse.ArgumentParser(description="Recompute gamma sensitivity from author replication archive")
parser.add_argument('--archive', default='GitHub_Update_Final.zip', help='Path to supplied ZIP archive')
parser.add_argument('--outdir', default='.', help='Directory for CSV output')
args = parser.parse_args()
ROOT = Path(args.outdir).resolve()
ROOT.mkdir(parents=True, exist_ok=True)
ZIP = Path(args.archive).resolve()
PREFIX = '12_completed_validation/'
GAMMAS = (0.5, 0.75, 1., 1.25, 1.5)
SEEDS = (0,)
SAMPLE_SEED = 20261112
SAMPLE_SIZE = 1000

with zipfile.ZipFile(ZIP) as archive:
    def input_csv(name):
        return pd.read_csv(io.BytesIO(archive.read(PREFIX+name)), keep_default_na=False)
    p = input_csv('repo_inputs/per_doc_keywords.csv')
    memberships = input_csv('repo_inputs/keyword_memberships.csv')
    edges = input_csv('repo_inputs/org_edges.csv')

terms = set(memberships.key)
c = Counter()
for _, group in p.groupby('row_index', sort=False):
    names = sorted(set(group.keyword) & terms)
    c.update(itertools.combinations(names, 2))
keyword_graph = nx.Graph()
keyword_graph.add_nodes_from(memberships.key)
for (a,b), weight in c.items():
    keyword_graph.add_edge(a,b,weight=int(weight))
assert len(keyword_graph)==5957 and keyword_graph.number_of_edges()==194172
org_graph = nx.Graph()
for r in edges.itertuples(index=False):
    org_graph.add_edge(r.source,r.target,weight=float(r.weight))
assert len(org_graph)==864 and org_graph.number_of_edges()==1347

allrows=[]; seedrows=[]
for label,G in [('keyword',keyword_graph),('organization',org_graph)]:
    nodes=list(G.nodes)
    node_idx={v:i for i,v in enumerate(nodes)}
    A=nx.to_scipy_sparse_array(G,nodelist=nodes,weight='weight',dtype=float,format='csr')
    A=sparse.csr_matrix(A)
    rng=np.random.default_rng(SAMPLE_SEED)
    subset=np.sort(rng.choice(len(nodes), size=min(SAMPLE_SIZE,len(nodes)),replace=False))
    X=A[subset]
    best={}
    for gamma in GAMMAS:
        choices=[]
        for seed in SEEDS:
            coms=nx.community.louvain_communities(G, weight='weight',resolution=gamma,seed=seed)
            q=nx.community.modularity(G,coms,weight='weight',resolution=gamma)
            seedrows.append(dict(network=label,gamma=gamma,seed=seed,communities=len(coms),modularity_at_gamma=q))
            choices.append((q,seed,coms))
        q,seed,coms=max(choices,key=lambda x:(x[0],-x[1]))
        y=np.full(len(nodes),-1,dtype=int)
        for k,group in enumerate(coms):
            for name in group:y[node_idx[name]]=k
        assert (y>=0).all()
        classes=len(set(y[subset]))
        if 1 < classes < len(subset):
            sil=float(silhouette_score(X,y[subset],metric='cosine'))
        else:
            sil=np.nan
        q_fixed=float(nx.community.modularity(G,coms,weight='weight',resolution=1.0))
        best[gamma]=dict(gamma=gamma, network=label, n_nodes=len(nodes),n_edges=G.number_of_edges(),
           best_seed=seed,communities=len(coms), modularity_at_gamma=q,
           modularity_fixed_gamma_1=q_fixed, cosine_neighborhood_silhouette=sil,
           silhouette_sample_size=len(subset),silhouette_communities_present=classes,
           singleton_communities=sum(len(g)==1 for g in coms),
           nmi_vs_gamma_1=np.nan, labels=y)
        print(label, 'gamma',gamma,'comms',len(coms),'Qgamma',round(q,5),'Qfixed',round(q_fixed,5), 'sil',round(sil,5),flush=True)
    ref=best[1.]['labels']
    for gamma,r in best.items():
        r['nmi_vs_gamma_1']=float(normalized_mutual_info_score(ref,r['labels']))
        del r['labels']
        allrows.append(r)

out1=ROOT/'MLWA_gamma_sensitivity_results.csv'
out2=ROOT/'MLWA_gamma_sensitivity_seed_runs.csv'
pd.DataFrame(allrows).to_csv(out1,index=False,float_format='%.8f')
pd.DataFrame(seedrows).to_csv(out2,index=False,float_format='%.8f')
print('WROTE',out1, out2)
