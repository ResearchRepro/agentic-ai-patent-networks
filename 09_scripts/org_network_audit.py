"""Independent weighted Louvain and binary degree-preserving diagnostics.

Uses the supplied reciprocal edge spreadsheet; reciprocal weights are not added.
The randomized experiment is explicitly binary and does not preserve strengths.
"""
from pathlib import Path
from collections import defaultdict
import json, random, math, csv
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

def graph(edges,n):
    a=[defaultdict(float) for _ in range(n)]
    for u,v,w in edges:
        a[u][v]+=w
        if u!=v:a[v][u]+=w
    return a

def modularity(a,labels):
    degrees=[sum(row.values())+row.get(i,0) for i,row in enumerate(a)]
    m2=sum(degrees);tot=defaultdict(float);internal=defaultdict(float)
    for i,row in enumerate(a):
        c=labels[i];tot[c]+=degrees[i]
        for j,w in row.items():
            if labels[j]==c:internal[c]+=w*(2 if i==j else 1)
    return sum(internal[c]/m2-(k/m2)**2 for c,k in tot.items())

def louvain(a,seed):
    rng=random.Random(seed);original=a;members=[{i} for i in range(len(a))]
    for level in range(100):
        n=len(a);labels=list(range(n));deg=[sum(row.values())+row.get(i,0) for i,row in enumerate(a)]
        total=deg.copy();m2=sum(deg)
        for sweep in range(200):
            moved=0;order=list(range(n));rng.shuffle(order)
            for i in order:
                old=labels[i];ki=deg[i];total[old]-=ki
                kin=defaultdict(float)
                for j,w in a[i].items():
                    if j!=i:kin[labels[j]]+=w
                best=old;gain=kin[old]-ki*total[old]/m2
                for c,weight in kin.items():
                    g=weight-ki*total[c]/m2
                    if g>gain+1e-12:best=c;gain=g
                labels[i]=best;total[best]+=ki;moved+=best!=old
            if moved==0:break
        else:raise RuntimeError('Local optimization did not converge')
        cs=sorted(set(labels));mapping={c:i for i,c in enumerate(cs)}
        newmembers=[set() for _ in cs]
        for i,c in enumerate(labels):newmembers[mapping[c]].update(members[i])
        if len(cs)==n:break
        aggregate=defaultdict(float)
        for i,row in enumerate(a):
            for j,w in row.items():
                if j<i:continue
                u,v=sorted((mapping[labels[i]],mapping[labels[j]]))
                aggregate[u,v]+=w
        a=graph([(u,v,w) for (u,v),w in aggregate.items()],len(cs));members=newmembers
    final=[0]*len(original)
    for c,m in enumerate(members):
        for i in m:final[i]=c
    return final,modularity(original,final)

def swapped(edges,n,seed,swaps):
    rng=random.Random(seed);e=[(u,v) for u,v,w in edges];present=set(e);accepted=0;attempts=0
    while accepted<swaps and attempts<swaps*200:
        attempts+=1;i,j=rng.sample(range(len(e)),2);u,v=e[i];x,y=e[j]
        if rng.random()<.5:u,v=v,u
        if rng.random()<.5:x,y=y,x
        if len({u,v,x,y})<4:continue
        p=tuple(sorted((u,y)));q=tuple(sorted((x,v)))
        if p in present or q in present:continue
        present.remove(e[i]);present.remove(e[j]);present.update([p,q]);e[i]=p;e[j]=q;accepted+=1
    if accepted<swaps:raise RuntimeError('Insufficient accepted swaps')
    before=np.bincount([u for u,v,w in edges]+[v for u,v,w in edges],minlength=n)
    after=np.bincount([u for u,v in e]+[v for u,v in e],minlength=n)
    assert np.array_equal(before,after)
    return [(u,v,1.) for u,v in e],attempts

def run(source,out,repetitions=100):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(source);nodes=sorted(set(d.source)|set(d.target));ids={v:i for i,v in enumerate(nodes)}
    pairs={}
    for r in d.itertuples(index=False):
        p=tuple(sorted((ids[r.source],ids[r.target])))
        if p in pairs:assert pairs[p]==float(r.weight)
        pairs[p]=float(r.weight)
    edges=[(u,v,w) for (u,v),w in sorted(pairs.items())];n=len(nodes)
    sparse=coo_matrix((np.ones(len(d)),([ids[x] for x in d.source],[ids[x] for x in d.target])),shape=(n,n)).tocsr()
    nc,comp=connected_components(sparse,directed=False);sizes=np.bincount(comp);gc=np.flatnonzero(comp==np.argmax(sizes));gcids={v:i for i,v in enumerate(gc)}
    gec=[(gcids[u],gcids[v],w) for u,v,w in edges if u in gcids and v in gcids]
    weighted=graph(edges,n);binary=graph([(u,v,1.) for u,v,w in edges],n)
    result={'nodes':n,'unique_undirected_edges':len(edges),'components':int(nc),'largest_component_nodes':len(gc),'largest_component_edges':len(gec),'largest_component_node_fraction':len(gc)/n,'component_sizes_descending':sorted(map(int,sizes),reverse=True)}
    labels,q=max((louvain(weighted,seed) for seed in range(5)),key=lambda x:x[1])
    gl,gq=max((louvain(graph(gec,len(gc)),seed) for seed in range(5)),key=lambda x:x[1])
    result.update(independent_weighted_modularity=q,independent_weighted_communities=len(set(labels)),weighted_louvain_seeds=[0,1,2,3,4],weighted_component_partition_modularity=modularity(weighted,list(comp)),giant_component_weighted_modularity=gq,giant_component_communities=len(set(gl)))
    observed=max(louvain(binary,seed)[1] for seed in range(5));null=[];attempts=[]
    for i in range(repetitions):
        randomized,attempt=swapped(edges,n,20261002+i,20*len(edges));attempts.append(attempt)
        rg=graph(randomized,n);null.append(max(louvain(rg,seed)[1] for seed in range(5)))
    values=np.array(null)
    result['binary_degree_preserving_null']={'observed_best_of_5_modularity':observed,'random_graphs':repetitions,'accepted_swaps_each':20*len(edges),'seeds_start':20261002,'louvain_seeds':[0,1,2,3,4],'null_mean':float(values.mean()),'null_sd':float(values.std(ddof=1)),'null_min':float(values.min()),'null_max':float(values.max()),'empirical_upper_tail_p':float((1+sum(values>=observed))/(repetitions+1)),'z_score':float((observed-values.mean())/values.std(ddof=1)),'preserves':'node degree sequence and edge count; not components or weighted strengths'}
    pd.DataFrame({'replicate':range(repetitions),'modularity':null,'swap_attempts':attempts}).to_csv(out/'org_binary_null_replicates.csv',index=False)
    pd.DataFrame({'applicant':nodes,'connected_component':comp,'independent_louvain_community':labels}).to_csv(out/'org_independent_memberships.csv',index=False)
    pd.DataFrame([(nodes[u],nodes[v],w) for u,v,w in edges],columns=['source','target','weight']).to_csv(out/'org_edges.csv',index=False)
    (out/'org_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='component_sizes_descending'},indent=2))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--source',default='astra_inputs/org org.csv');p.add_argument('--out',default='new_analysis');p.add_argument('--replicates',type=int,default=100);x=p.parse_args();run(x.source,x.out,x.replicates)
