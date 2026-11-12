#!/usr/bin/env python3
"""MLWA Reviewer 6 reproducibility supplement: patent cross-walk and applicant HHI.

Inputs: supplied GitHub_Update_Final.zip (as of 2026-10-05).
Outputs: reproducible, family-level exploratory analyses. No market or supply risk inferred.
"""
import argparse, csv, hashlib, io, json, math, os, random, sys, zipfile
from collections import defaultdict, Counter
from pathlib import Path

try:
    import numpy as np
    import networkx as nx
except ImportError as exc:
    sys.exit("Missing package. Install with: py -3 -m pip install numpy networkx\n" + str(exc))

N_SEEDS = [0, 1, 2, 3, 4]
SOURCE_ROOT = '12_completed_validation/'
VERSION = '1.0.0-2026-11-12'


def read_csv(z, filename):
    with z.open(SOURCE_ROOT + filename) as binary:
        with io.TextIOWrapper(binary, encoding='utf-8-sig', newline='') as handle:
            return list(csv.DictReader(handle))


def csv_output(path, headers, records):
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(records)


def write_json(path, obj):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)


def applicants_of(s):
    return sorted(set(x.strip() for x in (s or '').split(';;') if x.strip()))


def q_value_bh(pvalues):
    n = len(pvalues)
    by_p = sorted(range(n), key=lambda i: pvalues[i])
    out = np.ones(n)
    running = 1.0
    for r in range(n - 1, -1, -1):
        i = by_p[r]
        running = min(running, pvalues[i] * n / (r + 1))
        out[i] = running
    return out


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, a):
        if a not in self.parent:
            self.parent[a] = a
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            if a < b:
                self.parent[b] = a
            else:
                self.parent[a] = b


def assignee_hhi(families, label_fn=lambda x: x, weighted=False, technology=None):
    """HHI on 0..10,000; one family is one unit, allocated 1/k to its k applicants.
    weighted=True instead gives 1 unit PER applicant (full-counting sensitivity).
    technology: None for all families, or one 0..6 column in lexical profile.
    """
    counts = defaultdict(float)
    total = 0.
    families_n = 0
    for fam in families:
        apps = fam['apps']
        if not apps:
            continue
        mass = (1.0 if technology is None else float(fam['t'][technology]))
        if mass <= 0:
            continue
        families_n += 1
        w = mass if weighted else mass / len(apps)
        for a in apps:
            counts[label_fn(a)] += w
            total += w
    if total == 0:
        return dict(hhi=None, total_credit=0, families=0, distinct_applicants=0, top_assignee='', top_pct=None)
    shares = [c / total for c in counts.values()]
    top = max(counts, key=counts.get)
    return dict(hhi=10000 * sum(s*s for s in shares), total_credit=total,
                families=families_n, distinct_applicants=len(counts), top_assignee=top,
                top_pct=100*counts[top] / total)


def main():
    ap = argparse.ArgumentParser(description='Recompute fractional cross-walk and applicant HHI from the GitHub archive')
    ap.add_argument('--zip', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--permutations', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=20261112)
    args = ap.parse_args()
    if args.permutations < 99:
        sys.exit('At least 99 permutations required; use 1000 for paper reporting.')
    if not args.zip.is_file():
        sys.exit('ZIP not found: ' + str(args.zip))
    args.out.mkdir(parents=True, exist_ok=True)

    print('[1/6] Reading recorded cohort, keyword memberships and co-application edges ...', flush=True)
    raw_bytes = args.zip.read_bytes()
    zip_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    with zipfile.ZipFile(args.zip) as z:
        required = ['inputs/retained_7604_manifest.csv', 'inputs/lexical_profiles.csv',
                    'repo_inputs/org_edges.csv', 'inputs/entity_pairs_final_reference.csv']
        missing = [p for p in required if SOURCE_ROOT + p not in z.namelist()]
        if missing:
            sys.exit('Missing ZIP members: ' + repr(missing))
        manifest = read_csv(z, required[0]); profiles = read_csv(z, required[1]); edges = read_csv(z, required[2]); reviewed_pairs = read_csv(z, required[3])

    if len(manifest) != 7604 or len(set(x['family_id'] for x in manifest)) != 7604:
        raise ValueError('Expected 7,604 distinct source families; validate your archive.')
    profile_by_id = {p['family_id']:p for p in profiles}
    if len(profile_by_id) != 7604:
        raise ValueError('Expected lexical profiles for the full 7,604 source cohort.')
    cols = sorted((c for c in profiles[0] if c.startswith('community_')), key=lambda c:int(c.split('_')[1]))
    if len(cols) != 7:
        raise ValueError('Expected 7 source lexical communities, got '+str(cols))
    for p in profiles:
        if int(p['row_index']) < 0: raise ValueError('Invalid row index')

    print('[2/6] Reconstructing organizational partition (best weighted Louvain among seeds 0-4) ...', flush=True)
    G = nx.Graph()
    for e in sorted(edges, key=lambda e:(e['source'], e['target'])):
        a,b = e['source'].strip(),e['target'].strip(); w=float(e['weight'])
        if not a or not b or a == b or w <= 0: continue
        if G.has_edge(a,b): G[a][b]['weight'] += w
        else: G.add_edge(a,b,weight=w)
    if G.number_of_nodes() != 864 or G.number_of_edges() != 1347:
        raise ValueError(f'Organizational graph does not match expected 864 nodes, 1347 edges: {G.number_of_nodes()}, {G.number_of_edges()}')

    candidates = []
    for seed in N_SEEDS:
        partition = nx.community.louvain_communities(G, weight='weight', resolution=1.0, seed=seed)
        Q = nx.community.modularity(G, partition, weight='weight', resolution=1.0)
        candidates.append((Q, seed, partition))
    Q, best_seed, communities = max(candidates, key=lambda o:(o[0],-o[1]))
    communities = sorted(communities, key=lambda c:(-len(c), min(c)))
    name_to_louvain = {a:i+1 for i,c in enumerate(communities) for a in c}
    top8 = set(range(1, min(8,len(communities))+1))
    org_groups = ['LC%02d'%i for i in sorted(top8)] + ['OTHER_CONNECTED', 'NO_COAPPLICATION_NODE']
    g_index = {k:i for i,k in enumerate(org_groups)}
    catalog = []
    for i,c in enumerate(communities,1):
        degree_sum = sum(G.degree(a,weight='weight') for a in c)
        catalog.append([i, len(c),round(degree_sum,8), ' | '.join(sorted(c)[:10])])
    csv_output(args.out / '01_organization_community_catalog.csv',
               ['rank_by_node_count','node_count','sum_weighted_degree','first_10_members_alphabetical'], catalog)
    csv_output(args.out / '02_organization_member_labels.csv', ['applicant_label','rank_by_node_count','group_for_crosswalk'],
               [[a, i, 'LC%02d'%i if i in top8 else 'OTHER_CONNECTED'] for a,i in sorted(name_to_louvain.items())])

    print('[3/6] Constructing fractional patent-to-community flow, including single-applicant families ...', flush=True)
    families=[]
    bad_profiles=[]
    for row in manifest:
        f_id = row['family_id'];p = profile_by_id[f_id]
        if str(p['row_index']) != str(row['row_index']):
            raise ValueError('Family-to-profile row index mismatch: '+f_id)
        apps=applicants_of(row['applicants'])
        t=np.array([float(p[c]) for c in cols], dtype=np.float64)
        if np.any(t<0) or not np.isfinite(t).all():raise ValueError('Invalid lexical profile: '+f_id)
        if t.sum() > 0 and abs(t.sum()-1)>1e-6:bad_profiles.append(f_id)
        groups=Counter()
        for a in apps:
            c=name_to_louvain.get(a)
            group=('LC%02d'%c) if c in top8 else ('OTHER_CONNECTED' if c is not None else 'NO_COAPPLICATION_NODE')
            groups[group]+=1
        o=np.zeros(len(org_groups))
        if apps:
            for grp,count in groups.items():o[g_index[grp]]=count/len(apps)
        families.append(dict(id=f_id, apps=apps,t=t,o=o, multi=(len(apps)>1)))
    if bad_profiles:
        raise ValueError('Non-normalized lexical profiles, examples: '+repr(bad_profiles[:3]))
    # Include only families with both a nonempty technical profile and an applicant label.
    included = [f for f in families if f['t'].sum()>0 and f['apps']]
    T = np.vstack([f['t'] for f in included]); O=np.vstack([f['o'] for f in included]); n=len(included)
    if not np.allclose(T.sum(axis=1),1,atol=1e-7) or not np.allclose(O.sum(axis=1),1,atol=1e-7):
        raise ValueError('Family credit conservation failed; do not interpret results')
    observed=T.T @ O
    assert math.isclose(float(observed.sum()),n,rel_tol=1e-9)
    row_total=T.sum(axis=0); col_total=O.sum(axis=0)
    expected=np.outer(row_total,col_total)/n
    # stratified exchangeability: separate the single-applicant and multi-applicant subcohorts.
    multi=np.array([f['multi'] for f in included]); indices=[np.where(~multi)[0], np.where(multi)[0]]
    expected_strat=np.zeros_like(observed)
    for ix in indices:
        if len(ix):expected_strat += np.outer(T[ix].sum(axis=0),O[ix].sum(axis=0))/len(ix)
    mask=expected > 1e-10
    mask_s=expected_strat > 1e-10
    stat=lambda mat, exp, mk:float(np.sum(((mat[mk]-exp[mk])**2)/exp[mk]))
    chisq=stat(observed,expected,mask)
    chisq_strat=stat(observed,expected_strat,mask_s)

    print('[4/6] Running global and single/multi-applicant-stratified patent-label permutations ...', flush=True)
    rng=np.random.default_rng(args.seed)
    exceed_global=np.zeros_like(observed,dtype=np.int64)
    exceed_strat=np.zeros_like(observed,dtype=np.int64)
    stat_ge_global=0;stat_ge_strat=0
    null_rows=[]
    abs_obs=np.abs(observed-expected)
    abs_obs_s=np.abs(observed-expected_strat)
    for b in range(args.permutations):
        perm=rng.permutation(n)
        M=T[perm].T @ O
        stat_b=stat(M,expected,mask)
        exceed_global += (np.abs(M-expected)>=abs_obs-1e-10)
        stat_ge_global+= int(stat_b>=chisq-1e-10)
        perm2=np.arange(n)
        for ix in indices:
            perm2[ix]=rng.permutation(ix)
        M2=T[perm2].T @ O
        stat_b2=stat(M2,expected_strat,mask_s)
        exceed_strat += (np.abs(M2-expected_strat)>=abs_obs_s-1e-10)
        stat_ge_strat+= int(stat_b2>=chisq_strat-1e-10)
        null_rows.append([b+1,format(stat_b,'.9f'),format(stat_b2,'.9f')])
        if (b+1)%250==0: print(f'   completed {b+1}/{args.permutations}',flush=True)
    pglobal=(exceed_global+1)/(args.permutations+1)
    pstrat=(exceed_strat+1)/(args.permutations+1)
    qglobal=q_value_bh(pglobal.flatten()).reshape(pglobal.shape)
    qstrat=q_value_bh(pstrat.flatten()).reshape(pstrat.shape)
    csv_output(args.out / '03_crosswalk_null_distribution.csv', ['iteration','pearson_stat_global_null','pearson_stat_stratified_null'],null_rows)
    result_rows=[]
    for i,c in enumerate(cols):
        for j,g in enumerate(org_groups):
            result_rows.append([c,g,round(float(observed[i,j]),6),round(float(expected[i,j]),6),
                round(float(expected_strat[i,j]),6),round(float(100*observed[i,j]/row_total[i]),6),
                round(float(100*observed[i,j]/col_total[j]),6),
                round(float(pglobal[i,j]),6),round(float(qglobal[i,j]),6),
                round(float(pstrat[i,j]),6),round(float(qstrat[i,j]),6)])
    csv_output(args.out / '04_fractional_crosswalk_with_pvalues.csv',
               ['technology_source','organizational_group','observed_family_equivalents','expected_global','expected_stratified',
                'pct_organizational_group_within_technology','pct_technology_within_organizational_group',
                'two_sided_p_global','BH_q_global_70_cells','two_sided_p_stratified','BH_q_stratified_70_cells'],result_rows)

    print('[5/6] Computing original-label and reviewed-merge applicant HHIs ...', flush=True)
    uf=UnionFind()
    accepted=0
    for r in reviewed_pairs:
        if r.get('final_label','').strip().lower() == 'same entity':
            uf.join(r['name_a'].strip(),r['name_b'].strip());accepted+=1
    hhi_rows=[]
    for scope,tech in [('ALL_FAMILIES',None)]+[(c,i) for i,c in enumerate(cols)]:
        for method, func in [('original_applicant_labels',lambda s:s),('manually_reviewed_merges_only',uf.find)]:
            for counting in [False,True]:
                x=assignee_hhi(families,label_fn=func,weighted=counting,technology=tech)
                hhi_rows.append([scope, method, 'full_per_applicant' if counting else 'fractional_per_family',
                    x['families'],round(x['total_credit'],5),x['distinct_applicants'],
                    None if x['hhi'] is None else round(x['hhi'],6),x['top_assignee'],
                    None if x['top_pct'] is None else round(x['top_pct'],6)])
    csv_output(args.out / '05_assignee_patent_HHI.csv',
               ['patent_scope','identity_rule','credit_rule','families_with_nonzero_scope_mass','credit_mass',
                'number_applicant_labels','HHI_0_to_10000','top_applicant_label','top_applicant_share_pct'],hhi_rows)
    flow_summary=[]
    for j,g in enumerate(org_groups):
        count_fams=sum(f['o'][j]>0 for f in included)
        flow_summary.append([g,round(float(col_total[j]),6),count_fams])
    csv_output(args.out / '06_flow_coverage.csv', ['group','fractional_family_equivalents','number_families_touching_group'], flow_summary)

    qa={
        'script_version':VERSION,'archive_sha256':zip_sha256,'networkx_version':nx.__version__,'numpy_version':np.__version__,
        'input_families':len(families), 'families_with_applicant':sum(bool(f['apps']) for f in families),
        'families_with_lexical_profile':sum(bool(f['t'].sum()) for f in families),
        'crosswalk_included_families':n, 'crosswalk_skipped_families':len(families)-n,
        'crosswalk_single_applicant_families':int(np.sum(~multi)),
        'crosswalk_multi_applicant_families':int(np.sum(multi)),
        'families_with_more_than_one_organizational_group':sum(int((f['o']>0).sum()>1) for f in included),
        'families_with_more_than_one_technology_community':sum(int((f['t']>0).sum()>1) for f in included),
        'graph_nodes':G.number_of_nodes(),'graph_edges':G.number_of_edges(),
        'graph_components':nx.number_connected_components(G),
        'louvain_seed_scores': [{'seed':s,'weighted_modularity':q,'communities':len(cs)} for q,s,cs in candidates],
        'selected_seed':best_seed,'selected_modularity':Q,'total_organizational_communities':len(communities),
        'crosswalk_group_definition':'Eight largest by node count (not the authors\' earlier hand-picked eight); all others grouped; labels absent from coapplication graph in NO_COAPPLICATION_NODE',
        'technology_source_columns':cols,'total_fractional_flow':float(observed.sum()),
        'permutations_per_null':args.permutations,'seed':args.seed,
        'null_hypothesis_global':'Technology mixture vectors are exchangeable among eligible patent families, with fixed observed technology-vector distribution and fixed applicant-group vectors',
        'null_hypothesis_stratified':'Same exchangeability conditional on number of applicants: single-applicant vs multiple-applicant family',
        'crosswalk_statistic':'Pearson-form sum((observed-expected)^2/expected), calibrated by permutation, NOT theoretical chi-square p-values',
        'cell_pvalue':'Two-sided absolute observed-minus-expected tail; p=(b+1)/(B+1); BH adjusted over all 70 cells separately per null design',
        'global_stat_observed':chisq,'global_stat_p':(stat_ge_global+1)/(args.permutations+1),
        'stratified_stat_observed':chisq_strat,'stratified_stat_p':(stat_ge_strat+1)/(args.permutations+1),
        'reviewed_same_entity_pair_count':accepted,
        'caveat':'These tests permute patent families, not inventor groups or linked applicant portfolios. Repeated applicants may induce dependencies. They are exploratory conditional tests; no industrial market or supply concentration is measured.',
        'open_source_overlap':'8.5 percent cannot be established from this archive without the open-source project sample, matching identifiers and denominator.'
    }
    write_json(args.out / '00_QA_methods_summary.json',qa)
    with open(args.out / '07_README_interpretation.txt','w',encoding='utf-8') as f:
        f.write('MLWA Reviewer 6 – reproducibility results, source: GitHub_Update_Final.zip\n')
        f.write('SOURCE ZIP SHA256: '+zip_sha256+'\n')
        f.write('1. This is a NEW exploratory fractional cross-walk, not the previous Tableau cross-walk or its earlier p-values.\n')
        f.write('2. Patent-to-technology: source lexical profile fractions, not exclusive labels.\n')
        f.write('3. Patent-to-organization: equal 1/k credit for each distinct applicant, then aggregated to all coapplication Louvain communities; eight largest communities by node count are displayed, with other connected and no-coapplication labels retained.\n')
        f.write('4. One eligible family contributes exactly one patent-equivalent; same-patent applicants are NOT assumed to collaborate because of single-applicant filings. Coapplication edges remain separate evidence.\n')
        f.write('5. Both permutation nulls retain whole technology vectors, preserve all patent-group vectors, and use a Monte Carlo (+1)/(B+1) p-value; secondary null stratifies by single- versus multi-applicant families.\n')
        f.write('6. Individual cell tests are two-sided, adjusted using Benjamini-Hochberg across 70 combinations for each null. Global test uses observed cross-walk versus permutation distribution; no asymptotic chi-square p-value.\n')
        f.write('7. Applicant HHIs use patent credits over the available cohort, expressed 0–10000, original released labels and sensitivity using only 50 reviewed positive name pairs. Fractional credit sums to one per patent. Full counting is a sensitivity method and has a different denominator.\n')
        f.write('8. Patent-level HHI is NOT industrial market concentration and is NOT evidence of supply-chain vulnerability.\n')
        f.write('9. This analysis cannot validate the earlier 8.5% open-source overlap because open-source project sampling/matching files are absent. Do not cite the 8.5% as verified from this bundle.\n')
        f.write('10. Permutations do not fully account for repeated applicant portfolios, patent family relationships across group labels or applicant degree heterogeneity; interpret p-values conditionally.\n')
        f.write('11. Do not copy numerical results into a revised manuscript before author validation of the normalization and analysis scope.\n')
    print('[6/6] Checking flow conservation and writing final outputs ...',flush=True)
    if abs(float(observed.sum())-n)>1e-7: raise ValueError('Crosswalk mass did not conserve')
    print('OK. Cohort:', len(families),'Crosswalk:',n,'Louvain communities:',len(communities),
          'Global p:',qa['global_stat_p'],'Stratified p:',qa['stratified_stat_p'],flush=True)
    print('Results folder:',args.out.resolve(),flush=True)


if __name__=='__main__':
    main()
