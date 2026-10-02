"""Run the prospective SBERT baseline and complete fractional IPC/CPC tables.

Requires the author's exact retained-family manifest. No cohort is inferred.
The majority keyword label is a diagnostic comparator, not a replacement for
the manuscript's nonexclusive keyword communities.
"""
from pathlib import Path
import argparse, json, hashlib, re, importlib.metadata
import numpy as np
import pandas as pd

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run(root,model_name):
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
    root=Path(root);out=root/'10_validation'/'semantic';out.mkdir(parents=True,exist_ok=True)
    manifest_path=root/'07_communities'/'retained_families.csv'
    if not manifest_path.exists():raise FileNotFoundError('Provide the exact retained_families.csv: family_id and row_index, one representative row per retained family.')
    cohort=pd.read_csv(manifest_path,keep_default_na=False)
    assert {'family_id','row_index'}<=set(cohort)
    assert len(cohort)==7584 and cohort.family_id.nunique()==7584 and cohort.row_index.nunique()==7584
    raw_path=root/'01_raw'/'raw_snapshot.csv';raw=pd.read_csv(raw_path,keep_default_na=False)
    keys=pd.read_csv(root/'05_top15'/'per_doc_keywords.csv',keep_default_na=False)
    community=pd.read_csv(root/'07_communities'/'keyword_memberships.csv',keep_default_na=False)
    mapping=dict(zip(community['key'],community.community));columns=sorted(set(mapping.values()))
    assert len(columns)==7
    pos={x:i for i,x in enumerate(columns)};rowpos={x:i for i,x in enumerate(cohort.row_index)}
    counts=np.zeros((len(cohort),len(columns)),dtype=float)
    for r in keys.itertuples(index=False):
        if r.row_index in rowpos and r.keyword in mapping:counts[rowpos[r.row_index],pos[mapping[r.keyword]]]+=1
    total=counts.sum(axis=1);valid=total>0;fraction=np.divide(counts,total[:,None],out=np.zeros_like(counts),where=total[:,None]>0)
    dominant=counts.argmax(axis=1);ties=(counts==counts.max(axis=1,keepdims=True)).sum(axis=1)>1
    profiles=pd.DataFrame(fraction,columns=[f'fraction_community_{x}' for x in columns]);profiles.insert(0,'row_index',cohort.row_index);profiles.insert(0,'family_id',cohort.family_id)
    profiles['retained_keyword_count']=total;profiles['majority_tie']=ties;profiles['has_profile']=valid
    profiles.to_csv(out/'family_keyword_profiles.csv',index=False)
    docs=cohort.merge(raw,on='row_index',how='left',validate='one_to_one',indicator=True)
    assert docs['_merge'].eq('both').all()
    text=(docs.Title.astype(str)+'\n'+docs.Abstract.astype(str)).tolist()
    from huggingface_hub import HfApi
    model_revision=HfApi().model_info(model_name).sha
    model=SentenceTransformer(model_name,revision=model_revision);max_tokens=model.max_seq_length
    encoded=model.encode(text,batch_size=64,show_progress_bar=True,normalize_embeddings=True,convert_to_numpy=True)
    np.save(out/'sentence_embeddings.npy',encoded)
    sample=np.random.default_rng(20261002).choice(len(encoded),min(1000,len(encoded)),replace=False)
    token_lengths=model.tokenizer(text,add_special_tokens=True,truncation=False,return_length=True)['length']
    results=[];labels={}
    for seed in [0,1,2]:
        lab=KMeans(n_clusters=7,n_init=20,random_state=seed).fit_predict(encoded);labels[seed]=lab
        results.append({'seed':seed,'silhouette_cosine_fixed_sample':float(silhouette_score(encoded[sample],lab[sample],metric='cosine')),'ARI_vs_majority_including_ties':float(adjusted_rand_score(dominant[valid],lab[valid])),'NMI_vs_majority_including_ties':float(normalized_mutual_info_score(dominant[valid],lab[valid])),'ARI_vs_majority_unique_winner_only':float(adjusted_rand_score(dominant[valid&~ties],lab[valid&~ties])),'NMI_vs_majority_unique_winner_only':float(normalized_mutual_info_score(dominant[valid&~ties],lab[valid&~ties]))})
        pd.DataFrame({'family_id':cohort.family_id,'row_index':cohort.row_index,'embedding_cluster':lab}).to_csv(out/f'clusters_seed_{seed}.csv',index=False)
    pd.DataFrame(results).to_csv(out/'baseline_metrics.csv',index=False)
    metadata=pd.read_csv(root/'00_input_snapshot'/'agenticai.csv',keep_default_na=False)
    assert len(metadata)==len(raw)
    assert metadata['Lens ID'].astype(str).tolist()==raw.patent_id.astype(str).tolist()
    # Families may carry multiple class codes and multiple keyword-community fractions.
    # Each code receives that family's fractional profile; totals across codes may
    # exceed the number of families. Subclass-level codes avoid arbitrary top-code selection.
    for field,prefix in [('IPCR Classifications','ipc'),('CPC Classifications','cpc')]:
        table={};families_with_codes=0
        for i,row in cohort.reset_index(drop=True).iterrows():
            codes=set(re.findall(r'\b[A-HY]\d{2}[A-Z]\b',metadata.iloc[int(row.row_index)][field].upper()))
            families_with_codes+=bool(codes)
            for code in codes:table[code]=table.get(code,np.zeros(len(columns)))+fraction[i]
        t=pd.DataFrame.from_dict(table,orient='index',columns=[f'community_{x}' for x in columns]);t.index.name='subclass';t.sort_index().to_csv(out/f'{prefix}_fractional_contingency.csv')
        (out/f'{prefix}_coverage.json').write_text(json.dumps({'families_with_codes':families_with_codes,'families_with_keyword_profile':int(valid.sum()),'rule':'One contribution per distinct subclass per family, split over keyword communities; multi-code families contribute to multiple subclass rows.'},indent=2))
    metadata_out={'model':model_name,'model_revision':model_revision,'max_sequence_tokens':max_tokens,'documents_over_token_limit':sum(x>max_tokens for x in token_lengths),'family_count':len(cohort),'profiles_with_no_retained_keywords':int((~valid).sum()),'majority_ties':int((valid&ties).sum()),'manifest_sha256':digest(manifest_path),'raw_snapshot_sha256':digest(raw_path),'packages':{p:importlib.metadata.version(p) for p in ['sentence-transformers','transformers','torch','numpy','pandas','scikit-learn']},'interpretation':'Agreement between representations is descriptive; neither partition is ground truth. Ties use the smallest community ID for the first comparator and are excluded for the second. Fractional profiles remain the main lexical representation.'}
    (out/'run_metadata.json').write_text(json.dumps(metadata_out,indent=2));print(json.dumps(results,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--model',default='sentence-transformers/all-MiniLM-L6-v2');x=p.parse_args();run(x.root,x.model)
