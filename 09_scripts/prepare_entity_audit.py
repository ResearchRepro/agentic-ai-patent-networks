"""Prospective candidate-pair sample, not a retrospective merge-error estimate."""
from pathlib import Path
from collections import defaultdict
import pandas as pd, re, random, unicodedata, hashlib, json

def normalize(x):return re.sub(r'[^A-Z0-9]+',' ',unicodedata.normalize('NFKD',x).upper()).strip()
def distance(a,b):
    row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        new=[i]
        for j,y in enumerate(b,1):new.append(min(new[-1]+1,row[j]+1,row[j-1]+(x!=y)))
        row=new
    return row[-1]

def run():
    d=pd.read_csv('audit_inputs/agenticai.csv',keep_default_na=False)
    names=sorted({x.strip() for s in d.Applicants for x in s.split(';;') if x.strip()})
    blocks=defaultdict(list)
    for name in names:
        norm=normalize(name)
        if norm:blocks[norm[:3]].append((name,norm))
    pairs=[]
    for block in blocks.values():
        for i,(a,aa) in enumerate(block):
            for b,bb in block[i+1:]:
                if abs(len(aa)-len(bb))>.25*max(len(aa),len(bb)):continue
                sim=1-distance(aa,bb)/max(len(aa),len(bb))
                if sim<.75:continue
                pairs.append({'name_a':a,'name_b':b,'normalized_a':aa,'normalized_b':bb,'normalized_levenshtein_similarity':sim,'candidate_at_0_90':sim>=.9,'sampling_stratum':'at_or_above_0.90' if sim>=.9 else '0.75_to_below_0.90','same_entity_manual':'','rationale':'','reviewer':'','review_date':''})
    rng=random.Random(20261002);selected=[]
    for group in ['at_or_above_0.90','0.75_to_below_0.90']:
        pool=[p for p in pairs if p['sampling_stratum']==group];rng.shuffle(pool);selected+=pool[:50]
    out=Path('new_analysis/entity_audit');out.mkdir(exist_ok=True)
    pd.DataFrame(selected).to_csv(out/'entity_pairs_to_review.csv',index=False)
    (out/'sample_design.json').write_text(json.dumps({'source':'Original applicants in the supplied Lens export','unique_original_names':len(names),'blocking':'First three characters after uppercase Unicode decomposition and replacement of punctuation by spaces','candidate_floor':.75,'candidate_threshold':.90,'similarity':'1 - Levenshtein distance / maximum normalized-name length','seed':20261002,'candidate_pool':len(pairs),'selected_pairs':len(selected),'strata_pool':{s:sum(x['sampling_stratum']==s for x in pairs) for s in ['at_or_above_0.90','0.75_to_below_0.90']},'interpretation':'Prospective pair validation. This is not the original adjudication log, does not estimate errors outside the blocking scheme, and is not a global false-merge or false-split rate.'},indent=2))
    print('Prepared',len(selected),'pairs from',len(pairs),'candidate pairs')
if __name__=='__main__':run()
