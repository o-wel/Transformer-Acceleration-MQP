"""Freeze MR-disjoint subsets from pinned cleaned official splits; no truncation."""
import csv,hashlib,json,random
from collections import defaultdict
from pathlib import Path
from task import canonical,prompt,score
HERE=Path(__file__).resolve().parent

def main():
    from transformers import GPT2Tokenizer
    tokenizer=GPT2Tokenizer.from_pretrained('gpt2',local_files_only=True)
    dest=HERE/'data';dest.mkdir(exist_ok=False)
    manifest=dict(seed=20261116,source='tuetschek/e2e-cleaning cleaned no-overlap official splits',
        max_encoded_length=256,max_generated_tokens=96,selection='MR-uniform subset; seeded single reference for training/NLL; all references retained',splits={})
    keys={}
    for split,file,count in [('train','train-fixed.no-ol.csv',1200),('validation','devel-fixed.no-ol.csv',200),('test','test-fixed.csv',200)]:
        groups=defaultdict(list)
        with (HERE/'vendor'/('cleaned-data__'+file)).open() as f:
            for row in csv.DictReader(f):groups[json.dumps(canonical(row['mr']))].append(row)
        keys[split]=set(groups)
        eligible=[];excluded=0
        rng=random.Random(20261116)
        for key,refs in sorted(groups.items()):
            refs=sorted({(r['mr'],r['ref']) for r in refs})
            # Require every retained reference and the generation allowance to fit.
            valid=[(mr,ref) for mr,ref in refs if len(tokenizer.encode(prompt({'mr':mr})))+len(tokenizer.encode(' '+ref))+1<=256 and len(tokenizer.encode(prompt({'mr':mr})))+96<=1024]
            if not valid:excluded+=1;continue
            mr,ref=rng.choice(valid)
            eligible.append(dict(id=hashlib.sha256(key.encode()).hexdigest(),mr=mr,reference=ref,references=sorted({r[1] for r in valid}),slots=len(canonical(mr))))
        rng.shuffle(eligible)
        assert len(eligible)>=count,(split,len(eligible),count)
        chosen=eligible[:count]
        path=dest/f'{split}.jsonl';path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in chosen))
        manifest['splits'][split]=dict(groups=len(groups),eligible=len(eligible),excluded_overlength=excluded,selected=len(chosen),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            reference_semantic_exact=sum(score(r['mr'],r['reference'])['semantic_exact'] for r in chosen)/len(chosen))
    assert all(not keys[a]&keys[b] for a,b in [('train','validation'),('train','test'),('validation','test')])
    manifest['official_mr_disjoint']=True
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
