"""Existing grouped SMS split, original 96-token message cap and label-logit grading."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.sms_spam.experiment import metrics


def prompt(row, tokenizer):
    return (tokenizer.encode('Classify this SMS as ham or spam.\nSMS: ')
            + tokenizer.encode(row['text'])[:96] + tokenizer.encode('\nLabel:'))


def encode(rows, tokenizer, length, torch):
    result = []
    labels = [tokenizer.encode(' '+label) for label in ['ham','spam']]
    assert all(len(ids)==1 for ids in labels)
    for row in rows:
        ids = prompt(row, tokenizer)
        assert len(ids)+1 <= length
        label = labels[row['label']][0]
        x = torch.full((length,), tokenizer.eos_token_id, dtype=torch.long)
        y = torch.full((length,), -1, dtype=torch.long)
        x[:len(ids)+1] = torch.tensor(ids+[label])
        y[len(ids)-1] = label
        result.append((x,y))
    return result


def evaluate(model, rows, tokenizer, torch):
    gold, pred, outputs = [], [], []
    labels = [tokenizer.encode(' '+label)[0] for label in ['ham','spam']]
    total = 0.
    model.eval()
    with torch.inference_mode():
        for start in range(0,len(rows),8):
            chunk=rows[start:start+8]
            ids=[prompt(r,tokenizer) for r in chunk]
            x=torch.full((len(ids),128),tokenizer.eos_token_id,dtype=torch.long,device='cuda')
            for i,tokens in enumerate(ids):x[i,:len(tokens)]=torch.tensor(tokens,device='cuda')
            logits=model(x)[0][torch.arange(len(ids),device='cuda'),torch.tensor([len(t)-1 for t in ids],device='cuda')].float()
            target=torch.tensor([labels[r['label']] for r in chunk],device='cuda')
            total+=torch.nn.functional.cross_entropy(logits,target,reduction='sum').item()
            scores=logits[:,labels]
            predictions=scores.argmax(-1).tolist()
            probabilities=scores.softmax(-1)[:,1].tolist()
            for row,p,q in zip(chunk,predictions,probabilities):
                gold.append(row['label']);pred.append(p)
                outputs.append(dict(id=row['id'],gold=row['label'],pred=p,spam_probability=q))
    return dict(**metrics(gold,pred),label_nll=total/len(rows)),outputs
