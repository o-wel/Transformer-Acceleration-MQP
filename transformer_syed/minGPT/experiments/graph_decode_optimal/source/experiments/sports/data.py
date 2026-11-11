"""Deterministic fictional basketball results; split matches before creating questions."""
import json
import random
from pathlib import Path

ROOT = Path(__file__).parent
TRAIN_TEAMS = ['Falcons', 'Rockets', 'Comets', 'Tigers', 'Lions', 'Hawks', 'Bears', 'Wolves']
TEST_TEAMS = ['Eagles', 'Owls', 'Sharks', 'Bulls']


def example(match, kind, variant=0):
    a, b = match['teams']
    ha, hb = match['half']
    fa, fb = match['final']
    winner, loser = (a, b) if fa > fb else (b, a)
    leader = a if ha > hb else b
    wh, lh = (ha, hb) if fa > fb else (hb, ha)
    high, low = max(fa, fb), min(fa, fb)
    context = f'Basketball match: {a} vs {b}. Halftime score: {a} {ha}, {b} {hb}. Final score: {a} {fa}, {b} {fb}.'
    if kind == 'winner':
        questions = ['Who won the match, and what was the final score?', 'Which team won? Include the final score.']
        answer = f'{winner} won {high}-{low}.'
        primary, numbers = winner, [high, low]
    elif kind == 'halftime':
        questions = ['Who was ahead at halftime, and what was the halftime score?', 'Which team led at the break? Include the halftime score.']
        answer = f'{leader} led {max(ha,hb)}-{min(ha,hb)} at halftime.'
        primary, numbers = leader, [max(ha,hb), min(ha,hb)]
    elif kind == 'comeback':
        questions = [f'Did {winner} come back from a halftime deficit to win? Explain using the scores.', f'Was {winner} behind at halftime before winning? Explain using the scores.']
        primary = 'Yes' if wh < lh else 'No'
        verb = 'trailed' if wh < lh else 'led'
        answer = f'{primary}. {winner} {verb} {wh}-{lh} at halftime and won {high}-{low}.'
        numbers = [wh, lh, high, low]
    else:
        raise ValueError(kind)
    return dict(match_id=match['id'], kind=kind, teams=[a,b], match=match,
                question=context+'\n'+questions[variant], answer=answer,
                primary=primary, expected_numbers=numbers, winner=winner)


def generate():
    rng = random.Random(20260929)
    seen = set()
    outputs = {}
    for split, count, teams in [('train',480,TRAIN_TEAMS),('validation',40,TRAIN_TEAMS),('test',40,TEST_TEAMS)]:
        rows = []
        for i in range(count):
            while True:
                a,b = rng.sample(teams,2)
                half = rng.sample(range(25,56),2)
                final = rng.sample(range(65,106),2)
                signature = tuple(half+final)
                if signature not in seen:
                    seen.add(signature)
                    break
            m = dict(id=f'{split}-{i:03d}',teams=[a,b],half=half,final=final)
            for kind in ['winner','halftime','comeback']:
                rows.append(example(m,kind,variant=i%2 if split=='train' else 0))
        outputs[split] = rows
    counter = []
    for row in outputs['test']:
        m = dict(row['match'])
        field = 'final' if row['kind']=='winner' else 'half'
        m[field] = list(reversed(m[field]))
        m['id'] += '-counter-'+row['kind']
        cf = example(m,row['kind'])
        cf['pair_id'] = row['match_id']+':'+row['kind']
        counter.append(cf)
    outputs['counterfactual'] = counter
    for split, rows in outputs.items():
        (ROOT/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    manifest = dict(source='Programmatically authored fictional basketball scores; no external dataset.',seed=20260929,
                    train_teams=TRAIN_TEAMS,test_teams=TEST_TEAMS,
                    split_rows={k:len(v) for k,v in outputs.items()},
                    split_matches={k:len({r['match_id'] for r in v}) for k,v in outputs.items()},
                    design='No shared score quadruples or match IDs across train/validation/test. Test teams absent from training. Counterfactual rows are paired with test rows and deliberately not independent.')
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return outputs


if __name__ == '__main__':
    generate()
