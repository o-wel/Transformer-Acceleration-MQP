"""Author a fictional sports-club FAQ. No claims about real clubs or rules."""
import json
from pathlib import Path

facts = [
    ('Where do the Harbor Comets practice?', 'The Harbor Comets practice at Cedar Court.', 'Where is practice for the Harbor Comets?', 'Cedar Court'),
    ('When do the Harbor Comets practice?', 'The Harbor Comets practice on Tuesdays at 6 pm.', 'What is the practice schedule for the Harbor Comets?', 'Tuesdays at 6 pm'),
    ('Who coaches the Harbor Comets?', 'The Harbor Comets coach is Mira Vale.', 'Who is the coach of the Harbor Comets?', 'Mira Vale'),
    ('What color do the Harbor Comets wear?', 'The Harbor Comets wear teal jerseys.', 'What is the jersey color of the Harbor Comets?', 'teal'),
    ('Where do the Pine Falcons practice?', 'The Pine Falcons practice at Birch Field.', 'Where is practice for the Pine Falcons?', 'Birch Field'),
    ('When do the Pine Falcons practice?', 'The Pine Falcons practice on Thursdays at 5 pm.', 'What is the practice schedule for the Pine Falcons?', 'Thursdays at 5 pm'),
    ('Who coaches the Pine Falcons?', 'The Pine Falcons coach is Omar Reed.', 'Who is the coach of the Pine Falcons?', 'Omar Reed'),
    ('What color do the Pine Falcons wear?', 'The Pine Falcons wear orange jerseys.', 'What is the jersey color of the Pine Falcons?', 'orange'),
    ('Where do the Summit Owls practice?', 'The Summit Owls practice at Maple Arena.', 'Where is practice for the Summit Owls?', 'Maple Arena'),
    ('When do the Summit Owls practice?', 'The Summit Owls practice on Saturdays at 10 am.', 'What is the practice schedule for the Summit Owls?', 'Saturdays at 10 am'),
    ('Who coaches the Summit Owls?', 'The Summit Owls coach is Lena Park.', 'Who is the coach of the Summit Owls?', 'Lena Park'),
    ('What color do the Summit Owls wear?', 'The Summit Owls wear purple jerseys.', 'What is the jersey color of the Summit Owls?', 'purple'),
]
root = Path(__file__).parent
train, test = [], []
for question, answer, heldout, expected in facts:
    for prefix in ['', 'Please tell me: ', 'I am new to the club. ', 'Can you help? ']:
        train.append(dict(question=prefix + question, answer=answer, expected=expected))
    test.append(dict(question=heldout, answer=answer, expected=expected))
for split, rows in [('train', train), ('test', test)]:
    (root / f'{split}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
