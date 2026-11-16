"""Transparent answer scoring; retain raw outputs for human assessment."""
import re
from collections import Counter


def score(row, text):
    sentences = re.split(r'[.!?\n]',text.strip())
    first = sentences[0].lower()
    if row['kind']=='comeback':
        match = re.match(r'\s*(yes|no)\b',text,re.I)
        primary = bool(match and match.group(1).lower()==row['primary'].lower())
    else:
        # A first-sentence mention of the other team makes this ambiguous, not correct.
        mentioned = [team for team in row['teams'] if re.search(r'\b'+re.escape(team)+r'\b',first,re.I)]
        primary = mentioned == [row['primary']]
        negative = r'\b(not|never|unknown|unsure|lost|loses|trailed|trailing|behind)\b'
        if re.search(negative,first):
            primary = False
    # Exact multiset: allows alternate score punctuation/order, rejects extra/missing numbers.
    numbers = [int(n) for n in re.findall(r'\b\d+\b',text)]
    correct_numbers = Counter(numbers)==Counter(row['expected_numbers'])
    # Strict reference match supplements the lenient field score, catches relationships it misses.
    normalize = lambda s: ' '.join(re.findall(r'\w+',s.lower()))
    exact = normalize(text)==normalize(row['answer'])
    return dict(primary_correct=primary, score_numbers_correct=correct_numbers,
                fields_correct=primary and correct_numbers, reference_exact=exact)


def aggregate(rows):
    keys = ['primary_correct','score_numbers_correct','fields_correct','reference_exact']
    return {key:sum(r['scores'][key] for r in rows)/len(rows) for key in keys}
