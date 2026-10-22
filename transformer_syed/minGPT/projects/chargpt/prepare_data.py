"""Prepare literature or everyday-dialogue corpora; leaves the user's input.txt untouched."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from urllib.request import urlopen

SOURCES = {
    'shakespeare': 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',
    'alice': 'https://www.gutenberg.org/ebooks/11.txt.utf-8',
}


TASKMASTER_ROOT = 'https://raw.githubusercontent.com/google-research-datasets/Taskmaster/master/TM-1-2019/'
EVERYDAY_SOURCES = {
    'taskmaster': TASKMASTER_ROOT + 'woz-dialogs.json',
    'sherlock': 'https://www.gutenberg.org/ebooks/1661.txt.utf-8',
    'pride_and_prejudice': 'https://www.gutenberg.org/ebooks/1342.txt.utf-8',
    'alice': SOURCES['alice'],
}


def dialogue_text(raw):
    """Keep human utterances and roles; discard IDs and annotation metadata."""
    conversations = []
    for conversation in json.loads(raw):
        # Drop incomplete conversations rather than invent speaker assignments.
        if any(not u['speaker'] for u in conversation['utterances']):
            continue
        turns = []
        for utterance in conversation['utterances']:
            role = utterance['speaker']
            if role not in ('USER', 'ASSISTANT'):
                raise ValueError(f'Unexpected speaker: {role}')
            text = ' '.join(utterance['text'].split())
            if text:
                turns.append(f"{role.title()}: {text}")
        if turns:
            conversations.append('\n'.join(turns))
    if not conversations:
        raise ValueError('No conversations found')
    return '\n\n'.join(conversations) + '\n'


def prose_sample(text, limit=200_000):
    """Limit each book's influence, ending at a paragraph boundary."""
    if len(text) <= limit:
        return text
    end = text.rfind('\n\n', 0, limit)
    return text[:end if end > 0 else limit].rstrip() + '\n'


def clean_text(text):
    text = text.replace('\r\n', '\n').replace('\r', '\n').lstrip('\ufeff')
    if '*** START OF' in text:
        match = re.search(r'\*\*\* START OF[^\n]*\n(.*?)\*\*\* END OF', text, re.DOTALL)
        if match is None:
            raise ValueError('Could not locate Gutenberg text boundaries')
        text = match.group(1)
    return text.strip() + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preset', choices=('everyday', 'literature'), default='everyday')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = Path('data') / args.preset
    sources = EVERYDAY_SOURCES if args.preset == 'everyday' else SOURCES
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    texts = []
    for name, url in sources.items():
        raw_path = args.output_dir / (name + '.raw.txt')
        if not raw_path.exists():
            with urlopen(url, timeout=60) as response:
                raw = response.read()
            # Decode before caching so a decoding error cannot poison the cache.
            raw.decode('utf-8')
            raw_path.write_bytes(raw)
        raw = raw_path.read_bytes()
        decoded = raw.decode('utf-8')
        text = dialogue_text(decoded) if name == 'taskmaster' else clean_text(decoded)
        if args.preset == 'everyday' and name != 'taskmaster':
            text = prose_sample(text)
        (args.output_dir / (name + '.txt')).write_text(text, encoding='utf-8')
        texts.append(text)
        manifest[name] = dict(url=url, characters=len(text),
                              raw_sha256=hashlib.sha256(raw).hexdigest())
    if args.preset == 'everyday':
        with urlopen(TASKMASTER_ROOT + 'README.md', timeout=60) as response:
            (args.output_dir / 'TASKMASTER_ATTRIBUTION.md').write_bytes(response.read())
    combined = '\n\n'.join(texts)
    (args.output_dir / 'input.txt').write_text(combined, encoding='utf-8')
    manifest['combined'] = dict(characters=len(combined), vocabulary=sorted(set(combined)),
                                sha256=hashlib.sha256(combined.encode('utf-8')).hexdigest())
    (args.output_dir / 'sources.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f"Prepared {len(combined):,} characters, {len(set(combined))} unique characters in {args.output_dir}")


if __name__ == '__main__':
    main()
