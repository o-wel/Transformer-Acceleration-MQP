import json
import unittest
from projects.chargpt.prepare_data import clean_text, dialogue_text, prose_sample


class TestCorpusPreparation(unittest.TestCase):
    def test_dialogue_preserves_roles_and_boundaries(self):
        raw = json.dumps([
            {'conversationId': 'not training text', 'utterances': [
                {'speaker': 'USER', 'text': 'Hello\n there', 'segments': []},
                {'speaker': 'ASSISTANT', 'text': 'Hi!'}]},
            {'utterances': [{'speaker': 'USER', 'text': 'Coffee, please.'}]}])
        self.assertEqual(dialogue_text(raw),
                         'User: Hello there\nAssistant: Hi!\n\nUser: Coffee, please.\n')
        invalid = json.dumps([{'utterances': [{'speaker': '', 'text': 'Unknown'}]}])
        with self.assertRaises(ValueError):
            dialogue_text(invalid)
        with self.assertRaises(ValueError):
            dialogue_text('[]')

    def test_book_cleanup_and_cap(self):
        self.assertEqual(clean_text('notice\r\n*** START OF BOOK ***\r\nStory.\r\n*** END OF BOOK ***\r\nlicense'), 'Story.\n')
        self.assertEqual(prose_sample('First.\n\nSecond paragraph.\n', limit=12), 'First.\n')
        self.assertEqual(prose_sample('Short.\n'), 'Short.\n')


if __name__ == '__main__':
    unittest.main()
