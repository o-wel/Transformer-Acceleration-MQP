import unittest
from task import score,canonical,encode
class Correctness(unittest.TestCase):
    mr='name[The Eagle], food[Italian], priceRange[cheap], familyFriendly[no]'
    def test_paraphrases(self):
        for t in ['The Eagle serves cheap Italian food and is not family friendly.', 'For inexpensive Italian food, try The Eagle. It is adults only.']:
            self.assertEqual(score(self.mr,t)['semantic_exact'],1)
    def test_contradiction(self):
        self.assertGreater(score(self.mr,'The Eagle offers expensive Italian food and is family friendly.')['slot_error_rate'],0)
    def test_omission(self):
        self.assertGreater(score(self.mr,'The Eagle serves Italian food.')['missing'],0)
    def test_addition(self):
        self.assertGreater(score(self.mr,'The Eagle is by the river and serves cheap Italian food. Adults only.')['added'],0)
    def test_canonical_order(self):
        self.assertEqual(canonical('food[Italian], name[The Eagle]'),canonical('name[The Eagle], food[Italian]'))
    def test_no_reference_leak(self):
        from task import prompt
        self.assertNotIn('secret answer',prompt(dict(mr=self.mr,reference='secret answer')))
if __name__=='__main__':unittest.main()
