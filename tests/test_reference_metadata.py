"""Reference identity guards using the official proceedings record."""
import csv
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TITLE = 'Provable Tensor Factorization with Missing Data'
AUTHORS = 'Prateek Jain and Sewoong Oh'
URL = ('https://proceedings.neurips.cc/paper_files/paper/2014/hash/'
       'd04d4a1518efb7b7b80c8352d567ecd5-Abstract.html')


def validate_record(row):
    expected = {'key': 'jain2014', 'authors': AUTHORS, 'title': TITLE, 'year': '2014',
                'canonical_identifier_or_record': URL, 'primary_record_checked': URL}
    if any(row.get(key) != value for key, value in expected.items()):
        raise ValueError('Jain-Oh proceedings identity mismatch')


class ReferenceMetadataTests(unittest.TestCase):
    def setUp(self):
        with (ROOT / 'reference_audit.csv').open(encoding='utf-8', newline='') as f:
            self.record = next(r for r in csv.DictReader(f) if r['key'] == 'jain2014')

    def test_official_record_and_resource_catalog_agree(self):
        validate_record(self.record)
        with (ROOT / 'external_resources.csv').open(encoding='utf-8', newline='') as f:
            rows = [r for r in csv.reader(f) if r and r[0] == TITLE]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][1], URL)

    def test_mutually_consistent_wrong_route_is_rejected(self):
        row = dict(self.record)
        row['canonical_identifier_or_record'] = row['primary_record_checked'] = URL.replace(
            'd04d4a1518efb7b7b80c8352d567ecd5', '81e5f81db77c596492e6f1a5a792ed53')
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            validate_record(row)

    def test_mismatched_author_is_rejected(self):
        row = dict(self.record, authors='Prateek Jain')
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            validate_record(row)


if __name__ == '__main__':
    unittest.main()
