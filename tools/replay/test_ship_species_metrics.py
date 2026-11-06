"""Classification accuracy must not hide missed enemies or unmatched predictions."""
from collections import Counter
import unittest

from verify_ship_species import species_metrics


class SpeciesMetrics(unittest.TestCase):
    def test_selective_accuracy_does_not_hide_end_to_end_misses(self):
        result = species_metrics(
            Counter({('Cruiser', 'Cruiser'): 3, ('Cruiser', 'unknown'): 2,
                     ('Destroyer', 'Cruiser'): 1}),
            Counter(matched=6, unmatched=4, unmatched_classified=2),
            Counter(unknown=2, Cruiser=2), 10, 8, ('Cruiser', 'Destroyer'))
        self.assertEqual(.75, result['accuracy_when_classified'])
        self.assertEqual(4/6, result['coverage'])
        self.assertEqual(3/8, result['correct_species_recall'])
        self.assertEqual(.6, result['detection_precision'])
        self.assertEqual(.75, result['detection_recall'])
        self.assertEqual({'unknown': 2, 'Cruiser': 2}, result['unmatched_species'])

    def test_unresolved_reference_is_not_a_classification_failure(self):
        result = species_metrics(Counter({('Cruiser', 'Cruiser'): 1}),
                                 Counter(matched=2, unresolved_species=1),
                                 Counter(), 2, 3, ('Cruiser', 'Submarine'))
        self.assertEqual(1, result['species_reference'])
        self.assertEqual(1, result['accuracy_when_classified'])
        self.assertEqual(1/3, result['correct_species_recall'])
        self.assertIsNone(result['per_species']['Submarine']['accuracy_when_classified'])

    def test_no_confirmed_classification_is_not_perfect_accuracy(self):
        result = species_metrics(Counter({('Battleship', 'unknown'): 2}),
                                 Counter(matched=2), Counter(), 2, 2, ('Battleship',))
        self.assertIsNone(result['accuracy_when_classified'])
        self.assertEqual(0, result['coverage'])
        self.assertEqual(0, result['correct_species_recall'])
        empty = species_metrics(Counter(), Counter(), Counter(), 0, 0, ('Battleship',))
        self.assertIsNone(empty['detection_precision'])
        self.assertIsNone(empty['detection_recall'])
        self.assertIsNone(empty['correct_species_recall'])
