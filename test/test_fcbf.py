
import unittest

import pandas as pd

from fcbf import fcbf


class TestFCBF(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset_lung_cancer = pd.read_csv('data\lung-cancer.data',
                                               header=None, na_values='?')

    def test_fcbf(self):
        data = self.dataset_lung_cancer
        X = data[data.columns[1:]]
        y = data[data.columns[0]].astype(int)
        print(X)
        print(y)
        relevant_features, irrelevant_features, correlations = fcbf(X, y, su_threshold=0.1, base=2)
        print('relevant_features:', relevant_features, '(count:', len(relevant_features), ')')
        print('irrelevant_features:', irrelevant_features, '(count:', len(irrelevant_features), ')')
        print('correlations:', correlations)
        relevant_features_target = [40, 20, 56, 2, 10]
        correlations_target = {
            40: 0.320545012933852, 20: 0.3201758595974314, 56: 0.19562364540298677,
            2: 0.1525108254335835, 10: 0.12478090695260607, 17: 0.1034942352122279,
            51: 0.10759808783471542, 7: 0.10930581775948547, 48: 0.11729774162877939,
            47: 0.11729774162877939, 15: 0.11888084314748912, 49: 0.12746174086460466,
            50: 0.13251653887408732, 39: 0.13808569277723473, 45: 0.1514451640790911,
            13: 0.16692403703085912, 53: 0.1700663644419547, 37: 0.18335869124498036,
            46: 0.18480247861196022, 8: 0.18848330303206917, 33: 0.21263447548330644,
            23: 0.2154721906234732, 43: 0.21712431344241784, 6: 0.24878669151941848,
            14: 0.25114400843106244, 19: 0.30498083504946366}
        self.assertEqual(relevant_features, relevant_features_target)
        self.assertEqual(correlations, correlations_target)
