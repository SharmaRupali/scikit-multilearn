from sklearn.datasets import make_multilabel_classification
import matplotlib.pyplot as plt
import scipy.sparse as sp
import unittest


class VisualizeNetworkTests(unittest.TestCase):

    def test_works_on_appropriate_params(self):

        x, y = make_multilabel_classification(sparse=True, n_classes=7, return_indicator='sparse',
                                              allow_unlabeled=False)

        assert sp.issparse(y)
        self.assertTrue()