from sklearn.datasets import make_multilabel_classification
import matplotlib.pyplot as plt
import scipy.sparse as sp
import unittest
from skmultilearn.visualize import VisualizeNetwork

class VisualizeNetworkTests(unittest.TestCase):

    def test_works_on_appropriate_params(self):

        n_classes = 7
        x, y = make_multilabel_classification(sparse=True, n_classes=n_classes, return_indicator='sparse',
                                              allow_unlabeled=False)

        assert sp.issparse(y)

        VisualizeNetwork.visualize_input_label_network(x, y, n_classes, "copper", plt.cm.BrBG)