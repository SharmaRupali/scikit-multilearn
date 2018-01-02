from sklearn.datasets import make_multilabel_classification
from skmultilearn.visualize import VisualizeClusterPie
import unittest
import scipy.sparse as sp
import types



class VisualizeClusterPieTests(unittest.TestCase):

    def test_check_params(self):

        x, y = make_multilabel_classification(sparse=True, n_classes=7, return_indicator='sparse',
                                              allow_unlabeled=False)

        assert sp.issparse(y)

        VisualizeClusterPie.visualize_input_cluster(x, y, labels=None, n_clusters=3, size_proportion=50, colors="hsv")

        parameters = {
            'labels': None,
            'n_clusters': 3,
            'size_proportion': 50,
            'colors': "hsv",
        }

        for p in list(parameters.keys()):
            self.assertTrue(p, )


if __name__ == '__main__':
    unittest.main()

