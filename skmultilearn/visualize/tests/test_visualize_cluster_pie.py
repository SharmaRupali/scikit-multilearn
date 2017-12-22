from sklearn.datasets import make_multilabel_classification
from skmultilearn.visualize import VisualizeClusterPie
import unittest
import scipy.sparse as sp
import types



class VisualizeClusterPieTests(unittest.TestCase):

    def test_works_on_appropriate_params(self):

        x, y = make_multilabel_classification(sparse=True, n_classes=7, return_indicator='sparse',
                                              allow_unlabeled=False)

        assert sp.issparse(y)

        VisualizeClusterPie.visualize_input_cluster(x, y, labels=None, n_clusters=3, size_proportion=50)

        labels = self.labels

        def test_works(self):
            print('labels ='), self.labels
            isinstance(labels, types.StringTypes)

        parameters = {
            'labels': None,
            'n_clusters': 3,
            'size_proportion': 50,
        }

        for p in list(parameters.keys()):
            self.assertTrue(p, )


if __name__ == '__main__':
    unittest.main()