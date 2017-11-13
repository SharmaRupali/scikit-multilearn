from sklearn.datasets import make_multilabel_classification
import matplotlib.pyplot as plt
import scipy.sparse as sp
import unittest
import types


class VisualizeNetworkTests(unittest.TestCase):

    def test_works_on_appropriate_params(self):

        x, y = make_multilabel_classification(sparse=True, n_classes=7, return_indicator='sparse',
                                              allow_unlabeled=False)

        assert sp.issparse(y)

        VisualizeNetwork.visualize_input_network(x, y, labels=None, k=2, iterations=50, n_size=25,
                                                 n_e_ratiotype="linear", linear_ratio_val=0.5, log_ratio_base_val=1.2,
                                                 cmap_node="copper", cmap_edge=plt.cm.BrBG)

        labels = self.labels


        def test_works(self):
            print('labels ='), self.labels
            isinstance(labels, types.StringTypes)

        parameters = {
            'labels': None,
            'k': 2,
            'iterations': 50,
            'n_size': 25,
            'n_e_ratiotype': "linear",
            'linear_ratio_val': 0.5,
            'log_ratio_base_val': 1.2,
            'cmap_node': "copper",
            'cmap_edge': plt.cm.BrBG,
        }

        for p in list(parameters.keys()):
            self.assertTrue(p, )


if __name__ == '__main__':
    unittest.main()