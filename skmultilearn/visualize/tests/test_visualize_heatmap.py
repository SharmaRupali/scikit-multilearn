from sklearn.datasets import make_multilabel_classification
from skmultilearn.visualize import VisualizeHeatmap
import scipy.sparse as sp
import unittest
import types


class VisualizeHeatmapTests(unittest.TestCase):

    def test_validate_params(self):

        x, y = make_multilabel_classification(sparse=True, n_classes=7, return_indicator='sparse',
                                              allow_unlabeled=False)

        assert sp.issparse(y)

        VisualizeHeatmap.visualize_input_heatmap(x, y, cmap="reds")

        parameters = {
            'cmap': "Reds",
        }

        for p in list(parameters.keys()):
            self.assertTrue(p, )


if __name__ == '__main__':
    unittest.main()