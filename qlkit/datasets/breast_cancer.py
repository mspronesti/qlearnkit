import sklearn.datasets as skdatasets
from .dataset_helper import features_labels_from_data
from typing import Optional, Union


def load_breast_cancer(train_size: Optional[Union[float, int]] = None,
                       test_size: Optional[Union[float, int]] = None,
                       n_features: Optional[int] = None,
                       *,
                       use_pca: Optional[bool] = False,
                       return_bunch: Optional[bool] = False):
    """
    This script loads breast cancer dataset from sklearn and splits it according to
    the required train size, test size and number of features

    Args:
        test_size :
            float or int, default=None
            If float, should be between 0.0 and 1.0 and represent the proportion
            of the dataset to include in the test split. If int, represents the
            absolute number of test samples. If None, the value is set to the
            complement of the train size. If ``train_size`` is also None, it will
            be set to 0.25.

        train_size : float or int, default=None
            If float, should be between 0.0 and 1.0 and represent the
            proportion of the dataset to include in the train split. If
            int, represents the absolute number of train samples. If None,
            the value is automatically set to the complement of the test size.

        n_features:
            number of desired features

        use_pca:
            whether to use PCA for dimensionality reduction or not
            default False

        return_bunch:
            whether to return a :class:`~sklearn.utils.Bunch`
                    (similar to a dictionary) or not

        Returns:
            Breast Cancer dataset as available in sklearn
    """
    # X: data
    # y: labels
    X, y = skdatasets.load_breast_cancer(return_X_y=True)

    return features_labels_from_data(
        X, y, train_size, test_size, n_features,
        use_pca=use_pca,
        return_bunch=return_bunch
    )
