# stdlib
from unittest.mock import patch

# third party
import numpy as np
import pandas as pd

# hyperimpute absolute
from hyperimpute.plugins import Imputers
from hyperimpute.plugins.prediction.base import PredictionPlugin
from hyperimpute.utils.serialization import load, save


def test_hyperimpute_transform_without_retraining() -> None:
    np.random.seed(0)

    # simple training dataset with missing values
    train = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, np.nan],
            "b": [1.0, 2.0, np.nan, 4.0, 5.0],
        }
    )

    # dataset for inference
    test = pd.DataFrame({"a": [np.nan, 5.0], "b": [5.0, np.nan]})

    plugin = Imputers().get(
        "hyperimpute",
        classifier_seed=["logistic_regression"],
        regression_seed=["linear_regression"],
        optimizer="simple",
        n_inner_iter=1,
        class_threshold=2,
    )

    with patch.object(PredictionPlugin, "fit", wraps=PredictionPlugin.fit) as spy_fit:
        plugin.fit(train)
        fit_calls = spy_fit.call_count

        buff = save(plugin)
        plugin_new = load(buff)
        res = plugin_new.transform(test)

        assert spy_fit.call_count == fit_calls
        assert not np.any(np.isnan(res))
