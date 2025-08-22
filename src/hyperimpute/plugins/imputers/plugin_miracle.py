# stdlib
from typing import Any, List, Optional

# third party
from miracle import MIRACLE
import numpy as np
import pandas as pd

# hyperimpute absolute
import hyperimpute.plugins.core.params as params
import hyperimpute.plugins.imputers.base as base
from hyperimpute.plugins.imputers.plugin_mean import MeanPlugin
from hyperimpute.plugins.imputers.plugin_median import MedianPlugin


class MiraclePlugin(base.ImputerPlugin):
    """MIRACLE (Missing data Imputation Refinement And Causal LEarning)
    MIRACLE iteratively refines the imputation of a baseline by simultaneously modeling the missingness generating mechanism and encouraging imputation to be consistent with the causal structure of the data.


    Example:
        >>> import numpy as np
        >>> from hyperimpute.plugins.imputers import Imputers
        >>> plugin = Imputers().get("miracle")
        >>> plugin.fit_transform([[1, 1, 1, 1], [np.nan, np.nan, np.nan, np.nan], [1, 2, 2, 1], [2, 2, 2, 2]])

    Reference: "MIRACLE: Causally-Aware Imputation via Learning Missing Data Mechanisms", Trent Kyono, Yao Zhang, Alexis Bellot, Mihaela van der Schaar
    """

    def __init__(
        self,
        lr: float = 0.001,
        batch_size: int = 1024,
        num_outputs: int = 1,
        n_hidden: int = 32,
        reg_lambda: float = 1,
        reg_beta: float = 1,
        DAG_only: bool = False,
        reg_m: float = 1.0,
        window: int = 10,
        max_steps: int = 400,
        seed_imputation: str = "mean",
        random_state: int = 0,
    ) -> None:
        super().__init__(random_state=random_state)

        if seed_imputation not in [
            "mean",
            "median",
        ]:
            raise RuntimeError(
                f"invalid seed imputation for MIRACLE: {seed_imputation}"
            )

        self.lr = lr
        self.batch_size = batch_size
        self.num_outputs = num_outputs
        self.n_hidden = n_hidden
        self.reg_lambda = reg_lambda
        self.reg_beta = reg_beta
        self.DAG_only = DAG_only
        self.reg_m = reg_m
        self.window = window
        self.max_steps = max_steps
        self.seed_imputation = seed_imputation
        self.random_state = random_state

        self._model: Optional[MIRACLE] = None
        self._seed_imputer: Optional[base.ImputerPlugin] = None

    @staticmethod
    def name() -> str:
        return "miracle"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [
            params.Integer("window", 2, 12),
            params.Integer("n_hidden", 32, 100),
            params.Integer("batch_size", 32, 100),
            params.Integer("max_steps", 100, 500),
            params.Categorical("lr", [0.001, 0.0001, 0.00001]),
            params.Categorical("reg_lambda", [1, 0.1, 10]),
            params.Categorical("reg_beta", [1, 3]),
            params.Categorical(
                "seed_imputation",
                ["mean", "median"],
            ),
        ]

    def _get_seed_imputer(self, method: str) -> base.ImputerPlugin:
        if method == "median":
            return MedianPlugin()
        else:
            return MeanPlugin()

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "MiraclePlugin":
        missing_idxs = np.where(np.any(np.isnan(X.values), axis=0))[0]

        self._model = MIRACLE(
            num_inputs=X.shape[1],
            lr=self.lr,
            batch_size=self.batch_size,
            num_outputs=self.num_outputs,
            n_hidden=self.n_hidden,
            reg_lambda=self.reg_lambda,
            reg_beta=self.reg_beta,
            DAG_only=self.DAG_only,
            reg_m=self.reg_m,
            window=self.window,
            max_steps=self.max_steps,
            missing_list=missing_idxs,
            random_seed=self.random_state,
        )

        self._seed_imputer = self._get_seed_imputer(self.seed_imputation)
        X_seed = self._seed_imputer.fit_transform(X)

        # Train model; ignore returned imputed data
        self._model.fit(X.values, X_seed=X_seed.values)
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._model is None or self._seed_imputer is None:
            raise RuntimeError("Model not fitted")

        df_mask = pd.DataFrame(X)
        df_mask = df_mask.where(df_mask.isnull(), 0)
        df_mask = df_mask.mask(df_mask.isnull(), 1)
        indicators = df_mask[df_mask.columns[df_mask.isin([1]).any()]].values
        indicators = 1 - indicators

        X_seed = self._seed_imputer.transform(X)
        X_seed_c = np.concatenate([X_seed.values, indicators], axis=1)

        transformed = self._model._transform(X_seed_c)
        imputed = transformed[:, : X.shape[1]]
        return pd.DataFrame(imputed, columns=X.columns)

    def save(self) -> bytes:
        return b""

    @classmethod
    def load(cls, buff: bytes) -> "MiraclePlugin":
        return MiraclePlugin()


plugin = MiraclePlugin
