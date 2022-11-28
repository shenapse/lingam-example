# bottom up parce lingam
# with single unobservable common cause
# import time
import warnings
from typing import Final

import numpy as np
import pandas as pd
from fns import get_amat, run_model, save_dot
from lingam.utils import make_dot, make_prior_knowledge, simulate_linear_sem
from nptyping import NDArray

warnings.filterwarnings("ignore")


def gen_data(
    mat: NDArray,
    seed: int = 123,
    n_sample: int = 1000,
    sem_type: str = "gumbel",
    scale: float = 1.0,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """generate samples data from given adjacency matrix"""
    n_variable_full: Final[int] = mat.shape[0]
    np.random.seed(seed)
    col_names: list[str] = (
        columns
        if columns is not None and len(columns) == n_variable_full
        else [f"x{i}" for i in range(n_variable_full)]
    )
    _sem = simulate_linear_sem(mat, n_samples=n_sample, sem_type=sem_type, noise_scale=scale)
    return pd.DataFrame(_sem, columns=col_names)


def main():
    # define true effect to estimate
    effect_true: NDArray = get_amat()
    dot_true = make_dot(effect_true)
    save_dot(dot_true, name="bup_true")

    # generate sample data from the adjacency matrix given above
    sem_types: Final[list[str]] = ["gauss", "exp", "gumbel", "logistic"]
    for sem_type in sem_types:
        X_full: pd.DataFrame = gen_data(effect_true, sem_type=sem_type)
        # let x6 unobservable
        X = X_full.drop("x6", axis="columns")
        n_variable: Final[int] = X.shape[1]
        # define lingam model with prior
        prior = make_prior_knowledge(
            n_variables=n_variable,
            sink_variables=[1, 4, 5],
        )

        run_model(prior_mat=prior, data=X, n_boot=100, name_graph=f"bup_est_{sem_type}")


if __name__ == "__main__":
    main()
