# bottom up parce lingam
# with single unobservable common cause
# that is subject to normal distribution
import warnings
from typing import Final

import numpy as np
import pandas as pd
from fns import get_amat, run_model, save_dot
from lingam.utils import make_dot, make_prior_knowledge
from nptyping import NDArray

warnings.filterwarnings("ignore")


def gen_data(mat: NDArray, seed: int = 123, n_sample: int = 1000, columns: list[str] | None = None) -> pd.DataFrame:
    """generate samples data from given adjacency matrix"""
    n_variable_full: Final[int] = mat.shape[0]
    np.random.seed(seed)
    x = np.zeros((n_variable_full, n_sample))
    x[6] = np.random.normal(size=n_sample)
    x[3] = np.dot(mat[3, :], x) + np.random.uniform(size=n_sample)
    x[0] = np.dot(mat[0, :], x) + np.random.uniform(size=n_sample)
    x[2] = np.dot(mat[2, :], x) + np.random.uniform(size=n_sample)
    x[1] = np.dot(mat[1, :], x) + np.random.uniform(size=n_sample)
    x[5] = np.dot(mat[5, :], x) + np.random.uniform(size=n_sample)
    x[4] = np.dot(mat[4, :], x) + np.random.uniform(size=n_sample)
    col_names: list[str] = (
        columns
        if columns is not None and len(columns) == n_variable_full
        else [f"x{i}" for i in range(n_variable_full)]
    )
    return pd.DataFrame(np.array(x).T, columns=col_names)


def main():
    # true effect to estimate
    effect_true: NDArray = get_amat()
    dot_true = make_dot(effect_true)
    save_dot(dot_true, name="bup_true")

    # generate sample data from the adjacency matrix given above
    X_full: pd.DataFrame = gen_data(effect_true)
    # let x6 unobservable
    X = X_full.drop("x6", axis="columns")
    n_variable: Final[int] = X.shape[1]

    # define lingam model with prior
    prior = make_prior_knowledge(
        n_variables=n_variable,
        sink_variables=[1, 4, 5],
    )
    run_model(prior_mat=prior, data=X, n_boot=100, name_graph="bup_est_normal_unif")


if __name__ == "__main__":
    main()
