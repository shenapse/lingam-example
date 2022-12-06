from typing import Final

import numpy as np
import pandas as pd
from lingam.utils import simulate_linear_sem
from nptyping import NDArray


def generate_unif(
    mat: NDArray, seed: int = 123, n_sample: int = 1000, columns: list[str] | None = None
) -> pd.DataFrame:
    """generate samples data from given adjacency matrix"""
    n_variable_full: Final[int] = mat.shape[0]
    np.random.seed(seed)
    x = np.zeros((n_variable_full, n_sample))
    # x[6] = np.random.normal(size=n_sample)
    x[6] = np.random.uniform(size=n_sample)
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


def generate_normal_unif(
    mat: NDArray, seed: int = 123, n_sample: int = 1000, columns: list[str] | None = None
) -> pd.DataFrame:
    """generate samples data from given adjacency matrix"""
    n_variable_full: Final[int] = mat.shape[0]
    np.random.seed(seed)
    x = np.zeros((n_variable_full, n_sample))
    x[6] = np.random.normal(scale=0.25, size=n_sample)
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


def generate_gamma_unif(
    mat: NDArray, seed: int = 123, n_sample: int = 1000, columns: list[str] | None = None
) -> pd.DataFrame:
    """generate samples data from given adjacency matrix"""
    n_variable_full: Final[int] = mat.shape[0]
    np.random.seed(seed)
    x = np.zeros((n_variable_full, n_sample))
    x[6] = np.random.gamma(shape=2.0, scale=0.5, size=n_sample)
    x[3] = np.dot(mat[3, :], x) + np.random.uniform(low=-1.0, high=1.0, size=n_sample)
    x[0] = np.dot(mat[0, :], x) + np.random.uniform(low=-1.0, high=1.0, size=n_sample)
    x[2] = np.dot(mat[2, :], x) + np.random.uniform(low=-1.0, high=1.0, size=n_sample)
    x[1] = np.dot(mat[1, :], x) - np.random.gamma(shape=2.0, scale=1.0, size=n_sample)
    x[5] = np.dot(mat[5, :], x) - np.random.gamma(shape=2.0, scale=1.0, size=n_sample)
    x[4] = np.dot(mat[4, :], x) - np.random.gamma(shape=2.0, scale=1.0, size=n_sample)
    col_names: list[str] = (
        columns
        if columns is not None and len(columns) == n_variable_full
        else [f"x{i}" for i in range(n_variable_full)]
    )
    return pd.DataFrame(np.array(x).T, columns=col_names)


def generate_data_from_sem(
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
