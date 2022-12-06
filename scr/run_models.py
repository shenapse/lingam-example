import warnings
from typing import Final

import pandas as pd
from fns import get_amat, make_dot, run_model, save_dot
from generate_data import generate_data_from_sem, generate_gamma_unif, generate_normal_unif, generate_unif
from lingam import RCD, BottomUpParceLiNGAM
from lingam.utils import make_prior_knowledge
from nptyping import NDArray

warnings.filterwarnings("ignore")


def bup_with_sem_data(n_boot: int):
    # define true effect to estimate
    effect_true: NDArray = get_amat()
    dot_true = make_dot(effect_true)
    save_dot(dot_true, name="bup_true")

    # generate sample data from the adjacency matrix given above
    sem_types: Final[list[str]] = ["gauss", "exp", "gumbel"]
    for sem_type in sem_types:
        print(f"{sem_type} model")
        X_full: pd.DataFrame = generate_data_from_sem(effect_true, sem_type=sem_type)
        # let x6 unobservable
        X = X_full.drop("x6", axis="columns")
        n_variable: Final[int] = X.shape[1]
        # define lingam model with prior
        prior = make_prior_knowledge(
            n_variables=n_variable,
            sink_variables=[1, 4, 5],
        )
        model = BottomUpParceLiNGAM(prior_knowledge=prior)
        run_model(model=model, data=X, n_boot=n_boot, name_graph=f"bup_est_{sem_type}")


def bup_with_normal_unif_data(n_boot: int):
    # true effect to estimate
    effect_true: NDArray = get_amat()
    # dot_true = make_dot(effect_true)
    # save_dot(dot_true, name="bup_true")

    # generate sample data from the adjacency matrix given above
    X_full: pd.DataFrame = generate_normal_unif(effect_true)
    # let x6 unobservable
    X = X_full.drop("x6", axis="columns")
    n_variable: Final[int] = X.shape[1]

    # define lingam model with prior
    prior = make_prior_knowledge(
        n_variables=n_variable,
        sink_variables=[1, 4, 5],
    )
    model = BottomUpParceLiNGAM(prior_knowledge=prior)
    run_model(model=model, data=X, n_boot=n_boot, name_graph="bup_est_normal_unif")


def bup_with_unif_data(n_boot: int):
    # true effect to estimate
    effect_true: NDArray = get_amat()
    # dot_true = make_dot(effect_true)
    # save_dot(dot_true, name="bup_true")

    # generate sample data from the adjacency matrix given above
    X_full: pd.DataFrame = generate_unif(effect_true)
    # let x6 unobservable
    X = X_full.drop("x6", axis="columns")
    n_variable: Final[int] = X.shape[1]

    # define lingam model with prior
    prior = make_prior_knowledge(
        n_variables=n_variable,
        sink_variables=[1, 4, 5],
    )
    model = BottomUpParceLiNGAM(prior_knowledge=prior)
    run_model(model=model, data=X, n_boot=n_boot, name_graph="bup_est_unif")


def rcd_with_sem_data(n_boot: int):
    # true effect to estimate
    effect_true: NDArray = get_amat()
    # dot_true = make_dot(effect_true)
    # save_dot(dot_true, name="bup_true")

    # generate sample data from the adjacency matrix given above
    sem_types: Final[list[str]] = ["gauss", "exp", "gumbel"]
    for sem_type in sem_types:
        print(f"{sem_type} model")
        X_full: pd.DataFrame = generate_data_from_sem(effect_true, sem_type=sem_type)
        # let x6 unobservable
        X = X_full.drop("x6", axis="columns")

        model = RCD()
        run_model(model=model, data=X, n_boot=n_boot, name_graph=f"rcd_est_{sem_type}")


def rcd_with_normal_unif_data(n_boot: int):
    # true effect to estimate
    effect_true: NDArray = get_amat()
    # dot_true = make_dot(effect_true)
    # save_dot(dot_true, name="bup_true")

    # generate sample data from the adjacency matrix given above
    X_full: pd.DataFrame = generate_normal_unif(effect_true)
    # let x6 unobservable
    X = X_full.drop("x6", axis="columns")

    model = RCD()
    run_model(model=model, data=X, n_boot=n_boot, name_graph="rcd_est_normal_unif")


def rcd_with_gamma_unif_data(n_boot: int):
    # true effect to estimate
    effect_true: NDArray = get_amat()
    # dot_true = make_dot(effect_true)
    # save_dot(dot_true, name="bup_true")

    # generate sample data from the adjacency matrix given above
    X_full: pd.DataFrame = generate_gamma_unif(effect_true)
    # let x6 unobservable
    X = X_full.drop("x6", axis="columns")

    model = RCD()
    run_model(model=model, data=X, n_boot=n_boot, name_graph="rcd_est_gamma_unif")


def rcd_with_unif_data(n_boot: int):
    # true effect to estimate
    effect_true: NDArray = get_amat()
    # dot_true = make_dot(effect_true)
    # save_dot(dot_true, name="bup_true")

    # generate sample data from the adjacency matrix given above
    X_full: pd.DataFrame = generate_unif(effect_true)
    # let x6 unobservable
    X = X_full.drop("x6", axis="columns")

    model = RCD()
    run_model(model=model, data=X, n_boot=n_boot, name_graph="rcd_est_unif")


if __name__ == "__main__":
    n_boot: Final[int] = 100
    # bup_with_sem_data(n_boot)
    # bup_with_normal_unif_data(n_boot)
    # bup_with_unif_data(n_boot)
    # rcd_with_sem_data(n_boot)
    # rcd_with_unif_data(n_boot)
    # rcd_with_gamma_unif_data(n_boot)
    rcd_with_normal_unif_data(n_boot)
