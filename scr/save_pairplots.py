import pandas as pd
import seaborn as sns
from fns import get_amat
from generate_data import generate_data_from_sem, generate_gamma_unif, generate_normal_unif, generate_unif


def save_pairplot(df: pd.DataFrame, name: str = "pairplot"):
    sns.pairplot(df).savefig(f"graph/pairplot_{name}.png")
    # print(df.describe())


def save_all():
    amat = get_amat()
    dfs: list[pd.DataFrame] = [generate_gamma_unif(amat), generate_normal_unif(amat), generate_unif(amat)]
    names: list[str] = ["gamma_unif", "normal_unif", "unif"]
    sem_types: list[str] = ["gauss", "exp", "gumbel"]
    dfs += [generate_data_from_sem(amat, sem_type=sem_type) for sem_type in sem_types]
    names += sem_types
    for name, df in zip(names, dfs):
        save_pairplot(df, name)


if __name__ == "__main__":
    save_all()
