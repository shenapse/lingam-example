import inspect
import pathlib
import time
from typing import Any, Final, Optional

import lingam
import numpy as np
import pandas as pd
from graphviz import Digraph
from lingam.utils import make_dot
from nptyping import NDArray


def show_caller():
    chain: list[str] = [s.function for s in inspect.stack()[1:-1]]
    caller: str = " -> ".join(chain)
    print(f"running: {caller}")


def make_dot_from_dag(dag: dict) -> Digraph:
    graph = Digraph(format="png")
    graph.attr("node", shape="circle")
    for from_, to_ in zip(dag["from"], dag["to"]):
        graph.edge(str(from_), str(to_))
    return graph


def save_dot(dot: Digraph, name: str, dir: pathlib.Path | None = None, format: str = "png") -> str:
    """save Digraph object as png file."""
    dot.format = format
    dir_dot: pathlib.Path = dir if dir is not None else pathlib.Path.cwd() / "graph"
    if not dir_dot.exists():
        raise ValueError(f"no such directory exists. {dir_dot.resolve()}")
    return dot.render(filename=name, directory=str(dir_dot.resolve()))


def get_causal_effects_df(causal_effect: dict[str, Any], labels: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(causal_effect)
    df["from"] = df["from"].apply(lambda x: labels[x])
    df["to"] = df["to"].apply(lambda x: labels[x])
    return df


def make_prior_knowledge_graph(prior_knowledge_matrix: NDArray) -> Digraph:
    d = Digraph(engine="dot")
    labels = [f"x{i}" for i in range(prior_knowledge_matrix.shape[0])]
    for label in labels:
        d.node(label, label)

    dirs = np.where(prior_knowledge_matrix < 0)
    for to, from_ in zip(dirs[0], dirs[1]):
        if to != from_:
            d.edge(labels[from_], labels[to], style="dashed")
    return d


# def get_adjacency_matrix(result: lingam.BootstrapResult, n_variable: int, min_causal_effect: float = 0.1) -> NDArray:
#     causal_effects = result.get_total_causal_effects(min_causal_effect=min_causal_effect)
#     mat = np.zeros((n_variable, n_variable))
#     for i, from_ in enumerate(causal_effects["from"]):
#         to_: int = causal_effects["to"][i]
#         mat[to_, from_] = causal_effects["effect"][i]
#     return mat


def get_causal_paths_df(
    result: lingam.BootstrapResult,
    labels: Optional[list[str]] = None,
    min_causal_effect: Optional[float] = None,
    min_prob: Optional[float] = 0.0,
) -> pd.DataFrame:
    """calculate all causal paths from a bootstrap"""
    df = pd.DataFrame(columns=["from", "to", "effect", "probability"])
    cdc = result.get_causal_direction_counts(min_causal_effect=min_causal_effect)
    for i, from_ in enumerate(cdc["from"]):
        to_ = cdc["to"][i]
        path = result.get_paths(from_, to_)
        if (prob := path["probability"][0]) < min_prob:
            continue
        df_tmp = pd.DataFrame({"from": from_, "to": to_, "effect": path["effect"][0], "probability": prob}, index=[i])
        df = df.append(df_tmp)
    if labels is not None:
        df["from"] = df["from"].apply(lambda x: labels[x])
        df["to"] = df["to"].apply(lambda x: labels[x])
    return df.sort_values("probability", ascending=False)


def get_adjacency_matrix(
    result: lingam.BootstrapResult,
    n_variable: int | None = None,
    min_causal_effect: Optional[float] = None,
    min_prob: float = 0.0,
) -> NDArray:
    """calculate adjacency matrix from bootstrap object"""
    cdc = result.get_causal_direction_counts(min_causal_effect=min_causal_effect)
    # set up zero matrix to be filled later
    # if n_variable is not provided, the size of matrix is chosen as the minimum one capable of accommodating all variables appearing in cdc
    mat_size: Final[int] = (
        n_variable if n_variable is not None else max([max(t) for t in zip(cdc["from"], cdc["to"])]) + 1
    )
    mat = np.zeros((mat_size, mat_size))
    # fill cells in matrix
    for from_, to_ in zip(cdc["from"], cdc["to"]):
        path = result.get_paths(from_, to_)
        if path["probability"][0] < min_prob or abs((effect := path["effect"][0])) < min_causal_effect:
            continue
        mat[to_, from_] = effect
    return mat


def define_adjacency_mat(effects: list[tuple[tuple[int, int], float]]) -> NDArray:
    # largest index of nodes
    # used for defining matrix size
    node_max: int = max([max(nodes) for nodes, _ in effects])
    mat = np.zeros((node_max + 1, node_max + 1))
    for nodes, weight in effects:
        to_, from_ = nodes
        mat[to_, from_] = weight
    return mat


def get_suspected_dependence(mat: NDArray, p_value_thr: float = 0.1, labels: list[str] | None = None) -> pd.DataFrame:
    """find possible dependent pair(s) of variables from the matrix of error independence p-values."""
    size: Final[int] = mat.shape[0]
    names = list(range(size)) if labels is None or len(labels) == size else labels
    variables = []
    pvs = []
    for i in range(size):
        for j in range(i + 1, size):
            if mat[i, j] > p_value_thr:
                continue
            variables.append((names[i], names[j]))
            pvs.append(mat[i, j])
    return pd.DataFrame({"variables": variables, "p-value": pvs}, index=list(range(len(pvs))))


def get_amat() -> NDArray:
    mat: list[tuple[tuple[int, int], float]] = [
        # e.g., node3 <- node6 with weight 2.0
        ((3, 6), 2.0),
        ((0, 3), 0.5),
        ((2, 6), 2.0),
        ((1, 0), 0.5),
        ((1, 2), 0.5),
        ((5, 0), 0.5),
        ((5, 6), 0.05),
        ((4, 0), 0.5),
        ((4, 2), -0.5),
    ]
    return define_adjacency_mat(mat)


def get_amat_from_model(model):
    cands: list[str] = ["_adjacency_matrix", "adjacency_matrix_"]
    for key, value in model.__dict__.items():
        for cand in cands:
            if key == cand:
                return value


def run_model(
    model,
    data: pd.DataFrame,
    n_boot: int,
    name_graph: str,
    dir_graph: pathlib.Path | None = None,
    min_prob: float = 0.1,
    min_causal_effect: float = 0.1,
    n_dags: int = 3,
):
    # show caller function
    show_caller()
    # fit
    model.fit(data)
    # print causal order if it is available
    if "causal_order_" in model.__dict__.keys():
        print(f"causal order = {model.causal_order_}")
    # print adjacency matrix if it is available
    amat = get_amat_from_model(model)
    print("adjacency matrix\n", amat)
    # independence p-values
    pvs_mat = model.get_error_independence_p_values(data)
    print("independence p-values\n", pvs_mat)
    p_thr = 0.1
    print(f"suspected dependence p<{p_thr}\n", get_suspected_dependence(pvs_mat, p_value_thr=p_thr))
    # when model misses a prior knowledge, bootstrap takes a lot of time
    # and it is less likely that the estimation provides fruitful information
    start = time.perf_counter()
    result = model.bootstrap(data, n_sampling=n_boot)
    process_time = time.perf_counter() - start
    print(f"bootstrap takes {process_time} sec.")
    # show effect table
    # df = get_causal_paths_df(result, min_prob=min_prob, min_causal_effect=min_causal_effect)
    # print("causal_paths\n", df)
    # save estimated graph as png image
    # dot_est = make_dot(get_adjacency_matrix(result, min_prob=min_prob, min_causal_effect=min_causal_effect))
    dot_est = make_dot(amat)
    save_dot(dot_est, name=name_graph, dir=dir_graph)

    # save detected dags
    dags = result.get_directed_acyclic_graph_counts(n_dags=n_dags, min_causal_effect=min_causal_effect)
    for i, dag in enumerate(dags["dag"]):
        freq = dags["count"][i] / n_boot
        if freq < min_prob:
            continue
        dot = make_dot_from_dag(dag)
        save_dot(dot, str(f"{name_graph}_dag{i}_freq={round(freq,3)}"))
