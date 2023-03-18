import argparse
import os
from collections import namedtuple
from itertools import cycle
from typing import Dict, List

import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

from pips_utils.util import get_str_formatted_time, ensure_dir

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
sns.set_style("ticks")
sns.set_palette("flare")

plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
plt.rc('text', usetex=True)
plt.rcParams.update({
    # 'figure.figsize': (15, 5),
    'figure.titlesize': '20',
    'axes.titlesize': '22',
    'legend.title_fontsize': '16',
    'legend.fontsize': '14',
    'axes.labelsize': '18',
    'xtick.labelsize': '16',
    'ytick.labelsize': '16',
    'figure.dpi': 200,
})


def average_displacement_error(trajectory_a: torch.Tensor, trajectory_b: torch.Tensor) -> float:
    """
    Computes the average displacement error between two trajectory tensors.

    Parameters
    ----------
    trajectory_a : torch.Tensor
        A 2D tensor representing the first trajectory. Its shape should be (S, D),
        where S is the number of time steps and D is the number of dimensions.
    trajectory_b : torch.Tensor
        A 2D tensor representing the second trajectory. Its shape should be (S, D),
        where S is the number of time steps and D is the number of dimensions.

    Returns
    -------
    float
        The average displacement error between the two trajectories, computed as the
        mean L2 norm of the element-wise difference between the two trajectories.

    Raises
    ------
    AssertionError
        If either of the input tensors is not a 2D tensor, or if they do not have
        the same shape.

    Examples
    --------
    >>> trajectory_a = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    >>> trajectory_b = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    >>> average_displacement_error(trajectory_a, trajectory_b)
    1.4142135381698608
    """
    assert trajectory_a.ndim == trajectory_b.ndim == 2, "Input tensors must be 2D tensors"
    assert trajectory_a.shape == trajectory_b.shape, "Input tensors must have the same shape"
    return (trajectory_a - trajectory_b).norm(dim=1).mean().item()


def extract_visible_trajectory(trajectory_a: torch.Tensor, trajectory_b: torch.Tensor,
                               visibility: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extracts the visible portion of two trajectory tensors according to a visibility mask.

    Parameters
    ----------
    trajectory_a : torch.Tensor
        A 2D tensor representing the first trajectory. Its shape should be (S, D),
        where S is the number of time steps and D is the number of dimensions.
    trajectory_b : torch.Tensor
        A 2D tensor representing the second trajectory. Its shape should be (S, D),
        where S is the number of time steps and D is the number of dimensions.
    visibility : torch.Tensor
        A 1D tensor representing the visibility of each time step. Its length should be S.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        A tuple of two 2D tensors representing the visible portion of the input trajectories.
        The output tensor shapes are (N, D), where N is the number of visible time steps
        and D is the number of dimensions.

    Examples
    --------
    >>> trajectory_a = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    >>> trajectory_b = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    >>> visibility = torch.tensor([1, 0, 1])
    >>> extract_visible_trajectory_chain(trajectory_a, trajectory_b, visibility)
    (tensor([[0., 0.], [2., 2.]]), tensor([[1., 1.], [3., 3.]]))
    """
    return trajectory_a[visibility == 1], trajectory_b[visibility == 1]


def extract_visible_trajectory_chain(trajectory_a: torch.Tensor, trajectory_b: torch.Tensor,
                                     visibility: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extracts the visible portion of two trajectory tensors according to a visibility mask.

    Parameters
    ----------
    trajectory_a : torch.Tensor
        A 2D tensor representing the first trajectory. Its shape should be (S, D),
        where S is the number of time steps and D is the number of dimensions.
    trajectory_b : torch.Tensor
        A 2D tensor representing the second trajectory. Its shape should be (S, D),
        where S is the number of time steps and D is the number of dimensions.
    visibility : torch.Tensor
        A 1D tensor representing the visibility of each time step. Its length should be S.

    Returns
    -------
    tuple
        A tuple of two 2D tensors representing the visible portion of the input trajectories.
        If the entire trajectories are visible, this is just the original input trajectories.

    Raises
    ------
    AssertionError
        If the visibility tensor is not a 1D tensor, or if it does not have the same length as
        the input trajectories.
        If the visibility tensor indicates that the first time step is occluded.

    Examples
    --------
    >>> trajectory_a = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    >>> trajectory_b = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    >>> visibility = torch.tensor([1, 1, 0])
    >>> extract_visible_trajectory_chain(trajectory_a, trajectory_b, visibility)
    (tensor([[0., 0.], [1., 1.]]), tensor([[1., 1.], [2., 2.]]))
    """
    assert visibility.ndim == 1, "Visibility tensor must be a 1D tensor"
    assert len(visibility) == len(trajectory_a) == len(trajectory_b), "Input tensors must have the same length"
    occluded_indices = (visibility == 0).nonzero()
    if len(occluded_indices) > 0:
        first_occluded_index = occluded_indices[0].item()
        assert first_occluded_index != 0, "The first index must not be occluded"
        trajectory_a = trajectory_a[:first_occluded_index]
        trajectory_b = trajectory_b[:first_occluded_index]
    return trajectory_a, trajectory_b


def compute_summary(results: Dict) -> Dict:
    """
    Computes a summary of the trajectory prediction results.

    Parameters
    ----------
    results : Dict
        A dictionary containing the trajectory prediction results. It should have the following keys:
        - 'valids': A 1D boolean tensor indicating which points in the trajectory are valid.
        - 'trajectory_gt': A 2D tensor representing the ground-truth trajectory. Its shape should be (S, D),
          where S is the number of time steps and D is the number of dimensions.
        - 'trajectory_pred': A 2D tensor representing the predicted trajectory. Its shape should be (S, D),
          where S is the number of time steps and D is the number of dimensions.
        - 'visibility_gt': A 1D tensor representing the visibility of each time step. Its length should be S.

    Returns
    -------
    Dict
        A dictionary containing the computed summary statistics, with the following keys:
        - 'idx': A string representing the trajectory index, in the format "<iter>--<video_idx>--<point_idx_in_video>".
        - 'ade': The average displacement error between the ground-truth and predicted trajectories.
        - 'ade_visible': The average displacement error between the visible portion
           of the ground-truth and predicted trajectories.
        - 'ade_visible_chain': The average displacement error between the first visible chain
           of the ground-truth and predicted trajectories.
        - 'n_timesteps': The length of the trajectory.
        - 'n_timesteps_visible': The number of visible points in the ground-truth trajectory.
        - 'n_timesteps_visible_chain': The number of visible points in the ground-truth trajectory, assuming a chain structure.

    Examples
    --------
    >>> results = {
    ...     'iter': 123,
    ...     'video_idx': 2,
    ...     'point_idx_in_video': 31,
    ...     'valids': torch.tensor([True, True, True]),
    ...     'trajectory_gt': torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
    ...     'trajectory_pred': torch.tensor([[0.0, 0.0], [2.0, 2.0], [3.0, 3.0]]),
    ...     'visibility_gt': torch.tensor([1, 1, 0])
    ... }
    >>> compute_summary(results)
    {
        'idx': '123--2--31',
        'ade': 0.9428090453147888,
        'ade_visible': 0.7071067690849304,
        'ade_visible_chain': 0.7071067690849304,
        'n_timesteps': 3,
        'n_timesteps_visible': 2,
        'n_timesteps_visible_chain': 2
    }
    """
    assert results["valids"].all(), "Assume all points are valid for each timestep"
    traj_gt = results["trajectory_gt"]
    traj_pred = results["trajectory_pred"]
    vis_gt = results["visibility_gt"]
    traj_gt_visible, traj_pred_visible = extract_visible_trajectory(traj_gt, traj_pred, vis_gt)
    traj_gt_visible_chain, traj_pred_visible_chain = extract_visible_trajectory_chain(traj_gt, traj_pred, vis_gt)
    assert vis_gt.sum() == len(traj_gt_visible)
    assert vis_gt.sum() >= len(traj_gt_visible_chain)
    summary = {
        "idx": f'{results["iter"]}--{results["video_idx"]}--{results["point_idx_in_video"]}',
        "ade": average_displacement_error(traj_gt, traj_pred),
        "ade_visible": average_displacement_error(traj_gt_visible, traj_pred_visible),
        "ade_visible_chain": average_displacement_error(traj_gt_visible_chain, traj_pred_visible_chain),
        "n_timesteps": len(traj_gt),
        "n_timesteps_visible": len(traj_gt_visible),
        "n_timesteps_visible_chain": len(traj_gt_visible_chain),
    }
    return summary


def compute_summary_df(results_list: List[Dict]) -> pd.DataFrame:
    """
    Computes a summary dataframe of the results of multiple trajectory prediction experiments.

    Parameters:
        results_list (List[Dict]): a list of dictionaries, where each dictionary has the same format as the dictionary
                                   returned by the compute_summary function.

    Returns:
        summary_df (pd.DataFrame): a dataframe containing the following columns:
            - 'idx' (str): a unique identifier for each trajectory.
            - 'ade' (float): the average displacement error between the ground truth and predicted trajectories.
            - 'ade_visible' (float): the average displacement error between the visible parts
                                     of the ground truth and predicted trajectories.
            - 'ade_visible_chain' (float): the average displacement error between the visible trajectory chains
                                           of the ground truth and predicted trajectories.
            - 'n_timesteps' (int): the length of the trajectory.
            - 'n_timesteps_visible' (int): the number of visible points in the ground truth trajectory.
            - 'n_timesteps_visible_chain' (int): the number of visible points in the visible trajectory chain
                                       of the ground truth trajectory.
    """
    summaries = []
    for results in results_list:
        summaries += [compute_summary(results)]
    return pd.DataFrame.from_records(summaries)


def figure1(
        df: pd.DataFrame,
        output_path: str,
        log_y: bool = False,
        save_pdf: bool = True,
        name: str = "figure1",
        title: str = rf"ADE per visible chain length (w/ 95\% CI)"
) -> None:
    df = df.copy()
    df = df[df.n_timesteps_visible_chain > 1]

    fig, ax = plt.subplots(figsize=(6, 4))
    fig.suptitle(title)
    sns.lineplot(
        df,
        x="n_timesteps_visible_chain",
        y="ade_visible_chain",
        hue="name",
        palette=cycle(["GoldenRod", "r", "forestgreen", "yellow"]),
        linestyle="-",
        linewidth=2,
        errorbar=("ci", 95),
        markers=True,
        dashes=False,
        err_style="band",
        # err_style="bars", err_kws={"fmt": 'o', "linewidth": 2, "capsize": 6},
        alpha=1,
        ax=ax,
    )
    ax.set_xlabel(rf"visibility chain length")
    ax.set_ylabel(rf"ADE")
    if log_y:
        plt.yscale('log')
    ax.legend_.set_title(None)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"{name}.png"))
    if save_pdf:
        plt.savefig(os.path.join(output_path, f"PDF__{name}.pdf"))
    plt.show()
    plt.close()
    plt.clf()


def figure2(
        df: pd.DataFrame,
        output_path: str,
        ade_metric: str = "ade_visible",
        log_y: bool = False,
        save_pdf: bool = True,
        name: str = "figure2",
        title: str = rf"ADE for mostly visible (w/ 95\% CI)"
) -> None:
    df = df.copy()
    df_list = []
    for mostly_visible_threshold in range(1, int(df.n_timesteps_visible.max()) + 1):
        df_ = df.copy()
        df_["mostly_visible_threshold"] = mostly_visible_threshold
        df_["mostly_visible"] = (df_.n_timesteps_visible >= mostly_visible_threshold).apply(
            lambda x: "Mostly Visible" if x else "Mostly Occluded")
        df_list += [df_]

        df_ = df.copy()
        df_["mostly_visible_threshold"] = mostly_visible_threshold
        df_["mostly_visible"] = "All"
        df_list += [df_]

    df = pd.concat(df_list).reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(title)
    mostly_visible_types = ["Mostly Visible", "Mostly Occluded", "All"]
    mostly_visible_linestlye = ["--", ":", "-"]
    for mostly_visible, linestyle in zip(mostly_visible_types, mostly_visible_linestlye, strict=True):
        sns.lineplot(
            df[df.mostly_visible == mostly_visible],
            x="mostly_visible_threshold",
            y=ade_metric,
            hue="name",
            palette=cycle(["GoldenRod", "r", "forestgreen", "yellow"]),
            linestyle=linestyle,
            linewidth=2,
            errorbar=("ci", 95),
            markers=True,
            dashes=False,
            err_style="band",
            alpha=1,
            legend=mostly_visible == "All",
            ax=ax,
        )
    texts = [t.get_text() for t in ax.get_legend().get_texts()] + mostly_visible_types
    lines = ax.get_legend().get_lines() + [Line2D([0, 10], [0, 10], linewidth=2, color="black", linestyle=linestyle)
                                           for linestyle in mostly_visible_linestlye]
    new_legend = plt.legend(lines, texts, loc="center left")
    ax.add_artist(new_legend)
    ax.set_xlabel(rf"mostly visible threshold")
    ax.set_ylabel(rf"ADE")
    if log_y:
        plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"{name}_{ade_metric}.png"))
    if save_pdf:
        plt.savefig(os.path.join(output_path, f"PDF__{name}_{ade_metric}.pdf"))
    plt.show()
    plt.close()
    plt.clf()


def figure3(
        df: pd.DataFrame,
        output_path: str,
        log_y: bool = False,
        save_pdf: bool = True,
        name: str = "figure3",
        title: str = rf"ADE per number of visible points (w/ 95\% CI)"
) -> None:
    df = df.copy()
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.suptitle(title)
    MetricLabelStyle = namedtuple("MetricLabelStle", ["metric", "label", "style"])
    metric_label_style_list = [MetricLabelStyle("ade", "ADE", "-"), MetricLabelStyle("ade_visible", "ADE Visible", ":")]
    for metric_label_style in metric_label_style_list:
        sns.lineplot(
            df,
            x="n_timesteps_visible",
            y=metric_label_style.metric,
            hue="name",
            palette=cycle(["GoldenRod", "r", "forestgreen", "yellow"]),
            linestyle=metric_label_style.style,
            linewidth=2,
            errorbar=("ci", 95),
            markers=True,
            dashes=False,
            err_style="band",
            alpha=1,
            legend=metric_label_style.metric == "ade",
            ax=ax,
        )
    texts = [t.get_text() for t in ax.get_legend().get_texts()] + [mls.label for mls in metric_label_style_list]
    lines = ax.get_legend().get_lines() + [Line2D([0, 10], [0, 10], linewidth=2, color="black", linestyle=mls.style)
                                           for mls in metric_label_style_list]
    new_legend = plt.legend(lines, texts, loc="center left")
    ax.add_artist(new_legend)
    ax.set_xlabel(rf"number of visible points")
    ax.set_ylabel(rf"ADE")
    if log_y:
        plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"{name}.png"))
    if save_pdf:
        plt.savefig(os.path.join(output_path, f"PDF__{name}.pdf"))
    plt.show()
    plt.close()
    plt.clf()


def table1(
        df: pd.DataFrame,
        output_path: str,
        ade_metric: str = "ade_visible",
        mostly_visible_threshold: int = 4,
        name: str = "table1"
) -> None:
    mostly_visible_indices = df.n_timesteps_visible >= mostly_visible_threshold
    mostly_visible_ade_df = df[mostly_visible_indices][["name", ade_metric]].groupby("name").mean()
    mostly_occluded_ade_df = df[~mostly_visible_indices][["name", ade_metric]].groupby("name").mean().rename(
        columns={ade_metric: ade_metric + "_occluded"})
    table_df = pd.merge(mostly_visible_ade_df, mostly_occluded_ade_df, left_index=True, right_index=True)
    table_df = table_df.sort_values(ade_metric, ascending=False)
    table_df.to_csv(os.path.join(output_path, f"{name}_threshold-{mostly_visible_threshold}_metric-{ade_metric}.csv"))
    print(f"TABLE: '{name}' (metric={ade_metric})")
    print(table_df)
    print()


def table2(
        df: pd.DataFrame,
        output_path: str,
        mostly_visible_threshold: int = 4,
        add_legacy_metrics: bool = True,
        create_heatmap: bool = True,
        name: str = "table1"
) -> None:
    df = df.copy()
    df_list = []

    df_global_ade = df[["name", "ade", "ade_visible", "ade_occluded", "ade_visible_chain"]].groupby(["name"]).mean()
    df_list += [df_global_ade]

    for chain_length in [2, 4, 8]:
        if chain_length not in df.n_timesteps_visible_chain.unique():
            continue
        df_chain_ade = df[df.n_timesteps_visible_chain == chain_length][["name", "ade_visible_chain"]].groupby(
            ["name"]).mean().rename(columns={"ade_visible_chain": f"ade_visible_chain_{chain_length}"})
        df_list += [df_chain_ade]

    # Legacy metrics (reported in paper for FlyingThings++ with threshold 4 and for CroHD with threshold 8):
    #    1. ADE of Mostly Visible Trajectories (reported as "Vis." in paper)
    #    2. ADE of Mostly Occluded Trajectories (reported as "Occ." in paper)
    # These legacy metrics were computed as the average of pre-iteration metrics
    if add_legacy_metrics:
        df["iter"] = df.idx.apply(lambda x: int(x.split("--")[0]))
        df["mostly_visible"] = df.n_timesteps_visible >= mostly_visible_threshold
        df_average_per_iter = df.groupby(["iter", "mostly_visible", "name"]).mean().reset_index()
        df_mostly_visible_ade = df_average_per_iter[df_average_per_iter.mostly_visible][["name", "ade"]].groupby(
            "name").mean().rename(columns={"ade": "ade_mostly_visible"})
        df_mostly_occluded_ade = df_average_per_iter[~df_average_per_iter.mostly_visible][["name", "ade"]].groupby(
            "name").mean().rename(columns={"ade": "ade_mostly_occluded"})
        df_list += [df_mostly_visible_ade, df_mostly_occluded_ade]

    table_df = df_list[0]
    for df_i in df_list[1:]:
        table_df = pd.merge(table_df, df_i, left_index=True, right_index=True)
        assert len(table_df) == len(df_list[0])

    table_df = table_df.sort_values("ade_visible", ascending=False)
    table_df.to_csv(os.path.join(output_path, f"{name}_threshold-{mostly_visible_threshold}.csv"))

    print(f"TABLE: '{name}'")
    print(table_df)
    print()

    if create_heatmap:
        fig, ax = plt.subplots(figsize=(7, 1.5 + 1 * len(table_df)))
        fig.suptitle(name)
        sns.heatmap(table_df, annot=True, linewidths=0.3, fmt=".2f", norm=LogNorm(vmin=3, vmax=80))
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"{name}.png"))
        plt.show()
        plt.close()
        plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_df_path_list", nargs='+', required=True,
                        help="List of paths to results dataframes of contining evaluation results of different runs.")
    parser.add_argument("--results_name_list", nargs='+', required=True,
                        help="List of names to be used to identify each run.")
    parser.add_argument("--mostly_visible_threshold", type=int, default=4,
                        help='Threshold used to define when a trajectory is "mostly visible" '
                             'compared to being "mostly occluded". The trajectory is mostly visible '
                             'if there is a number of visible points greater or equal to the threshold. '
                             'Otherwise, it is considered as mostly occluded. '
                             'This value was set to 4 for the FlyingThings++ dataset and to 8 for the CroHD dataset, '
                             'in the numbers reported in the paper')
    parser.add_argument("--output_path", type=str, default=f"logs/figures/{get_str_formatted_time()}",
                        help="path to folder to save gifs to")
    args = parser.parse_args()

    assert len(args.results_df_path_list) == len(args.results_name_list)
    assert len(args.results_name_list) == len(set(args.results_name_list))
    ensure_dir(args.output_path)

    results_df_list = []
    for path, name in zip(args.results_df_path_list, args.results_name_list, strict=True):
        df = pd.read_csv(path)
        df["name"] = name
        print(f"Loaded results df with name `{name}` from path `{path}`.")
        print(df.describe())
        print()
        results_df_list += [df]
    df = pd.concat(results_df_list)

    # table1(df, args.output_path, "ade", name="withquery__table1")
    # table1(df, args.output_path, "ade_visible", name="withquery__table1")
    # figure1(df, args.output_path, name="withquery__figure1")
    # figure2(df, args.output_path, "ade", name="withquery__figure2")
    # figure2(df, args.output_path, "ade_visible", name="withquery__figure2")
    # figure3(df, args.output_path, name="withquery__figure3")

    # TODO Ad hoc fix: ADE only for non-query points
    df.ade = df.ade * df.n_timesteps / (df.n_timesteps - 1)
    df.ade_visible = df.ade_visible * df.n_timesteps_visible / (df.n_timesteps_visible - 1)
    df.ade_visible_chain = df.ade_visible_chain * df.n_timesteps_visible_chain / (df.n_timesteps_visible_chain - 1)

    # Compute `ade_occluded`
    df["ade_occluded"] = (df.ade * df.n_timesteps - df.ade_visible * df.n_timesteps_visible) / (
            df.n_timesteps - df.n_timesteps_visible)

    # table1(df, args.output_path, "ade", name="table1")
    # table1(df, args.output_path, "ade_visible", name="table1")
    table2(df, args.output_path, args.mostly_visible_threshold, name="table2")

    figure1(df, args.output_path, name="figure1")
    # figure1(df, args.output_path, name="log__figure1")
    # figure2(df, args.output_path, "ade", name="figure2")
    # figure2(df, args.output_path, "ade_visible", name="figure2")
    # figure2(df, args.output_path, "ade", log_y=True, name="log__figure2")
    # figure2(df, args.output_path, "ade_visible", log_y=True, name="log__figure2")
    figure3(df, args.output_path, name="figure3")
