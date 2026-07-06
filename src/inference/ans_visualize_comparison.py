import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib.lines import Line2D

from itertools import product

from ..utils.myplot import figure_grid, figure_save
from ..utils.myutils import pickle_save, pickle_load
from ..utils.mymath import get_r_squared

from ..config_ans.constant import PARAM_SYMBOL
from ..datamanagekit.ijhcs25_data import ExperimentIJHCS

from scipy import stats
import seaborn as sns

from matplotlib.lines import Line2D

import warnings
warnings.filterwarnings("ignore", message=".*use_inf_as_na.*")

DIR_TO_DATA = Path(__file__).parent.parent.parent

# BASE = "AnSBaseAmort_PTE_07290953"
# COND = "AnSCondAmort_PTE_08071040"
BASE = 'AnSBaseAmort_PTE_01140125'
COND = 'AnSCondAmort_PTE_01151337'

BASE_COLOR = "#F33276"
COND_COLOR = "#0000FE"
ORIG_COLOR = "#008000"

PARAM_LIST = ["theta_m", "theta_c", "theta_p", "theta_s", 
              "rew_succ", "rew_fail", "decay_succ", "decay_fail"]



def within_subject_lineplot(
    data, x, y, ax, color,
    subject='user',
    linewidth=1, markersize=6,
    errorbar=('ci', 95),
    err_style='bars', err_kws=None,
    x_offset=0.0, marker='s'
):
    if err_kws is None:
        err_kws = {'capsize': 3}

    levels = sorted(data[x].dropna().unique())
    k = len(levels)
    n_subj = data[subject].nunique()

    if k < 2:
        raise ValueError(f"{x} needs at least 2 levels for within-subject CI.")

    # Step 1: Cousineau normalization
    grand_mean = data[y].mean()
    df_norm = data.copy()
    df_norm['dv_norm'] = data[y] - data.groupby(subject)[y].transform('mean') + grand_mean

    # Step 2: Morey-corrected SE
    group = df_norm.groupby(x)['dv_norm']
    means = group.mean()
    ses = group.sem()
    ses *= np.sqrt(k / (k - 1))

    # Step 3: CI using t-distribution
    alpha = 1 - errorbar[1] / 100
    tcrit = stats.t.ppf(1 - alpha / 2, df=n_subj - 1)
    ci = ses * tcrit

    # Step 4: Plot
    x_pos = np.array(levels, dtype=float) + x_offset
    ax.errorbar(
        x=x_pos,
        y=means.values,
        yerr=[ci.values, ci.values],
        fmt=marker,
        linewidth=linewidth,
        markersize=markersize,
        color=color,
        linestyle='-',
        capsize=err_kws.get('capsize', 3)
    )




def between_subject_lineplot(*, data, x, y, ax,
                              color='k',
                              marker='s',
                              linewidth=1,
                              markersize=7,
                              errorbar=('ci', 95),
                              err_style='bars',
                              err_kws={'capsize': 3},
                              x_offset=0.0):
    """
    seaborn lineplot wrapper with adjustable offset
    """
    data = data.copy()
    data['adjusted_x'] = data[x] + x_offset
    sns.lineplot(
        data=data,
        x='adjusted_x',
        y=y,
        marker=marker,
        color=color,
        linewidth=linewidth,
        markersize=markersize,
        errorbar=errorbar,
        err_style=err_style,
        err_kws=err_kws,
        ax=ax
    )


def horizontal_line_with_shading(data,
                                  y,
                                  ax,
                                  color='k',
                                  linewidth=1,
                                  alpha=0.2,
                                  errorbar=('ci', 95)):
    ydata = data[y].dropna().values
    mean = np.mean(ydata)
    ci_level = errorbar[1]
    alpha_val = 1 - ci_level / 100

    # CI 계산 (using t-distribution)
    from scipy import stats
    n = len(ydata)
    if n < 2:
        return  # CI 불가능한 경우 skip

    se = stats.sem(ydata)
    tcrit = stats.t.ppf(1 - alpha_val / 2, n - 1)
    ci = tcrit * se

    lower = mean - ci
    upper = mean + ci

    ax.axhline(mean, color=color, linewidth=linewidth)
    ax.axhspan(lower, upper, color=color, alpha=alpha)




def visualize_parameter_comparison():
    figpath = os.path.join(DIR_TO_DATA, f"data/visualization/inference/ans_infer_comparison")
    os.makedirs(figpath, exist_ok=True)

    feature_order = ["tier", "col_level", "sen_level", "spd_level", "rad_level2"]
    xtick_list = [
        ["Pro.", "Ama."],
        ["White", "Gray"],
        ["Habitual", "Default"],
        ["Stat.", "Moving"],
        ["Large", "Small"]
    ]
    xlabel_list = ["Player group",  "Target color", "Mouse sensi.", "Target speed", "Target radius"]
    x_offsets = [-0.05, 0.05]


    fig, axs = figure_grid(8, 5, size_ax=np.array([3.5, 2.5]) / 2.0)

    # ---------- 1) Conditional ----------
    inv_df  = pd.read_csv(os.path.join(DIR_TO_DATA, f"data/ijhcs25_data/infer_result/{COND}/inv_param.csv"))
    var1_df = pd.read_csv(os.path.join(DIR_TO_DATA, f"data/ijhcs25_data/infer_result/{COND}/var1_param.csv"))
    var2_df = pd.read_csv(os.path.join(DIR_TO_DATA, f"data/ijhcs25_data/infer_result/{COND}/var2_param.csv"))
    var2_df["rad_level2"] = var2_df["rad_level"].map({0:1, 1:0})

    # ---------- 2) Baseline ----------
    param_df_base = pd.read_csv(os.path.join(DIR_TO_DATA, f"data/ijhcs25_data/infer_result/{BASE}/param.csv"))
    param_df_base["rad_level2"] = param_df_base["rad_level"].map({0:1, 1:0})

    # ---------- 2) Baseline ----------
    orig_df = pd.read_csv(os.path.join(DIR_TO_DATA, f"data/ijhcs25_data/summary/original_inference_result_revised.csv"))
    orig_df["tier"] = orig_df["group"].map({"PRO": 0, "AMA": 1})
    orig_df["col_level"] = orig_df["color"].map({"white":0, "gray":1})
    orig_df["sen_level"] = orig_df["sensi"].map({"custom": 0, "default": 1})

    for (i, param), j in product(enumerate(PARAM_LIST), range(len(xlabel_list))):
        ax = axs[i][j]
        xfac = feature_order[j]

        # 1. theta_m, theta_c
        if param in ["theta_m", "theta_c"]:
            if xfac == "tier":
                between_subject_lineplot(
                    data=inv_df, x=xfac, y=param, ax=ax, color=COND_COLOR, x_offset=-0.05, marker='s'
                )
                between_subject_lineplot(
                    data=param_df_base, x=xfac, y=param, ax=ax, color=BASE_COLOR, x_offset=0.0, marker='D'
                )
                # between_subject_lineplot(
                #     data=orig_df, x=xfac, y=param, ax=ax, color=ORIG_COLOR, x_offset=0.05, marker='o'
                # )
            
            else:
                horizontal_line_with_shading(
                    data=inv_df, y=param, ax=ax, color=COND_COLOR
                )
                if xfac in ["col_level", "sen_level"]:
                    within_subject_lineplot(
                        data=param_df_base, x=xfac, y=param, ax=ax, color=BASE_COLOR, x_offset=-0.05, marker='D'
                    )
                    # within_subject_lineplot(
                    #     data=orig_df, x=xfac, y=param, ax=ax, color=ORIG_COLOR, x_offset=0.05, marker='o'
                    # )
                else:
                    within_subject_lineplot(
                        data=param_df_base, x=xfac, y=param, ax=ax, color=BASE_COLOR, x_offset=0.0, marker='D'
                    )
                    # horizontal_line_with_shading(
                    #     data=orig_df, y=param, ax=ax, color=ORIG_COLOR
                    # )


        # 2. theta_p, theta_s
        elif param in ["theta_p", "theta_s"]:
            if xfac == "tier":
                between_subject_lineplot(
                    data=var1_df, x=xfac, y=param, ax=ax, color=COND_COLOR, x_offset=-0.05, marker='s'
                )
                between_subject_lineplot(
                    data=param_df_base, x=xfac, y=param, ax=ax, color=BASE_COLOR, x_offset=0.0, marker='D'
                )
                # between_subject_lineplot(
                #     data=orig_df, x=xfac, y=param, ax=ax, color=ORIG_COLOR, x_offset=0.05, marker='o'
                # )
            elif xfac == "col_level":
                within_subject_lineplot(
                    data=var1_df, x=xfac, y=param, ax=ax, color=COND_COLOR, x_offset=-0.05, marker='s'
                )
                within_subject_lineplot(
                    data=param_df_base, x=xfac, y=param, ax=ax, color=BASE_COLOR, x_offset=0.0, marker='D'
                )
                # within_subject_lineplot(
                #     data=orig_df, x=xfac, y=param, ax=ax, color=ORIG_COLOR, x_offset=0.05, marker='o'
                # )
            elif xfac == "sen_level":
                horizontal_line_with_shading(
                    data=var1_df, y=param, ax=ax, color=COND_COLOR
                )
                within_subject_lineplot(
                    data=param_df_base, x=xfac, y=param, ax=ax, color=BASE_COLOR, x_offset=-0.05, marker='D'
                )
                # within_subject_lineplot(
                #     data=orig_df, x=xfac, y=param, ax=ax, color=ORIG_COLOR, x_offset=0.05, marker='o'
                # )
            else:
                horizontal_line_with_shading(
                    data=var1_df, y=param, ax=ax, color=COND_COLOR
                )
                within_subject_lineplot(
                    data=param_df_base, x=xfac, y=param, ax=ax, color=BASE_COLOR, x_offset=-0.0, marker='D'
                )
                # horizontal_line_with_shading(
                #     data=orig_df, y=param, ax=ax, color=ORIG_COLOR
                # )
                

        # 3. 나머지 4개 파라미터
        else:
            if xfac == "tier":
                between_subject_lineplot(
                    data=var2_df, x=xfac, y=param, ax=ax, color=COND_COLOR, x_offset=-0.05, marker='s'
                )
                between_subject_lineplot(
                    data=param_df_base, x=xfac, y=param, ax=ax, color=BASE_COLOR, x_offset=0.0, marker='D'
                )
                # between_subject_lineplot(
                #     data=orig_df, x=xfac, y=param, ax=ax, color=ORIG_COLOR, x_offset=0.05, marker='o'
                # )
            elif xfac in ["sen_level", "col_level"]:
                within_subject_lineplot(
                    data=var2_df, x=xfac, y=param, ax=ax, color=COND_COLOR, x_offset=-0.05, marker='s'
                )
                within_subject_lineplot(
                    data=param_df_base, x=xfac, y=param, ax=ax, color=BASE_COLOR, x_offset=0.0, marker='D'
                )
                # within_subject_lineplot(
                #     data=orig_df, x=xfac, y=param, ax=ax, color=ORIG_COLOR, x_offset=0.05, marker='o'
                # )
            else:
                within_subject_lineplot(
                    data=var2_df, x=xfac, y=param, ax=ax, color=COND_COLOR, x_offset=-0.05, marker='s'
                )
                within_subject_lineplot(
                    data=param_df_base, x=xfac, y=param, ax=ax, color=BASE_COLOR, x_offset=0.05, marker='D'
                )
                # horizontal_line_with_shading(
                #     data=orig_df, y=param, ax=ax, color=ORIG_COLOR
                # )

        ax.set_xlabel("")
        ax.set_xlim(-0.2, 1.2)
        if j == 0:
            # ax.set_ylabel(f"Inferred {PARAM_SYMBOL[param]}")
            ax.set_ylabel([
                "Motor noise",
                "Clock noise",
                "Position noise",
                "Speed noise",
                "Hit reward",
                "Miss penalty",
                "Reward decay",
                "Penalty decay"
            ][i])
        else:
            ax.set_ylabel("")
        ax.set_xticks(np.arange(2), xtick_list[j])
        ax.grid(axis='y', alpha=0.5, linestyle='--')

        # ylim_min, ylim_max = ax.get_ylim()
        # ax.set_ylim(ylim_min, ylim_max + 0.15 * (ylim_max - ylim_min))

 

    for j, l in enumerate(xlabel_list):
        axs[-1][j].set_xlabel(l)
    
    # for r in range(8):
    #     ylim_min, ylim_max = 999, -999
    #     for c in range(5):
    #         _ylim_min, _ylim_max = axs[r][c].get_ylim()
    #         ylim_min = _ylim_min if _ylim_min < ylim_min else ylim_min
    #         ylim_max = _ylim_max if _ylim_max > ylim_max else ylim_max
    #     for c in range(5):
    #         ax = axs[r][c]
    #         ax.set_ylim(ylim_min - 0.13 * (ylim_max - ylim_min), ylim_max + 0.13 * (ylim_max - ylim_min))
    #         if c > 0:
    #             yticks = ax.get_yticks()
    #             ax.set_yticklabels([""] * len(yticks))

    cond = Line2D([0], [0], marker='s', color=COND_COLOR, linestyle='-', linewidth=1, markersize=7)
    base = Line2D([0], [0], marker='D', color=BASE_COLOR, linestyle='-', linewidth=1, markersize=7)
    orig = Line2D([0], [0], marker='o', color=ORIG_COLOR, linestyle='-', linewidth=1, markersize=7)

    fig.legend(
        [cond, base],
        ["Hierarchical amortized inference", "Non-hierarchical amortized inference"],
        fontsize=10, ncol=2,
        bbox_to_anchor=(0.8, -0.005),
    )

    figure_save(fig, os.path.join(figpath, "ans_parameter_comparison.png"), save_svg=True)



def organize_original_data():
    exp_pkl = pickle_load(os.path.join(DIR_TO_DATA, "data/ijhcs25_data/summary/original/experiment_result_dict.pkl"))
    exp_pkl = exp_pkl["half"]["sub"]

    player_list = list(exp_pkl.keys())
    cols = ["tier", "player", "mode", "session_id", "trial_index", "trial_completion_time", "result", "glancing_distance"]
    player_df = pd.concat([exp_pkl[player][sensi][color][cols] 
                           for player in player_list
                           for sensi in ["custom", "default"]
                           for color in ["white", "gray"]])
    
    simul_df = list()
    cols = ["trial_completion_time", "result", "glancing_distance"]
    for player in player_list:
        data = pickle_load(os.path.join(DIR_TO_DATA, f"data/ijhcs25_data/summary/original/simul_pkl/{player}.pkl"))
        for sensi, color in product(["custom", "default"], ["white", "gray"]):
            simul_df.append(data[sensi][color]["df"][cols])
    simul_df = pd.concat(simul_df)

    ### Merge
    player_df = player_df.rename(columns={"trial_completion_time": "trial_completion_time_user", 
                                          "result": "accuracy_user", 
                                          "glancing_distance": "saccadic_deviation_user",
                                          "player": "user"})
    simul_df = simul_df.rename(columns={"trial_completion_time": "trial_completion_time_simul", 
                                          "result": "accuracy_simul", 
                                          "glancing_distance": "saccadic_deviation_simul"})
    player_df["tier"] = player_df["tier"].map({"PRO": 0, "AMA": 1})
    player_df["user"] = player_df["user"].map({player: player_list.index(player) for player in player_list})
    player_df["col_id"] = player_df["session_id"].map({
        "ssw": 0,
        "ssg": 1,
        "slw": 0,
        "slg": 1,
        "fsw": 0,
        "fsg": 1,
        "flw": 0,
        "flg": 1
    })
    player_df["sen_id"] = player_df["mode"].map({"default": 1, "custom": 0})
    player_df["rad_id"] = player_df["session_id"].map({
        "ssw": 0,
        "ssg": 0,
        "slw": 1,
        "slg": 1,
        "fsw": 0,
        "fsg": 0,
        "flw": 1,
        "flg": 1
    })
    player_df["spd_id"] = player_df["session_id"].map({
        "ssw": 0,
        "ssg": 0,
        "slw": 0,
        "slg": 0,
        "fsw": 1,
        "fsg": 1,
        "flw": 1,
        "flg": 1
    })
    cols_left  = ["tier", "user", "col_id", "sen_id", "rad_id", "spd_id",
              "trial_completion_time_user", "accuracy_user", "saccadic_deviation_user"]
    cols_right = ["trial_completion_time_simul", "accuracy_simul", "saccadic_deviation_simul"]

    merged_df = pd.concat(
        [
            player_df[cols_left].reset_index(drop=True),
            simul_df[cols_right].reset_index(drop=True)
        ],
        axis=1
    )
    merged_df.to_csv(os.path.join(DIR_TO_DATA, "data/ijhcs25_data/summary/original/original_fitting.csv"), index=False)


def visualize_model_fitting():
    cond_df = pd.read_csv(os.path.join(DIR_TO_DATA, f"data/ijhcs25_data/infer_result/{COND}/mf_result.csv"))
    base_df = pd.read_csv(os.path.join(DIR_TO_DATA, f"data/ijhcs25_data/infer_result/{BASE}/mf_result.csv"))
    orig_df = pd.read_csv(os.path.join(DIR_TO_DATA, "data/ijhcs25_data/summary/original/original_fitting.csv"))

    cond_df = cond_df[cond_df["truncated"] == 0.0]
    base_df = base_df[base_df["truncated"] == 0.0]

    dfs = {
        "COND": cond_df,
        "BASE": base_df,
        # "ORIG": orig_df
    }
    plotcolor = {
        "COND": COND_COLOR,
        "BASE": BASE_COLOR,
        "ORIG": ORIG_COLOR
    }
    zorder = {
        "COND": 10,
        "BASE": 50,
        "ORIG": 100
    }

    METRICS = ["trial_completion_time", "accuracy", "saccadic_deviation"]

    cond = Line2D([0], [0], color=COND_COLOR, linestyle='-', linewidth=1)
    base = Line2D([0], [0], color=BASE_COLOR, linestyle='-', linewidth=1)
    orig = Line2D([0], [0], color=ORIG_COLOR, linestyle='-', linewidth=1)

    fig, axs = figure_grid(2, 3, size_ax=np.array([3, 3]) / 1.1)
    for c, mt in enumerate(METRICS):
        # User aggregation
        ax = axs[0][c]
        r2s = list()
        for key in dfs:
            df = dfs[key].copy()
            df = df.groupby("user", as_index=False).mean()
            
            x = df[f"{mt}_user"].to_numpy()
            y = df[f"{mt}_simul"].to_numpy()
            if c == 1:
                x *= 100
                y *= 100
            r2s.append(get_r_squared(x, y))

            ax.scatter(x, y, color=plotcolor[key], s=5, alpha=0.6, zorder=zorder[key], marker='o', linewidth=0.75)
            sns.regplot(
                x=x, y=y,
                scatter_kws={"color": "black", "alpha": 0, "marker": 'o'},
                line_kws={"color": plotcolor[key], "lw": 2, "alpha": 1, "zorder": zorder[key]},
                ax=ax,
                ci=None,
            )

        if c == 0:
            _min, _max = 0.43, 0.67
            ax.set_xlabel("Player TCT (s)")
            ax.set_ylabel("Simulation TCT (s)")
        elif c == 1:
            _min, _max = 45, 92
            ax.set_xlabel("Player ACC (%)")
            ax.set_ylabel("Simulation ACC (%)")
        else:
            _min, _max = 1.5, 8
            ax.set_xlabel(f"Player SCD ($^\circ$)")
            ax.set_ylabel(f"Simulation SCD ($^\circ$)")
        
        ax.plot([_min, _max], [_min, _max], color='gray', linewidth=0.7, linestyle='--', zorder=1)
        ax.set_xlim(_min, _max)
        ax.set_ylim(_min, _max)
        ax.grid(linestyle='--', color='gray', linewidth=0.5)
        ax.set_aspect('equal')

        ax.legend([cond, base], [f"$R^2$={r2s[0]:.2f}", f"$R^2$={r2s[1]:.2f}"], fontsize=7)

        # User x task
        ax = axs[1][c]
        r2s = list()
        for key in dfs:
            df = dfs[key].copy()
            df = df.groupby(["user", "col_id", "sen_id", "spd_id", "rad_id"], as_index=False).mean()

            _r2 = list()
            for user in range(20):
                x = df[df["user"] == user][f"{mt}_user"].to_numpy()
                y = df[df["user"] == user][f"{mt}_simul"].to_numpy()
                if c == 1:
                    x *= 100
                    y *= 100
                _r2.append(get_r_squared(x, y))

                ax.scatter(x, y, color=plotcolor[key], s=5, alpha=0.3, zorder=zorder[key], marker='o', linewidth=0.75)
                sns.regplot(
                    x=x, y=y,
                    scatter_kws={"color": "black", "alpha": 0, "marker": 'o'},
                    line_kws={"color": plotcolor[key], "lw": 0.7, "alpha": 0.4, "zorder": zorder[key]},
                    ax=ax,
                    ci=None,
                )
            r2s.append(np.mean(_r2))

        ax.legend([cond, base], 
                  [f"Avg $R^2$={r2s[0]:.2f}", f"Avg $R^2$={r2s[1]:.2f}"], fontsize=7, loc='lower right')
            
        if c == 0:
            _min, _max = 0.3, 1.0
            ax.set_xlabel("Player TCT (s)")
            ax.set_ylabel("Simulation TCT (s)")
        elif c == 1:
            _min, _max = -1, 101
            ax.set_xlabel("Player ACC (%)")
            ax.set_ylabel("Simulation ACC (%)")
        else:
            _min, _max = 1, 9
            ax.set_xlabel(f"Player SCD ($^\circ$)")
            ax.set_ylabel(f"Simulation SCD ($^\circ$)")
            
        ax.plot([_min, _max], [_min, _max], color='gray', linewidth=0.7, linestyle='--', zorder=1)
        ax.set_xlim(_min, _max)
        ax.set_ylim(_min, _max)
        ax.grid(linestyle='--', color='gray', linewidth=0.5)
        ax.set_aspect('equal')

    cond = Line2D([0], [0], color=COND_COLOR, linestyle='-', linewidth=1)
    base = Line2D([0], [0], color=BASE_COLOR, linestyle='-', linewidth=1)
    orig = Line2D([0], [0], color=ORIG_COLOR, linestyle='-', linewidth=1)

    fig.legend(
        [cond, base],
        ["Hierarchical amortized inference", "Non-hierarchical amortized inference"],
        fontsize=10, ncol=2,
        bbox_to_anchor=(0.8, -0.005),
    )

    figpath = os.path.join(DIR_TO_DATA, f"data/visualization/inference/ans_infer_comparison")
    figure_save(fig, os.path.join(figpath, "ans_model_fitting.png"), save_svg=True, dpi=300)

            

            





if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    visualize_parameter_comparison()
    # organize_original_data()
    visualize_model_fitting()