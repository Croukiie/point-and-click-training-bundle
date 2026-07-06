import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from itertools import product
from einops import rearrange

from ..utils.myplot import figure_grid, figure_save
from ..utils.myutils import pickle_save, pickle_load
from ..utils.mymath import get_r_squared

from ..config.constant import FIELD, PARAM_SYMBOL, NAME_ABBR, NAME_FULL
from .pnc_inferer import PnCInferer
from ..datamanagekit.prolific_data import PnCUserDatasetProlific

from scipy import stats
import seaborn as sns

from matplotlib.lines import Line2D

import warnings
warnings.filterwarnings("ignore", message=".*use_inf_as_na.*")

DIR_TO_DATA = Path(__file__).parent.parent.parent
N_USER = 90

N_CPU = 10


def infer_and_simulate(inferer:PnCInferer, dataset:PnCUserDatasetProlific, **kwargs):
    stat_true, traj_true, task_list, user_stat_list = dataset.load_dataset(**kwargs)
    stat_norm, traj_norm = inferer.process_data(stat_true, traj_true)
    param_z, param_w = inferer.infer(stat_norm, traj_norm, return_z=True, return_w=True, return_dict=True)
    stat_simul, traj_simul = inferer.simulate(param_z=param_z,
                                              task_list=task_list, 
                                              user_stat_list=user_stat_list, 
                                              resimulate_max_num=15,
                                              verbose=False)
    return param_w, stat_true, stat_simul


def infer_batch(inferer:PnCInferer, dataset:PnCUserDatasetProlific, 
                user:int, fmode:int, bin_id: int,
                metrics=["TCT", "MVT", "HIT", "CLICK_DIST", "CLICK_DIST_NORM", "TRAV_LEN", "MAX_CURSOR_SPD", "MOTOR_EFFORT"]):
    """
    fmode := 0, 1, 2, or 3
    0 := 1 parameter set from all dataset (no binning)
    1 := 4 parameter sets, binning by stake (4)
    2 := 12 parameter sets, binning by stake (4) x period (3)
    3 := 24 parameter sets, binning by stake (4) x period (3) x difficulty (2)
    """

    save_dir = os.path.join(DIR_TO_DATA, f"data/prolific_data/infer_result/{inferer.name}")
    os.makedirs(save_dir, exist_ok=True)

    if fmode == 0:
        dataset_filter_cond = {}

    elif fmode == 1:
        assert bin_id in range(4)
        dataset_filter_cond = dict(stake_level=bin_id)

    elif fmode == 2:
        assert bin_id in range(12)
        dataset_filter_cond = dict(stake_level=bin_id//3, period_level=bin_id%3)

    elif fmode == 3:
        assert bin_id in range(24)
        dataset_filter_cond = dict(stake_level=bin_id//6, period_level=(bin_id%6)//2, difficulty_level=(bin_id%6)%2)

    else:
        raise RuntimeError(f"fmode={fmode} out of range.")

    param_w, stat_t, stat_s = infer_and_simulate(inferer, dataset, user=user, **dataset_filter_cond)

    df = stat_t[[
        "user", "trial", "cursor_pos_x0", "cursor_pos_y0", "target_pos_x0", "target_pos_y0",
        "target_rel_pos_x0", "target_rel_pos_y0", "target_radius", "reaction_time",
        "target_index_of_difficulty", "target_p", "elapsed_time",
        "stake_level", "period_level", "difficulty_level", "condition_id"
    ]].copy()
    for mt in metrics:
        df.loc[:,f"{FIELD[mt]}_user"] = stat_t[FIELD[mt]].to_numpy()
        df.loc[:,f"{FIELD[mt]}_simul"] = stat_s[FIELD[mt]].to_numpy()
    
    return {"user": user, **dataset_filter_cond, **param_w}, df


def infer_fitting_mode(inferer:PnCInferer, dataset:PnCUserDatasetProlific, fmode:int, num_cpu=N_CPU):
    def _simul(user, bin_id):
        return infer_batch(inferer, dataset, user, fmode, bin_id)
    
    os.makedirs(os.path.join(DIR_TO_DATA, f"data/prolific_data/infer_result/{inferer.name}"), exist_ok=True)
    
    iterator = list(product(range(N_USER), range([1, 4, 12, 24][fmode])))
    
    result = Parallel(n_jobs=num_cpu)(delayed(_simul)(user, bin_id) for user, bin_id in tqdm(iterator, desc=f"Fitting mode={fmode}: "))
    param_w = pd.DataFrame([res[0] for res in result])
    param_w.to_csv(os.path.join(DIR_TO_DATA, f"data/prolific_data/infer_result/{inferer.name}/fmode_{fmode}_param.csv"), index=False)

    comp = pd.concat([res[1] for res in result])
    comp.to_csv(os.path.join(DIR_TO_DATA, f"data/prolific_data/infer_result/{inferer.name}/fmode_{fmode}_mf.csv"), index=False)



def compute_model_fitting_metric(inferer:PnCInferer, fmode:int, 
                                 metrics=["TCT", "MVT", "HIT", "CLICK_DIST", "CLICK_DIST_NORM", "TRAV_LEN", "MAX_CURSOR_SPD"]):
    
    orig_df = pd.read_csv(os.path.join(DIR_TO_DATA, f"data/prolific_data/infer_result/{inferer.name}/fmode_{fmode}_mf.csv"))
    result = dict()

    # Average metric per user
    result["user_average_metric"] = dict()
    df = orig_df.copy()
    df = df.groupby("user", as_index=False).mean()

    for mt in metrics:
        x_user = df[f"{FIELD[mt]}_user"].to_numpy()
        y_simul = df[f"{FIELD[mt]}_simul"].to_numpy()
        result["user_average_metric"][mt] = {"user": x_user, "simul": y_simul}
    
    # Average mteric per cond
    result["cond_average_metric"] = dict()
    df = orig_df.copy()
    df = df.groupby("condition_id", as_index=False).mean()

    for mt in metrics:
        x_user = df[f"{FIELD[mt]}_user"].to_numpy()
        y_simul = df[f"{FIELD[mt]}_simul"].to_numpy()
        result["cond_average_metric"][mt] = {"user": x_user, "simul": y_simul}

    # Average metric per user x cond
    result["user_cond_average_metric"] = dict()
    df = orig_df.copy()
    df = df.groupby(["user", "stake_level", "period_level", "difficulty_level"], as_index=False).mean()

    for mt in metrics:
        x_user = np.array([df[df["user"] == user][f"{FIELD[mt]}_user"].to_numpy() for user in range(N_USER)])   # shape [90, 24]
        y_simul = np.array([df[df["user"] == user][f"{FIELD[mt]}_simul"].to_numpy() for user in range(N_USER)])
        result["user_cond_average_metric"][mt] = {"user": x_user, "simul": y_simul}

    # R2 with user average
    result["R2_inter_user"] = dict()
    for mt in metrics:
        result["R2_inter_user"][mt] = get_r_squared(result["user_average_metric"][mt]["user"], 
                                                    result["user_average_metric"][mt]["simul"])
    
    # R2 with cond average
    result["R2_inter_cond"] = dict()
    for mt in metrics:
        result["R2_inter_cond"][mt] = get_r_squared(result["cond_average_metric"][mt]["user"], 
                                                    result["cond_average_metric"][mt]["simul"])
    
    # Average R2 of each user, binned by stake x period x diff
    result["R2_intra_user"] = dict()
    for mt in metrics:
        # length = 90
        result["R2_intra_user"][mt] = np.array([get_r_squared(x, y) for x, y in zip(
            result["user_cond_average_metric"][mt]["user"],
            result["user_cond_average_metric"][mt]["simul"]
        )])

    # Average MD of each user
    result["mean_diff"] = dict()
    for mt in metrics:
        result["mean_diff"][mt] = np.abs(result["user_average_metric"][mt]["user"] - result["user_average_metric"][mt]["simul"])

    # Average MAE of each user, binned by stake / stake x period / stake x period x diff
    result["mae"] = dict()
    for mt in metrics:
        result["mae"][mt] = np.abs(result["user_cond_average_metric"][mt]["user"] - result["user_cond_average_metric"][mt]["simul"]).mean(axis=1)

    return result



def visualize_fitting_performance(inferer:PnCInferer, fmode: int):
    figpath = os.path.join(DIR_TO_DATA, f"data/prolific_data/infer_result/{inferer.name}/visualization")
    os.makedirs(figpath, exist_ok=True)

    result = compute_model_fitting_metric(inferer, fmode)

    # 1 Scatterplot, 1 point = 1 user
    metrics = ["MVT", "HIT"]
    xlabels = ["User MT (s)", "User ACC (%)"]
    ylabels = ["Simulator MT (s)", "Simulator ACC (%)"]

    STAKE_LABEL = [0, 0.1, 1, 10]
    PERIOD_LABEL = ["Early", "Middle", "Late"]
    DIFF_LABEL = ["Easy", "Hard"]

    markercolor_stake = ["#FFFFFF", "#ABE7FF", "#57D0FF", "#00AAFF"]

    # Columnwise order - inter-user, inter-condition, user x condition
    fig, axs = figure_grid(len(metrics), 3, size_ax=np.array([3, 3]))
    for (i, mt), col in product(enumerate(metrics), range(3)):
        ax = axs[i][col]

        if i == 0:
            ax.set_title(
                [
                    "User average",
                    "Condition average",
                    "User x Condition"
                ][col]
            )

        if col == 0:
            x = result["user_average_metric"][mt]["user"]
            y = result["user_average_metric"][mt]["simul"]
            r2 = result["R2_inter_user"][mt]
            ax.scatter(x, y, edgecolor='k', s=50, alpha=0.8, zorder=50, linewidth=0.75,
                    color="#353535", marker='o')
            sns.regplot(
                x=x, y=y,
                scatter_kws={"color": "black", "alpha": 0, "marker": 'o'},
                line_kws={"color": 'r', "lw": 2, "alpha": 1, "zorder": 500},
                ax=ax,
                ci=None
            )
        elif col == 1:
            x = result["cond_average_metric"][mt]["user"]
            y = result["cond_average_metric"][mt]["simul"]
            r2 = result["R2_inter_cond"][mt]
            for j, (_x, _y) in enumerate(zip(x, y)):
                ax.scatter(_x, _y, s=50, alpha=0.8, zorder=50, linewidth=0.75,
                        color=markercolor_stake[j//6], marker='^sD'[(j%6)//2], edgecolor='kr'[(j%6)%2])
            sns.regplot(
                x=x, y=y,
                scatter_kws={"color": "black", "alpha": 0, "marker": 'o'},
                line_kws={"color": 'r', "lw": 2, "alpha": 1, "zorder": 500},
                ax=ax,
                ci=None
            )
        elif col == 2:
            xs = result["user_cond_average_metric"][mt]["user"]
            ys = result["user_cond_average_metric"][mt]["simul"]
            r2 = np.mean(result["R2_intra_user"][mt])
            for x, y in zip(xs, ys):
                for j, (_x, _y) in enumerate(zip(x, y)):
                    ax.scatter(_x, _y, s=50, alpha=0.8, zorder=50, linewidth=0.75,
                            color=markercolor_stake[j//6], marker='^sD'[(j%6)//2], edgecolor='kr'[(j%6)%2])
                sns.regplot(
                    x=x, y=y,
                    scatter_kws={"color": "black", "alpha": 0, "marker": 'o'},
                    line_kws={"color": 'r', "lw": 0.5, "alpha": 0.5, "zorder": 500},
                    ax=ax,
                    ci=None
                )


        if mt == "MVT":
            v_min, v_max = 0.0, 2.0
            
        elif mt == "HIT":
            v_min, v_max = -0.02, 1.02
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1], [0, 25, 50, 75, 100])
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], [0, 20, 40, 60, 80, 100])
        
        ax.plot([v_min, v_max], [v_min, v_max], 
                color='gray', linewidth=0.7, linestyle='--', zorder=1)
        
        legend = ax.legend(
            [Line2D([0], [0], color='r', lw=2, alpha=0.6)],
            [f"$R^2={r2:.2f}$"] if col != 2 else [f"$Avg. R^2={r2:.2f}$"],
            fontsize=8, loc='lower right'
        )
        legend.set_zorder(1000)

        ax.set_xlabel(xlabels[i])
        ax.set_ylabel(ylabels[i])
        ax.set_xlim(v_min, v_max)
        ax.set_ylim(v_min, v_max)
        ax.grid(linestyle='--', linewidth=0.5)
        ax.set_aspect('equal')

    # Legend
    rew0 = ax.scatter([], [], marker='o', edgecolor='k', linewidth=1, facecolor="#FFFFFF", s=50, alpha=0.9)
    rew1 = ax.scatter([], [], marker='o', edgecolor='k', linewidth=1, facecolor="#ABE7FF", s=50, alpha=0.9)
    rew2 = ax.scatter([], [], marker='o', edgecolor='k', linewidth=1, facecolor="#57D0FF", s=50, alpha=0.9)
    rew3 = ax.scatter([], [], marker='o', edgecolor='k', linewidth=1, facecolor="#00AAFF", s=50, alpha=0.9)

    per0 = ax.scatter([], [], marker='^', edgecolor='k', linewidth=1, facecolor="k", s=50, alpha=0.9)
    per1 = ax.scatter([], [], marker='s', edgecolor='k', linewidth=1, facecolor="k", s=50, alpha=0.9)
    per2 = ax.scatter([], [], marker='D', edgecolor='k', linewidth=1, facecolor="k", s=50, alpha=0.9)

    dif0 = ax.scatter([], [], marker='o', edgecolor='k', linewidth=1, facecolor="white", s=50, alpha=0.9)
    dif1 = ax.scatter([], [], marker='o', edgecolor='r', linewidth=1, facecolor="white", s=50, alpha=0.9)
    
    fig.legend(
        [rew0, rew1, rew2, rew3],
        STAKE_LABEL,
        fontsize=10, ncol=4,
        bbox_to_anchor=(0.39, -0.01),
        title="Stake (\u00A2)"
    )
    fig.legend(
        [per0, per1, per2],
        PERIOD_LABEL,
        fontsize=10, ncol=3,
        bbox_to_anchor=(0.73, -0.01),
        title="Period"
    )
    fig.legend(
        [dif0, dif1],
        DIFF_LABEL,
        fontsize=10, ncol=2,
        bbox_to_anchor=(0.95, -0.01),
        title="Difficulty"
    )
    fig.suptitle(f"Binning: {['All', 'Stake', 'Stake x Period', 'Stake x Period x Difficulty'][fmode]}", fontsize=14)

    figure_save(fig, os.path.join(figpath, f"model_fitting_visual_{fmode}.png"), save_svg=True)
    



def visualize_parameter_inference_result(inferer:PnCInferer):
    figpath = os.path.join(DIR_TO_DATA, f"data/prolific_data/infer_result/{inferer.name}/visualization")
    os.makedirs(figpath, exist_ok=True)

    param_list = inferer.simulator.total_target_param
    param_df = pd.read_csv(os.path.join(DIR_TO_DATA, f"data/prolific_data/infer_result/{inferer.name}/fmode_3_param.csv"))

    xkeys = ["stake_level", "period_level", "difficulty_level"]
    xlabels = [
        [0, 0.1, 1, 10],
        ["Early", "Middle", "Late"],
        ["Easy", "Hard"]
    ]

    # Fitting mode 3 - Stake only
    fig, axs = figure_grid(len(param_list), 3, size_ax=np.array([4, 2.2]) / 1.5)
    
    for (i, param), j in product(enumerate(param_list), range(3)):
        ax = axs[i][j]

        sns.lineplot(data=param_df, x=xkeys[j], y=param, marker='s', color='k',
                     linewidth=1, markersize=7, 
                     errorbar=('ci', 95), err_style='bars', err_kws={'capsize': 3}, 
                     ax=ax)
        
        ax.set_xlabel("")
        if j == 0:
            ax.set_ylabel(f"Inferred {PARAM_SYMBOL[param]}")
        else:
            ax.set_ylabel("")
        ax.set_xticks(np.arange(len(xlabels[j])), xlabels[j])
        ax.grid(axis='y', alpha=0.5, linestyle='--')

    
    axs[-1][0].set_xlabel("Stake")
    axs[-1][1].set_xlabel("Period")
    axs[-1][2].set_xlabel("Difficulty")
    
    figure_save(fig, os.path.join(figpath, "parameter_descriptive.png"), save_svg=True)


def visualize_parameter_inference_result_with_physical_unit(inferer:PnCInferer, name:str):
    figpath = os.path.join(DIR_TO_DATA, f"data/prolific_data/infer_result/{inferer.name}/visualization")
    os.makedirs(figpath, exist_ok=True)

    # param_list = inferer.simulator.total_target_param
    param_list = ["theta_m", "theta_c", "rew_succ", "rew_fail", "actual_time_cost", "actual_effort_cost"]
    param_df = pd.read_csv(os.path.join(DIR_TO_DATA, f"data/prolific_data/infer_result/{inferer.name}/fmode_3_param.csv"))

    # Processing df
    param_df2 = pd.read_csv(os.path.join(DIR_TO_DATA, f"data/prolific_data/infer_result/{inferer.name}/rew_merged_df.csv"))
    param_df2 = param_df2.groupby(["user", "stake_level", "period_level", "difficulty_level"], as_index=False).agg({
        "actual_time_cost": "mean", "actual_effort_cost": "mean"
    })
    param_df = pd.merge(param_df, param_df2, on=["user", "stake_level", "period_level", "difficulty_level"], how="left")

    xkeys = ["stake_level", "period_level", "difficulty_level"]
    xlabels = [
        [0, 0.1, 1, 10],
        ["Early", "Middle", "Late"],
        ["Easy", "Hard"]
    ]

    # Fitting mode 3 - Stake only
    fig, axs = figure_grid(len(param_list), 3, size_ax=np.array([4, 2.2]) / 1.5)
    
    for (i, param), j in product(enumerate(param_list), range(3)):
        ax = axs[i][j]

        sns.lineplot(data=param_df, x=xkeys[j], y=param, marker='s', color='k',
                     linewidth=1, markersize=7, 
                     errorbar=('ci', 95), err_style='bars', err_kws={'capsize': 3}, 
                     ax=ax)
        
        ax.set_xlabel("")
        if j == 0:
            if param == "actual_time_cost":
                ax.set_ylabel(r"$\hat{w}_{time}\times$ TCT")
            elif param == "actual_effort_cost":
                ax.set_ylabel(r"$\hat{w}_{effort}\times$ Accel.")
            else:
                ax.set_ylabel(f"Inferred {PARAM_SYMBOL[param]}")
        else:
            ax.set_ylabel("")
        ax.set_xticks(np.arange(len(xlabels[j])), xlabels[j])
        ax.grid(axis='y', alpha=0.5, linestyle='--')

    
    axs[-1][0].set_xlabel("Stake")
    axs[-1][1].set_xlabel("Period")
    axs[-1][2].set_xlabel("Difficulty")

    fig.suptitle(name, fontsize=12)
    
    figure_save(fig, os.path.join(figpath, "parameter_descriptive_2.png"), save_svg=True)
    param_df.to_csv(os.path.join(DIR_TO_DATA, f"data/prolific_data/infer_result/{inferer.name}/param_merged.csv"), index=False)

    # RM ANOVA FORMAT
    data = list()
    for user in range(N_USER):
        user_data = dict(user=user)
        for param, s, p, d in product(param_list, range(4), range(3), range(2)):
            v = param_df[
                (param_df["user"] == user) &
                (param_df["stake_level"] == s) &
                (param_df["period_level"] == p) &
                (param_df["difficulty_level"] == d)
            ][param].to_numpy().mean()
            user_data[f"{param}_{s}{p}{d}"] = v
        data.append(user_data)
    
    data = pd.DataFrame(data)
    data.to_csv(os.path.join(DIR_TO_DATA, f"data/prolific_data/infer_result/{inferer.name}/param_merged_df_rm_anova.csv"), index=False)




def construct_rm_anova_format(inferer:PnCInferer):
    param_csv_path = os.path.join(DIR_TO_DATA, f"data/prolific_data/infer_result/{inferer.name}/fmode_3_param.csv")
    df = pd.read_csv(param_csv_path)

    target_param = inferer.simulator.total_target_param

    data = list()
    for user in range(N_USER):
        user_data = dict(user=user)
        for param, s, p, d in product(target_param, range(4), range(3), range(2)):
            v = df[
                (df["user"] == user) &
                (df["stake_level"] == s) &
                (df["period_level"] == p) &
                (df["difficulty_level"] == d)
            ][param].to_numpy()[0]
            user_data[f"{param}_{s}{p}{d}"] = v
        data.append(user_data)
    
    data = pd.DataFrame(data)
    data.to_csv(os.path.join(DIR_TO_DATA, f"data/prolific_data/infer_result/{inferer.name}/fmode_3_param_rm_anova.csv"), index=False)



def compute_df_reward_term_relative_scale(inferer:PnCInferer):
    user_df = pd.read_csv(os.path.join(DIR_TO_DATA, "data/prolific_data/stat/merged_summary.csv"))
    param_df = pd.read_csv(os.path.join(DIR_TO_DATA, f"data/prolific_data/infer_result/{inferer.name}/fmode_3_param.csv"))

    df = user_df.merge(param_df[["user", "stake_level", "period_level", "difficulty_level",
                                "rew_succ", "rew_fail", "time_penalty", "effort_cost"]], 
                       on=["user", "stake_level", "period_level", "difficulty_level"], how='left')
    df = df[[
        "user", "stake_level", "period_level", "difficulty_level", "target_rew",
        "rew_succ", "rew_fail", "time_penalty", "effort_cost",
        "trial_completion_time", "target_hit", "motor_effort"
    ]]
    df["actual_time_cost"] = 1 + df["trial_completion_time"].to_numpy() * np.clip(df["time_penalty"].to_numpy(), -np.inf, np.inf)
    df["actual_effort_cost"] = df["motor_effort"].to_numpy() * df["effort_cost"].to_numpy()
    df["click_score"] = np.where(df["target_hit"] == 1, df["rew_succ"], -df["rew_fail"])
    df["actual_score"] = np.where(df["target_hit"] == 1, df["target_rew"], -df["target_rew"])

    df["rel_succ"] = df["rew_succ"] / df["actual_time_cost"]
    df["rel_fail"] = df["rew_fail"] / df["actual_time_cost"]
    df["rel_eff"] = df["actual_effort_cost"] / df["actual_time_cost"]

    df.to_csv(os.path.join(DIR_TO_DATA, f"data/prolific_data/infer_result/{inferer.name}/rew_merged_df.csv"))

    # RM ANOVA FORMAT
    data = list()
    target_param = [
        "rel_succ",
        "rel_fail",
        "rel_eff"
    ]
    for user in range(N_USER):
        user_data = dict(user=user)
        for param, s, p, d in product(target_param, range(4), range(3), range(2)):
            v = df[
                (df["user"] == user) &
                (df["stake_level"] == s) &
                (df["period_level"] == p) &
                (df["difficulty_level"] == d)
            ][param].to_numpy().mean()
            user_data[f"{param}_{s}{p}{d}"] = v
        data.append(user_data)
    
    data = pd.DataFrame(data)
    data.to_csv(os.path.join(DIR_TO_DATA, f"data/prolific_data/infer_result/{inferer.name}/rew_merged_df_rm_anova.csv"), index=False)




def visualize_relative_scales(inferer:PnCInferer):
    figpath = os.path.join(DIR_TO_DATA, f"data/prolific_data/infer_result/{inferer.name}/visualization")

    df = pd.read_csv(os.path.join(DIR_TO_DATA, f"data/prolific_data/infer_result/{inferer.name}/rew_merged_df.csv"))

    STAKE_LABEL = [0, 0.1, 1, 10]
    PERIOD_LABEL = ["Early", "Middle", "Late"]
    DIFF_LABEL = ["Easy", "Hard"]
    LABELS = [STAKE_LABEL, PERIOD_LABEL, DIFF_LABEL]

    # Relative scale
    param_list = ["rel_succ", "rel_fail", "rel_eff"]
    ylabels = ["Earn per time", "Loss per time", "Effort per time", "Actual time cost"]

    fig, axs = figure_grid(len(param_list), 3, size_ax=np.array([4, 2.2]) / 1.5)
    for (i, param), (j, key) in product(enumerate(param_list), 
                                        enumerate(["stake_level", "period_level", "difficulty_level"])):
        ax = axs[i][j]

        sns.lineplot(data=df, x=key, y=param, marker='s', color='k',
                     linewidth=1, markersize=7, 
                     errorbar=('ci', 95), err_style='bars', err_kws={'capsize': 3}, 
                     ax=ax)
        
        ax.set_xlabel("")
        if j == 0:
            ax.set_ylabel(ylabels[i])
        else:
            ax.set_ylabel("")
        ax.set_xticks(np.arange(len(LABELS[j])), LABELS[j])
        ax.grid(axis='y', alpha=0.5, linestyle='--')
    
    axs[-1][0].set_xlabel("Stake")
    axs[-1][1].set_xlabel("Period")
    axs[-1][2].set_xlabel("Difficulty")
    
    figure_save(fig, os.path.join(figpath, "reward_relative_scale.png"), save_svg=True)



def visualize_incentive_reward_correlation(inferer:PnCInferer):
    figpath = os.path.join(DIR_TO_DATA, f"data/prolific_data/infer_result/{inferer.name}/visualization")

    df = pd.read_csv(os.path.join(DIR_TO_DATA, f"data/prolific_data/infer_result/{inferer.name}/rew_merged_df.csv"))
    df = df.groupby("user", as_index=False).sum()

    x_money = df["actual_score"].to_numpy()
    y_reward = df["click_score"].to_numpy()
    r2 = get_r_squared(x_money, y_reward)

    fig, ax = figure_grid(1, 1, size_ax=np.array([4, 4]))

    ax.scatter(x_money, y_reward, edgecolor='k', s=50, alpha=0.8, zorder=50, linewidth=0.75,
                    color="#353535", marker='o')
    sns.regplot(
        x=x_money, y=y_reward,
        scatter_kws={"color": "black", "alpha": 0, "marker": 'o'},
        line_kws={"color": 'r', "lw": 2, "alpha": 1, "zorder": 500},
        ax=ax,
        ci=None
    )

    legend = ax.legend(
        [Line2D([0], [0], color='r', lw=2, alpha=0.6)],
        [f"$R^2={r2:.2f}$"],
        fontsize=8, loc='lower right'
    )
    legend.set_zorder(1000)

    ax.set_xlabel("Actual earning")
    ax.set_ylabel("Agent reward")
    
    figure_save(fig, os.path.join(figpath, "reward_correlation.png"), save_svg=True)


def visualize_posterior(inferer:PnCInferer):
    from scipy.stats import gaussian_kde
    # posterior = list()

    # for user, task in product(tqdm(range(N_USER)), range(24)):
    #     stat_true, traj_true, task_list, user_stat_list = dataset.load_dataset(user=user, condition_id=task)
    #     stat_norm, traj_norm = inferer.process_data(stat_true, traj_true)
    #     param_z, param_w, post_sampled = inferer.infer(stat_norm, traj_norm, return_z=True, return_w=True, return_dict=True, return_samples=True)
    #     posterior.append(post_sampled)
    
    # posterior = np.array(posterior)
    # np.save(os.path.join(DIR_TO_DATA, f"data/prolific_data/infer_result/{inferer.name}/param_post.npy"), posterior)

    posterior = np.load(os.path.join(DIR_TO_DATA, f"data/prolific_data/infer_result/{inferer.name}/param_post.npy"))
    posterior = rearrange(posterior, "(u t) n d -> t (u n) d", t=24)

    figpath = os.path.join(DIR_TO_DATA, f"data/prolific_data/infer_result/{inferer.name}/visualization")

    t = 24
    t_idx = np.arange(t)

    group_labels = np.array([i % 3 for i in range(t // 2)] * 2)
    groupings = {
        0: np.array_split(t_idx, 4),
        1: [t_idx[group_labels == i] for i in range(3)],
        2: [t_idx[t_idx % 2 == 0], t_idx[t_idx % 2 == 1]],
    }
    legend_labels = [
        ["0", "0.1", "1", "10"],
        ["Early", "Middle", "Late"],
        ["Easy", "Hard"]
    ]
    
    param_list = ["theta_m", "theta_c", "rew_succ", "rew_fail", "time_penalty", "effort_cost"]
    fig, axs = figure_grid(3, 6, size_ax=np.array([4, 2.5]) / 1.5)
    for (i, param), j in product(enumerate(param_list), range(3)):
        ax = axs[j][i]
        for k, group in enumerate(groupings[j]):
            data = posterior[group, :, i].reshape(-1)
            kde = gaussian_kde(data)
            x_eval = np.linspace(data.min(), data.max(), 200)
            y_eval = kde(x_eval)
            line = sns.lineplot(x=x_eval, y=y_eval, ax=ax, label=legend_labels[j][k])

            mean_val = np.mean(data)
            ax.axvline(mean_val, color=line.get_lines()[-1].get_color(), linestyle='--', linewidth=1)

            ax.set_yticks([])
            ax.tick_params(axis='y', length=0)

        if i == 0:
            if j == 2: ax.set_xlabel("Motor noise")
            ax.set_xlim(0, 0.5)
        elif i == 1:
            if j == 2: ax.set_xlabel("Clock noise")
            ax.set_xlim(0, 0.3)
        elif i == 2:
            if j == 2: ax.set_xlabel("Success reward")
            ax.set_xlim(0, 16)
        elif i == 3:
            if j == 2: ax.set_xlabel("Fail penalty")
            ax.set_xlim(0, 16)
        elif i == 4:
            if j == 2: ax.set_xlabel("Time cost weight")
            ax.set_xlim(0, 32)
        elif i == 5:
            if j == 2: ax.set_xlabel("Effort cost weight")
            ax.set_xlim(0, 32)
        
        if i == 5:
            ax.legend(fontsize=8)

        if j != 2:
            ax.set_xticklabels([])

    figure_save(fig, os.path.join(figpath, "post_param.png"), save_svg=True)



if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    amort_name, ckpt, name = "BaseAmort_DEN_07251102", 100, "Baseline Amortizer (Density Estimation)"
    # amort_name, ckpt, name = "BaseAmort_PTE_07250658", 100, "Baseline Amortizer (Point Estimation)"

    inferer = PnCInferer(amort_name, ckpt)
    dataset = PnCUserDatasetProlific()

    visualize_posterior(inferer)

    # infer_fitting_mode(inferer, dataset, 3, num_cpu=10)

    # visualize_parameter_inference_result(inferer)
    # visualize_fitting_performance(inferer, 3)
    # construct_rm_anova_format(inferer)
    
    # compute_df_reward_term_relative_scale(inferer)
    # visualize_relative_scales(inferer)
    # visualize_incentive_reward_correlation(inferer)
    # visualize_parameter_inference_result_with_physical_unit(inferer, name)

    # result = compute_model_fitting_metric(inferer, 3)
    # mt = 'MVT'
    # print(np.mean(np.abs(result["user_average_metric"][mt]["user"] - result["user_average_metric"][mt]["simul"])))
    # print(np.mean(np.abs(result["user_cond_average_metric"][mt]["user"] - result["user_cond_average_metric"][mt]["simul"])))

    # mt = 'HIT'
    # print(np.mean(np.abs(result["user_average_metric"][mt]["user"] - result["user_average_metric"][mt]["simul"])))
    # print(np.mean(np.abs(result["user_cond_average_metric"][mt]["user"] - result["user_cond_average_metric"][mt]["simul"])))






# def visualize_fitting_performance_comparison(inferer:PnCInferer):
#     figpath = os.path.join(DIR_TO_DATA, f"data/prolific_data/infer_result/{inferer.name}/visualization")
#     os.makedirs(figpath, exist_ok=True)

#     result = [compute_model_fitting_metric(inferer, fmode) for fmode in range(4)]

#     ### User inter R2 comparison
#     metrics = ['MVT', "HIT"]

#     fig, axs = figure_grid(1, len(metrics), size_ax=np.array([4, 3]))
#     for i, mt in enumerate(metrics):
#         ax = axs[i]

#         x = np.arange(4)
#         y = np.array([res["R2_inter_user"][mt] for res in result])
#         ax.plot(x, y, marker='o')

#         ax.set_ylabel(f"R2_USER ({NAME_FULL[mt]})")
#         ax.set_xlabel("Fitting mode")
#         ax.set_xticks(x, ["All", "Stake", "SxP", "SxPxD"])

#     figure_save(fig, os.path.join(figpath, "user_inter_r2.png"))


#     ### User intra R2 comparison
#     metrics = ['MVT', "HIT"]

#     fig, axs = figure_grid(1, len(metrics), size_ax=np.array([4, 3]))
#     for i, mt in enumerate(metrics):
#         ax = axs[i]

#         x = np.arange(4)
#         y = np.array([np.mean(res["R2_intra_user"][mt]) for res in result])
#         se = np.array([stats.sem(res["R2_intra_user"][mt]) for res in result])
#         ax.errorbar(x, y, yerr=se, fmt='o-', capsize=5)

#         ax.set_ylabel(f"R2_COND ({NAME_FULL[mt]})")
#         ax.set_xlabel("Fitting mode")
#         ax.set_xticks(x, ["All", "Stake", "SxP", "SxPxD"])

#     figure_save(fig, os.path.join(figpath, "user_intra_r2.png"))


#     ### MD
#     metrics = ['MVT', "HIT"]

#     fig, axs = figure_grid(1, len(metrics), size_ax=np.array([4, 3]))
#     for i, mt in enumerate(metrics):
#         ax = axs[i]

#         x = np.arange(4)
#         y = np.array([np.mean(res["mean_diff"][mt]) for res in result])
#         se = np.array([stats.sem(res["mean_diff"][mt]) for res in result])
#         ax.errorbar(x, y, yerr=se, fmt='o-', capsize=5)

#         ax.set_ylabel(f"Mean diff. ({NAME_FULL[mt]})")
#         ax.set_xlabel("Fitting mode")
#         ax.set_xticks(x, ["All", "Stake", "SxP", "SxPxD"])

#     figure_save(fig, os.path.join(figpath, "mean_diff.png"))


#     ### MAE
#     metrics = ['MVT', "HIT"]

#     fig, axs = figure_grid(1, len(metrics), size_ax=np.array([4, 3]))
#     for i, mt in enumerate(metrics):
#         ax = axs[i]

#         x = np.arange(4)
#         y = np.array([np.mean(res["mae"][mt]) for res in result])
#         se = np.array([stats.sem(res["mae"][mt]) for res in result])
#         ax.errorbar(x, y, yerr=se, fmt='o-', capsize=5)

#         ax.set_ylabel(f"MAE ({NAME_FULL[mt]})")
#         ax.set_xlabel("Fitting mode")
#         ax.set_xticks(x, ["All", "Stake", "SxP", "SxPxD"])

#     figure_save(fig, os.path.join(figpath, "mae.png"))
