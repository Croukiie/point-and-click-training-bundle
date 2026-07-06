import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from itertools import product

from ..utils.myplot import figure_grid, figure_save
from ..utils.myutils import pickle_save, pickle_load
from ..utils.mymath import get_r_squared

from ..config.constant import FIELD, PARAM_SYMBOL, NAME_ABBR, NAME_FULL
from .pnc_inferer_3step import PnCTwoLevelConditionalInferer
from ..datamanagekit.prolific_data import PnCUserDatasetProlific

from scipy import stats
import seaborn as sns

from matplotlib.lines import Line2D

DIR_TO_DATA = Path(__file__).parent.parent.parent
N_USER = 90

N_CPU = 10


def infer_and_simulate(inferer:PnCTwoLevelConditionalInferer, dataset:PnCUserDatasetProlific, user:int):
    stat_true, traj_true, task_list, user_stat_list = dataset.load_dataset(user=user)
    stat_norm, traj_norm, cond_i1, cond_i2 = inferer.process_data(stat_true, traj_true)
    (
        param_z_list,
        param_w_list,
        cog_param_w,
        rew_param_w1,
        rew_param_w2
    ) = inferer.infer(stat_norm, traj_norm, cond_i1, cond_i2, return_df=True)
    stat_simul, traj_simul = inferer.simulate(total_param_z_list=param_z_list,
                                              task_list=task_list,
                                              user_stat_list=user_stat_list,
                                              resimulate_max_num=15,
                                              verbose=False)
    return cog_param_w, rew_param_w1, rew_param_w2, stat_true, stat_simul


def infer_batch(inferer:PnCTwoLevelConditionalInferer,
                dataset:PnCUserDatasetProlific, 
                user:int,
                metrics=["TCT", "MVT", "HIT", "CLICK_DIST", "CLICK_DIST_NORM", "TRAV_LEN", "MAX_CURSOR_SPD"]):

    save_dir = os.path.join(DIR_TO_DATA, f"data/prolific_data/infer_result/{inferer.name}")
    os.makedirs(save_dir, exist_ok=True)

    cog_param, rew1_param, rew2_param, stat_t, stat_s = infer_and_simulate(inferer, dataset, user)

    df = stat_t[[
        "user", "trial", "cursor_pos_x0", "cursor_pos_y0", "target_pos_x0", "target_pos_y0",
        "target_rel_pos_x0", "target_rel_pos_y0", "target_radius", "reaction_time",
        "target_index_of_difficulty", "target_p", "elapsed_time",
        "stake_level", "period_level", "difficulty_level", "condition_id"
    ]].copy()
    for mt in metrics:
        df.loc[:,f"{FIELD[mt]}_user"] = stat_t[FIELD[mt]].to_numpy()
        df.loc[:,f"{FIELD[mt]}_simul"] = stat_s[FIELD[mt]].to_numpy()
    
    cog_df = pd.DataFrame([{"user": user, **cog_param}])
    rew1_df = pd.DataFrame([{"user": user, "period_level": i, **rew1_param[i]} for i in rew1_param.keys()])
    rew2_df = pd.DataFrame([{"user": user, 
                            "stake_level": i//6, "period_level": (i%6)//2, "difficulty_level": (i%6)%2, 
                            **rew2_param[i]} for i in rew2_param.keys()])
    
    return cog_df, rew1_df, rew2_df, df


def infer_and_fitting(inferer:PnCTwoLevelConditionalInferer, dataset:PnCUserDatasetProlific):
    def _simul(user):
        return infer_batch(inferer, dataset, user)
    
    os.makedirs(os.path.join(DIR_TO_DATA, f"data/prolific_data/infer_result/{inferer.name}"), exist_ok=True)
    
    result = Parallel(n_jobs=N_CPU)(delayed(_simul)(user) for user in tqdm(range(N_USER), desc=f"Fitting: "))

    cog_df = pd.concat([res[0] for res in result])
    cog_df.to_csv(os.path.join(DIR_TO_DATA, f"data/prolific_data/infer_result/{inferer.name}/cog_param.csv"), index=False)

    rew1_df = pd.concat([res[1] for res in result])
    rew1_df.to_csv(os.path.join(DIR_TO_DATA, f"data/prolific_data/infer_result/{inferer.name}/rew1_param.csv"), index=False)

    rew2_df = pd.concat([res[2] for res in result])
    rew2_df.to_csv(os.path.join(DIR_TO_DATA, f"data/prolific_data/infer_result/{inferer.name}/rew2_param.csv"), index=False)

    comp = pd.concat([res[3] for res in result])
    comp.to_csv(os.path.join(DIR_TO_DATA, f"data/prolific_data/infer_result/{inferer.name}/mf_result.csv"), index=False)



def compute_model_fitting_metric(inferer:PnCTwoLevelConditionalInferer, 
                                 metrics=["TCT", "MVT", "HIT", "CLICK_DIST", "CLICK_DIST_NORM", "TRAV_LEN", "MAX_CURSOR_SPD"]):
    
    orig_df = pd.read_csv(os.path.join(DIR_TO_DATA, f"data/prolific_data/infer_result/{inferer.name}/mf_result.csv"))
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



def visualize_fitting_performance(inferer:PnCTwoLevelConditionalInferer):
    figpath = os.path.join(DIR_TO_DATA, f"data/prolific_data/infer_result/{inferer.name}/visualization")
    os.makedirs(figpath, exist_ok=True)

    result = compute_model_fitting_metric(inferer)

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
            v_min, v_max = 0.0, 1.5
            
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

    figure_save(fig, os.path.join(figpath, f"model_fitting_visual.png"))

    

def visualize_parameter_inference_result(inferer:PnCTwoLevelConditionalInferer):
    figpath = os.path.join(DIR_TO_DATA, f"data/prolific_data/infer_result/{inferer.name}/visualization")
    os.makedirs(figpath, exist_ok=True)

    param_list = inferer.simulator.total_target_param

    STAKE_LABEL = [0, 0.1, 1, 10]
    PERIOD_LABEL = ["Early", "Middle", "Late"]
    DIFF_LABEL = ["Easy", "Hard"]


    fig, axs = figure_grid(len(param_list), 3, size_ax=np.array([4, 2.2]) / 1.5)

    cog_df = pd.read_csv(os.path.join(DIR_TO_DATA, f"data/prolific_data/infer_result/{inferer.name}/cog_param.csv"))
    rew1_df = pd.read_csv(os.path.join(DIR_TO_DATA, f"data/prolific_data/infer_result/{inferer.name}/rew1_param.csv"))
    rew2_df = pd.read_csv(os.path.join(DIR_TO_DATA, f"data/prolific_data/infer_result/{inferer.name}/rew2_param.csv"))

    param_df = pd.merge(rew2_df, cog_df[["user", *inferer.simulator.main_param_list]], on=["user"], how='left')
    param_df = pd.merge(param_df, rew1_df[["user", "period_level", *inferer.simulator.sub1_param_list]], on=["user", "period_level"], how='left')

    for i, param in enumerate(param_list):
        ax = axs[i][0]

        sns.lineplot(data=param_df, x="stake_level", y=param, marker='s', color='k',
                     linewidth=1, markersize=7, 
                     errorbar=('ci', 95), err_style='bars', err_kws={'capsize': 3}, 
                     ax=ax)
        
        ax.set_xlabel("")
        ax.set_ylabel(f"Inferred {PARAM_SYMBOL[param]}")
        ax.set_xticks(np.arange(4), STAKE_LABEL)
        ax.grid(axis='y', alpha=0.5, linestyle='--')

        ax = axs[i][1]

        sns.lineplot(data=param_df, x="period_level", y=param, marker='s', color='k',
                     linewidth=1, markersize=7, 
                     errorbar=('ci', 95), err_style='bars', err_kws={'capsize': 3}, 
                     ax=ax)
        
        ax.set_xlabel("")
        ax.set_ylabel(f"")
        ax.set_xticks(np.arange(3), PERIOD_LABEL)
        ax.grid(axis='y', alpha=0.5, linestyle='--')

        ax = axs[i][2]

        sns.lineplot(data=param_df, x="difficulty_level", y=param, marker='s', color='k',
                     linewidth=1, markersize=7, 
                     errorbar=('ci', 95), err_style='bars', err_kws={'capsize': 3}, 
                     ax=ax)
        
        ax.set_xlabel("")
        ax.set_ylabel(f"")
        ax.set_xticks(np.arange(2), DIFF_LABEL)
        ax.grid(axis='y', alpha=0.5, linestyle='--')
    
    axs[-1][0].set_xlabel("Stake")
    axs[-1][1].set_xlabel("Period")
    axs[-1][2].set_xlabel("Difficulty")
    
    figure_save(fig, os.path.join(figpath, "reward_parameter_descriptive.png"))



if __name__ == "__main__":
    amort_name = "testrun_3step"
    inferer = PnCTwoLevelConditionalInferer(amort_name, 6)
    dataset = PnCUserDatasetProlific()

    # infer_and_fitting(inferer, dataset)
    visualize_parameter_inference_result(inferer)
    # visualize_fitting_performance(inferer)