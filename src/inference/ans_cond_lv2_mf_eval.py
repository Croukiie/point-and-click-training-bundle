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

from ..config_ans.constant import PARAM_SYMBOL
from .ans_inferer_3step import AnSTwoLevelConditionalInferer
from ..datamanagekit.ijhcs25_data import ExperimentIJHCS

from scipy import stats
import seaborn as sns

from matplotlib.lines import Line2D

import warnings
warnings.filterwarnings("ignore", message=".*use_inf_as_na.*")

DIR_TO_DATA = Path(__file__).parent.parent.parent
N_USER = 20
N_COND = 16

N_CPU = 10

METRICS = ["trial_completion_time", "accuracy", "saccadic_deviation"]


def infer_and_simulate(inferer:AnSTwoLevelConditionalInferer, dataset:ExperimentIJHCS, user:int):
    stat_true, _ = dataset.load_dataset(user=user)
    task_list, user_stat_list = inferer.extract_task_and_user_stat(stat_true)
    stat_norm, _, cond1, cond2 = inferer.process_data(stat_true)
    (
        param_z_list,
        param_w_list,
        cog_param_w1,
        cog_param_w2,
        rew_param_w
    ) = inferer.infer(stat_norm, None, cond1, cond2, return_df=True)
    stat_simul, _ = inferer.simulate(
        total_param_z_list=param_z_list,
        task_list=task_list,
        user_stat_list=user_stat_list,
        resimulate_max_num=9,
        verbose=False
    )

    return cog_param_w1, cog_param_w2, rew_param_w, stat_true, stat_simul


def infer_batch(inferer:AnSTwoLevelConditionalInferer,
                dataset:ExperimentIJHCS, 
                user:int,
                metrics=METRICS):

    save_dir = os.path.join(DIR_TO_DATA, f"data/ijhcs25_data/infer_result/{inferer.name}")
    os.makedirs(save_dir, exist_ok=True)

    cog_param_w1, cog_param_w2, rew_param_w, stat_t, stat_s = infer_and_simulate(inferer, dataset, user)

    df = stat_t[[
        "user",
        "col_id", "sen_id", "spd_id", "rad_id", "condition1_id", "condition2_id"
    ]].copy()
    for mt in metrics:
        df.loc[:,f"{mt}_user"] = stat_t[mt].to_numpy()
        df.loc[:,f"{mt}_simul"] = stat_s[mt].to_numpy()
    df.loc[:,"truncated"] = stat_s['truncated'].to_numpy()
    
    cog_df = pd.DataFrame([{"tier": user//10, "user": user, **cog_param_w1}])
    rew1_df = pd.DataFrame([{"tier": user//10, "user": user, "col_level": i, 
                             **cog_param_w2[i]} for i in cog_param_w2.keys()])
    rew2_df = pd.DataFrame([{"tier": user//10, "user": user,
                             **dict(zip(
                                ["col_level", "sen_level", "spd_level", "rad_level"],
                                [(i >> j) & 1 for j in range(4)][::-1]
                            )),
                            **rew_param_w[i]} for i in rew_param_w.keys()])
    
    return cog_df, rew1_df, rew2_df, df


def infer_and_fitting(inferer:AnSTwoLevelConditionalInferer, dataset:ExperimentIJHCS, num_cpu=N_CPU):
    def _simul(user):
        return infer_batch(inferer, dataset, user)
    
    os.makedirs(os.path.join(DIR_TO_DATA, f"data/ijhcs25_data/infer_result/{inferer.name}"), exist_ok=True)
    
    result = Parallel(n_jobs=num_cpu)(delayed(_simul)(user) for user in tqdm(range(N_USER), desc=f"Model fitting: "))

    inv_df = pd.concat([res[0] for res in result])
    inv_df.to_csv(os.path.join(DIR_TO_DATA, f"data/ijhcs25_data/infer_result/{inferer.name}/inv_param.csv"), index=False)

    var1_df = pd.concat([res[1] for res in result])
    var1_df.to_csv(os.path.join(DIR_TO_DATA, f"data/ijhcs25_data/infer_result/{inferer.name}/var1_param.csv"), index=False)

    var2_df = pd.concat([res[2] for res in result])
    var2_df.to_csv(os.path.join(DIR_TO_DATA, f"data/ijhcs25_data/infer_result/{inferer.name}/var2_param.csv"), index=False)

    comp = pd.concat([res[3] for res in result])
    comp.to_csv(os.path.join(DIR_TO_DATA, f"data/ijhcs25_data/infer_result/{inferer.name}/mf_result.csv"), index=False)



def compute_model_fitting_metric(inferer:AnSTwoLevelConditionalInferer, metrics=METRICS):
    
    orig_df = pd.read_csv(os.path.join(DIR_TO_DATA, f"data/ijhcs25_data/infer_result/{inferer.name}/mf_result.csv"))
    print(orig_df.shape)
    orig_df = orig_df[orig_df["truncated"] == 0.0]
    print(orig_df.shape)
    result = dict()

    # Average metric per user
    result["user_average_metric"] = dict()
    df = orig_df.copy()
    df = df.groupby("user", as_index=False).mean()

    for mt in metrics:
        x_user = df[f"{mt}_user"].to_numpy()
        y_simul = df[f"{mt}_simul"].to_numpy()
        result["user_average_metric"][mt] = {"user": x_user, "simul": y_simul}
    
    # Average mteric per cond
    result["cond_average_metric"] = dict()
    df = orig_df.copy()
    df = df.groupby("condition2_id", as_index=False).mean()

    for mt in metrics:
        x_user = df[f"{mt}_user"].to_numpy()
        y_simul = df[f"{mt}_simul"].to_numpy()
        result["cond_average_metric"][mt] = {"user": x_user, "simul": y_simul}

    # Average metric per user x cond
    result["user_cond_average_metric"] = dict()
    df = orig_df.copy()
    df = df.groupby(["user", "col_id", "sen_id", 
                     "spd_id", "rad_id"], as_index=False).mean()

    for mt in metrics:
        x_user = np.array([df[df["user"] == user][f"{mt}_user"].to_numpy() for user in range(N_USER)])   # shape [90, 24]
        y_simul = np.array([df[df["user"] == user][f"{mt}_simul"].to_numpy() for user in range(N_USER)])
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



def visualize_fitting_performance(inferer:AnSTwoLevelConditionalInferer):
    figpath = os.path.join(DIR_TO_DATA, f"data/ijhcs25_data/infer_result/{inferer.name}/visualization")
    os.makedirs(figpath, exist_ok=True)

    result = compute_model_fitting_metric(inferer)

    # 1 Scatterplot, 1 point = 1 user
    metrics = METRICS
    xlabels = ["User TCT (s)", "User ACC (%)", f"User SCD ($^\circ$)"]
    ylabels = ["Simulator TCT (s)", "Simulator ACC (%)", f"Simulator SCD ($^\circ$)"]

    # Columnwise order - inter-user, inter-condition, user x condition
    fig, axs = figure_grid(2, len(metrics), size_ax=np.array([3, 3]))
    for (i, mt), row in product(enumerate(metrics), range(2)):
        ax = axs[row][i]

        # if i == 0:
        #     ax.set_title(
        #         [
        #             "User average",
        #             "Condition average",
        #             "User x Condition"
        #         ][col]
        #     )

        if row == 0:
            x = result["user_average_metric"][mt]["user"]
            y = result["user_average_metric"][mt]["simul"]
            r2 = result["R2_inter_user"][mt]
            ax.scatter(x[:10], y[:10], edgecolor='royalblue', s=60, alpha=0.9, zorder=50, 
                       linewidth=1.5, color="#FFFFFF", marker='s')
            ax.scatter(x[10:], y[10:], edgecolor='lightcoral', s=60, alpha=0.9, zorder=50, 
                       linewidth=1.5, color="#FFFFFF", marker='s')
            sns.regplot(
                x=x, y=y,
                scatter_kws={"color": "black", "alpha": 0, "marker": 'o'},
                line_kws={"color": 'r', "lw": 2, "alpha": 1, "zorder": 500},
                ax=ax,
                ci=None
            )
        # elif row == 1:
        #     x = result["cond_average_metric"][mt]["user"]
        #     y = result["cond_average_metric"][mt]["simul"]
        #     r2 = result["R2_inter_cond"][mt]
        #     for j, (_x, _y) in enumerate(zip(x, y)):
        #         ax.scatter(_x, _y, alpha=0.8, zorder=50, linewidth=1.0,
        #                    s=[40, 85][j % 2],
        #                    color=['white', 'gray'][j // 8],
        #                    marker='oD'[(j%4)//2], 
        #                    linestyle=['-', (0, (2, 1))][(j%8)//4],
        #                    edgecolor='k')
        #     sns.regplot(
        #         x=x, y=y,
        #         scatter_kws={"color": "black", "alpha": 0, "marker": 'o'},
        #         line_kws={"color": 'r', "lw": 2, "alpha": 1, "zorder": 500},
        #         ax=ax,
        #         ci=None
        #     )
        elif row == 1:
            xs = result["user_cond_average_metric"][mt]["user"]
            ys = result["user_cond_average_metric"][mt]["simul"]
            r2 = np.mean(result["R2_intra_user"][mt])
            for ju, (x, y) in enumerate(zip(xs, ys)):
                for j, (_x, _y) in enumerate(zip(x, y)):
                    ax.scatter(_x, _y, alpha=0.8, zorder=50, linewidth=0.9,
                                s=[40, 85][j % 2],
                                color=['white', 'gray'][j // 8],
                                marker='oD'[(j%4)//2], 
                                linestyle=['-', (0, (2, 1))][(j%8)//4],
                                edgecolor=['royalblue', 'lightcoral'][ju // 10])
                sns.regplot(
                    x=x, y=y,
                    scatter_kws={"color": "black", "alpha": 0, "marker": 'o'},
                    line_kws={"color": 'r', "lw": 0.5, "alpha": 0.5, "zorder": 500},
                    ax=ax,
                    ci=None
                )


        if mt == "trial_completion_time":
            if row == 0:
                v_min, v_max = 0.4, 0.7
            # elif row == 1:
            #     v_min, v_max = 0.4, 0.75
            else:
                v_min, v_max = 0.3, 1.2
            
        elif mt == "accuracy":
            if row == 0:
                v_min, v_max = 0.5, 0.95
                ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9], [50, 60, 70, 80, 90])
                ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9], [50, 60, 70, 80, 90])
            # elif row == 1:
            #     v_min, v_max = 0.4, 1.02
            #     ax.set_xticks([0.4, 0.7, 1.0], [40, 70, 100])
            #     ax.set_yticks([0.4, 0.6, 0.8, 1.0], [40, 60, 80, 100])
            else:
                v_min, v_max = -0.02, 1.02
                ax.set_xticks([0, 0.25, 0.5, 0.75, 1], [0, 25, 50, 75, 100])
                ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], [0, 20, 40, 60, 80, 100])

        elif mt == "saccadic_deviation":
            if row == 0:
                v_min, v_max = 1.0, 7.5
            # elif row == 1:
            #     v_min, v_max = 3.5, 6.0
            else:
                v_min, v_max = 0.0, 9.0
        
        ax.plot([v_min, v_max], [v_min, v_max], 
                color='gray', linewidth=0.7, linestyle='--', zorder=1)
        
        legend = ax.legend(
            [Line2D([0], [0], color='r', lw=2, alpha=0.6)],
            [f"$R^2={r2:.2f}$"] if row != 1 else [f"$Avg. R^2={r2:.2f}$"],
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
    pro = ax.scatter([], [], marker='s', edgecolor='royalblue', linewidth=1.5, facecolor="#FFFFFF", s=50, alpha=0.9)
    ama = ax.scatter([], [], marker='s', edgecolor='lightcoral', linewidth=1.5, facecolor="#FFFFFF", s=50, alpha=0.9)

    rad_small = ax.scatter([], [], marker='s', edgecolor='k', linewidth=1, facecolor="#FFFFFF", s=40, alpha=0.9)
    rad_large = ax.scatter([], [], marker='s', edgecolor='k', linewidth=1, facecolor="#FFFFFF", s=85, alpha=0.9)

    sen_hab = ax.scatter([], [], marker='s', edgecolor='k', linewidth=1, facecolor="#FFFFFF", s=50, alpha=0.9, linestyle='-')
    sen_def = ax.scatter([], [], marker='s', edgecolor='k', linewidth=1, facecolor="#FFFFFF", s=50, alpha=0.9, linestyle=(0, (2, 1)))

    spd_stt = ax.scatter([], [], marker='o', edgecolor='k', linewidth=1, facecolor="#FFFFFF", s=50, alpha=0.9)
    spd_mov = ax.scatter([], [], marker='D', edgecolor='k', linewidth=1, facecolor="#FFFFFF", s=50, alpha=0.9)

    col_w = ax.scatter([], [], marker='s', edgecolor='k', linewidth=1, facecolor="#FFFFFF", s=50, alpha=0.9)
    col_g = ax.scatter([], [], marker='s', edgecolor='k', linewidth=1, facecolor="gray", s=50, alpha=0.9)

    x_offset = 0.45

    fig.legend(
        [pro, ama], ["Professional", "Amateur"],
        fontsize=10, ncol=2,
        bbox_to_anchor=(0.0 + x_offset, -0.005),
        title="Group"
    )
    fig.legend(
        [col_w, col_g], ["White", "Gray"],
        fontsize=10, ncol=2,
        bbox_to_anchor=(0.22 + x_offset, -0.005),
        title="Target color"
    )
    fig.legend(
        [rad_small, rad_large], ["Small", "Large"],
        fontsize=10, ncol=2,
        bbox_to_anchor=(0.441 + x_offset, -0.005),
        title="Target radius"
    )
    fig.legend(
        [spd_stt, spd_mov], ["Stationary", "Moving"],
        fontsize=10, ncol=2,
        bbox_to_anchor=(0.55, -0.085),
        title="Target speed"
    )
    fig.legend(
        [sen_hab, sen_def], ["Habitual", "Default"],
        fontsize=10, ncol=2,
        bbox_to_anchor=(0.81, -0.085),
        title="Mouse sensitivity"
    )
    figure_save(fig, os.path.join(figpath, f"model_fitting_visual.png"), save_svg=True)

    

def visualize_parameter_inference_result(inferer:AnSTwoLevelConditionalInferer):
    figpath = os.path.join(DIR_TO_DATA, f"data/ijhcs25_data/infer_result/{inferer.name}/visualization")
    os.makedirs(figpath, exist_ok=True)

    param_list = inferer.simulator.total_target_param

    feature_order = ["tier", "sen_level", "col_level", "rad_level", "spd_level"]
    xtick_list = [
        ["Pro.", "Ama."],
        ["Habitual", "Default"],
        ["White", "Gray"],
        ["Small", "Large"],
        ["Stat.", "Moving"]
    ]
    xlabel_list = ["Group", "Sensitivity", "Color", "Radius", "Speed"]

    inv_df = pd.read_csv(os.path.join(DIR_TO_DATA, f"data/ijhcs25_data/infer_result/{inferer.name}/inv_param.csv"))
    var1_df = pd.read_csv(os.path.join(DIR_TO_DATA, f"data/ijhcs25_data/infer_result/{inferer.name}/var1_param.csv"))
    var2_df = pd.read_csv(os.path.join(DIR_TO_DATA, f"data/ijhcs25_data/infer_result/{inferer.name}/var2_param.csv"))

    param_df = pd.merge(var2_df, inv_df[["user", *inferer.simulator.main_param_list]], on=["user"], how='left')
    param_df = pd.merge(param_df, var1_df[["user", "col_level", *inferer.simulator.sub1_param_list]], 
                        on=["user", "col_level"], how='left')

    fig, axs = figure_grid(len(param_list), len(feature_order), size_ax=np.array([4, 2.2]) / 1.5)

    for (i, param), j in product(enumerate(param_list), range(len(xlabel_list))):
        ax = axs[i][j]

        sns.lineplot(data=param_df, x=feature_order[j], y=param, marker='s', color='k',
                     linewidth=1, markersize=7, 
                     errorbar=('ci', 95), err_style='bars', err_kws={'capsize': 3}, 
                     ax=ax)
        
        ax.set_xlabel("")
        if j == 0:
            ax.set_ylabel(f"Inferred {PARAM_SYMBOL[param]}")
        else:
            ax.set_ylabel("")
        ax.set_xticks(np.arange(2), xtick_list[j])
        ax.grid(axis='y', alpha=0.5, linestyle='--')

        labels = ax.get_xticklabels()
        for i, lbl in enumerate(labels):
            if i == 0:
                lbl.set_ha("left")
            elif i == len(labels) - 1:
                lbl.set_ha("right")
            else:
                lbl.set_ha("center")
    
    for j, l in enumerate(xlabel_list):
        axs[-1][j].set_xlabel(l)
    
    figure_save(fig, os.path.join(figpath, "parameter_descriptive.png"), save_svg=True)



def construct_rm_anova_format(inferer:AnSTwoLevelConditionalInferer):
    ### Depth 1 subparam
    param_csv_path = os.path.join(DIR_TO_DATA, f"data/ijhcs25_data/infer_result/{inferer.name}/var1_param.csv")
    df = pd.read_csv(param_csv_path)

    target_param = inferer.simulator.sub1_param_list

    data = list()
    for user in range(N_USER):
        user_data = dict(tier=['PRO', 'AMA'][user//10], user=user)
        for param, col in product(target_param, range(2)):
            v = df[
                (df["user"] == user) &
                (df["col_level"] == col)
            ][param].to_numpy()[0]
            user_data[f"{param}_{col}"] = v
        data.append(user_data)
    
    data = pd.DataFrame(data)
    data.to_csv(os.path.join(DIR_TO_DATA, f"data/ijhcs25_data/infer_result/{inferer.name}/var1_param_rm_anova.csv"), index=False)

    ### Depth 2 subparam
    param_csv_path = os.path.join(DIR_TO_DATA, f"data/ijhcs25_data/infer_result/{inferer.name}/var2_param.csv")
    df = pd.read_csv(param_csv_path)

    target_param = inferer.simulator.sub2_param_list

    data = list()
    for user in range(N_USER):
        user_data = dict(tier=['PRO', 'AMA'][user//10], user=user)
        for param, col, sen, spd, rad in product(target_param, range(2), range(2), range(2), range(2)):
            v = df[
                (df["user"] == user) &
                (df["col_level"] == col) &
                (df["sen_level"] == sen) &
                (df["spd_level"] == spd) &
                (df["rad_level"] == rad)
            ][param].to_numpy()[0]
            user_data[f"{param}_{col}{sen}{spd}{rad}"] = v
        data.append(user_data)
    
    data = pd.DataFrame(data)
    data.to_csv(os.path.join(DIR_TO_DATA, f"data/ijhcs25_data/infer_result/{inferer.name}/var2_param_rm_anova.csv"), index=False)







if __name__ == "__main__":
    # amort_name = "AnSCondAmort_PTE_08071040"
    amort_name = "AnSCondAmort_PTE_01151337"
    ckpt = 100

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    inferer = AnSTwoLevelConditionalInferer(amort_name, ckpt)
    dataset = ExperimentIJHCS()

    infer_and_fitting(inferer, dataset)

    visualize_parameter_inference_result(inferer)
    visualize_fitting_performance(inferer)
    construct_rm_anova_format(inferer)

    result = compute_model_fitting_metric(inferer)
    mt = METRICS[0]
    print(np.mean(np.abs(result["user_average_metric"][mt]["user"] - result["user_average_metric"][mt]["simul"])))
    print(np.mean(np.abs(result["user_cond_average_metric"][mt]["user"] - result["user_cond_average_metric"][mt]["simul"])))

    mt = METRICS[1]
    print(np.mean(np.abs(result["user_average_metric"][mt]["user"] - result["user_average_metric"][mt]["simul"])))
    print(np.mean(np.abs(result["user_cond_average_metric"][mt]["user"] - result["user_cond_average_metric"][mt]["simul"])))

    mt = METRICS[2]
    print(np.mean(np.abs(result["user_average_metric"][mt]["user"] - result["user_average_metric"][mt]["simul"])))
    print(np.mean(np.abs(result["user_cond_average_metric"][mt]["user"] - result["user_cond_average_metric"][mt]["simul"])))

    # 0.02803203078857643
    # 0.041752793820740965
    # 0.13819354632040612
    # 0.17399292643265085
    # 0.9258136015874827
    # 1.0404516525820484