import os
from pathlib import Path
# from tqdm import tqdm
import numpy as np
import pandas as pd
# from joblib import Parallel, delayed

from itertools import product

from ..utils.myplot import figure_grid, figure_save
# from ..utils.myutils import pickle_save, pickle_load
from ..utils.mymath import get_r_squared

from ..config.constant import FIELD #, PARAM_SYMBOL, NAME_ABBR, NAME_FULL
# from ..datamanagekit.ijhcs25_data import ExperimentIJHCS

from scipy import stats
import seaborn as sns

from matplotlib.lines import Line2D

import warnings
warnings.filterwarnings("ignore", message=".*use_inf_as_na.*")

DIR_TO_DATA = Path(__file__).parent.parent.parent
BASE_COLOR = "#F33276"
COND_COLOR = "#0000FE"

def visualize_parameter_comparison(cond, base, name, legend_title):
    fig, axs = figure_grid(6, 3, size_ax=np.array([4, 2.2]) / 1.5)

    base_cog_df = pd.read_csv(os.path.join(DIR_TO_DATA, f"data/prolific_data/infer_result/{base}/fmode_3_param.csv"))
    base_rew_df = pd.read_csv(os.path.join(DIR_TO_DATA, f"data/prolific_data/infer_result/{base}/rew_merged_df.csv"))
    cond_cog_df = pd.read_csv(os.path.join(DIR_TO_DATA, f"data/prolific_data/infer_result/{cond}/cog_param.csv"))
    cond_rew_df = pd.read_csv(os.path.join(DIR_TO_DATA, f"data/prolific_data/infer_result/{cond}/rew_merged_df.csv"))

    xkeys = ["stake_level", "period_level", "difficulty_level"]
    xlabels = [
        [0, 0.1, 1, 10],
        ["Early", "Middle", "Late"],
        ["Easy", "Hard"]
    ]

    ### Cognitive
    param_list = ["theta_m", "theta_c"]
    ylabels = ["Motor noise", "Clock noise"]

    # Fitting mode 3 - Stake only
    
    for (i, param), j in product(enumerate(param_list), range(3)):
        ax = axs[i][j]

        sns.lineplot(data=base_cog_df, x=xkeys[j], y=param, marker='o', color=BASE_COLOR,
                     linewidth=1, markersize=7, 
                     errorbar=('ci', 95), err_style='bars', err_kws={'capsize': 3}, 
                     ax=ax)
        
        ax.set_xlabel("")
        if j == 0:
            ax.set_ylabel(ylabels[i])
        else:
            ax.set_ylabel("")
        ax.set_xticks(np.arange(len(xlabels[j])), xlabels[j])
        ax.grid(axis='y', alpha=0.5, linestyle='--')

        cond_values = cond_cog_df[param].to_numpy()
        mean = np.mean(cond_values)
        se = stats.sem(cond_values)  # standard error
        t_crit = stats.t.ppf(1 - 0.05/2, df=len(cond_values)-1)
        ci = se * t_crit
        ax.axhline(mean, color=COND_COLOR, linestyle='-', linewidth=1)
        ax.axhspan(mean - ci, mean + ci, color=COND_COLOR, alpha=0.1)
    

    ### Reward
    param_list = ["rew_succ", "rew_fail", "actual_time_cost", "actual_effort_cost"]
    ylabels = ["Success reward", "Fail penalty", "Time cost", "Effort cost"]
    for (i, param), j in product(enumerate(param_list), range(3)):
        ax = axs[i+2][j]

        sns.lineplot(data=base_rew_df.assign(adjusted_x=base_rew_df[xkeys[j]] - 0.03),
                     x="adjusted_x", y=param, marker='o', color=BASE_COLOR,
                     linewidth=1, markersize=7, 
                     errorbar=('ci', 95), err_style='bars', err_kws={'capsize': 3}, 
                     ax=ax)
        
        sns.lineplot(data=cond_rew_df.assign(adjusted_x=cond_rew_df[xkeys[j]] + 0.03),
                     x="adjusted_x", y=param, marker='s', color=COND_COLOR,
                     linewidth=1, markersize=7, 
                     errorbar=('ci', 95), err_style='bars', err_kws={'capsize': 3}, 
                     ax=ax)
        
        ax.set_xlabel("")
        if j == 0:
            ax.set_ylabel(ylabels[i])
        else:
            ax.set_ylabel("")
        ax.set_xticks(np.arange(len(xlabels[j])), xlabels[j])
        ax.grid(axis='y', alpha=0.5, linestyle='--')

    # ymin = [0.158, 0.209, 0.209, 0.115, 0.137, 0.128, 8.149, 8.593, 8.511, 7.673, 8.480, 8.751, 5.384, 6.611, 7.097, 6.195, 8.580, 6.586]
    # ymax = [0.368, 0.357, 0.358, 0.280, 0.276, 0.278, 11.209, 10.849, 10.340, 13.283, 12.181, 12.043, 12.063, 10.362, 9.864, 17.000, 14.434, 15.402]

    # for r, c in product(range(6), range(3)):
    #     ax = axs[r][c]
    #     ax.set_ylim(ymin[r * 3 + c], ymax[r * 3 + c])

    # temp_min = list()
    # temp_max = list()
    for r, c in product(range(6), range(3)):
        ax = axs[r][c]
        ylim_min, ylim_max = ax.get_ylim()
        ax.set_ylim(ylim_min - 0.2 * (ylim_max - ylim_min), ylim_max + 0.2 * (ylim_max - ylim_min))

        # ylim_min, ylim_max = ax.get_ylim()
        # temp_min.append(ylim_min)
        # temp_max.append(ylim_max)

    
    axs[-1][0].set_xlabel("Stake")
    axs[-1][1].set_xlabel("Period")
    axs[-1][2].set_xlabel("Difficulty")

    cond = Line2D([0], [0], marker='s', color=COND_COLOR, linestyle='-', linewidth=1, markersize=7)
    base = Line2D([0], [0], marker='o', color=BASE_COLOR, linestyle='-', linewidth=1, markersize=7)

    fig.legend(
        [cond, base],
        ["Hierarchical amortized inference", "Non-hierarchical amortized inference"],
        fontsize=10, ncol=2,
        bbox_to_anchor=(0.75, -0.005),
        title=legend_title
    )

    figure_save(fig, os.path.join(DIR_TO_DATA, f"data/visualization/inference/pnc_infer_comparison/{name}"), save_svg=True)

    # return np.array(temp_min), np.array(temp_max)


def visualize_fitting_performance():
    cond_den = "CondAmort4_DEN_01221456"
    cond_pon = "CondAmort_PTE_07251129"
    base_den = "BaseAmort_DEN_07251102"
    base_pon = "BaseAmort_PTE_07250658"

    cond_d_df = pd.read_csv(os.path.join(DIR_TO_DATA, f"data/prolific_data/infer_result/{cond_den}/mf_result.csv"))
    cond_p_df = pd.read_csv(os.path.join(DIR_TO_DATA, f"data/prolific_data/infer_result/{cond_pon}/mf_result.csv"))
    base_d_df = pd.read_csv(os.path.join(DIR_TO_DATA, f"data/prolific_data/infer_result/{base_den}/fmode_3_mf.csv"))
    base_p_df = pd.read_csv(os.path.join(DIR_TO_DATA, f"data/prolific_data/infer_result/{base_pon}/fmode_3_mf.csv"))

    cond = Line2D([0], [0], color=COND_COLOR, linestyle='-', linewidth=1)
    base = Line2D([0], [0], color=BASE_COLOR, linestyle='-', linewidth=1)

    fig, axs = figure_grid(2, 4, size_ax=np.array([3, 3]) / 1.1)
    for i, (_cdf, _bdf) in enumerate([(cond_d_df, base_d_df), (cond_p_df, base_p_df)]):
        # User aggregation
        cdf = _cdf.groupby('user', as_index=False).mean()
        bdf = _bdf.groupby('user', as_index=False).mean()

        for j, mt in enumerate(["MVT", "HIT"]):
            ax = axs[i][j]

            x = cdf[f"{FIELD[mt]}_user"].to_numpy()
            y = cdf[f"{FIELD[mt]}_simul"].to_numpy()
            if mt == "HIT":
                x *= 100
                y *= 100
            r2_c = get_r_squared(x, y)

            ax.scatter(x, y, s=5, alpha=0.3, zorder=500, linewidth=0.75, color=COND_COLOR, marker='o')
            sns.regplot(
                x=x, y=y,
                scatter_kws={"color": "black", "alpha": 0, "marker": 'o'},
                line_kws={"color": COND_COLOR, "lw": 2, "alpha": 1, "zorder": 500},
                ax=ax,
                ci=None,
            )

            x = bdf[f"{FIELD[mt]}_user"].to_numpy()
            y = bdf[f"{FIELD[mt]}_simul"].to_numpy()
            if mt == "HIT":
                x *= 100
                y *= 100
            r2_b = get_r_squared(x, y)

            ax.scatter(x, y, s=5, alpha=0.3, zorder=300, linewidth=0.75, color=BASE_COLOR, marker='o')
            sns.regplot(
                x=x, y=y,
                scatter_kws={"color": "black", "alpha": 0, "marker": 'o'},
                line_kws={"color": BASE_COLOR, "lw": 2, "alpha": 1, "zorder": 300},
                ax=ax,
                ci=None,
            )
            ax.legend([cond, base], [f"$R^2$={r2_c:.2f}", f"$R^2$={r2_b:.2f}"], loc="lower right")

            if j == 0:
                _min, _max = 0.1, 2.2
                ax.set_xlabel("User MT (s)")
                ax.set_ylabel("Simulation MT (s)")
            else:
                _min, _max = -2, 102
                ax.set_xlabel("User ACC (%)")
                ax.set_ylabel("Simulation ACC (%)")

            ax.plot([_min, _max], [_min, _max], color='gray', linewidth=0.7, linestyle='--', zorder=1)
            ax.set_xlim(_min, _max)
            ax.set_ylim(_min, _max)
            ax.grid(linestyle='--', color='gray', linewidth=0.5)
            ax.set_aspect('equal')

        
        # User x task aggregation
        cdf = _cdf.groupby(["user", "stake_level", "period_level", "difficulty_level"], as_index=False).mean()
        bdf = _bdf.groupby(["user", "stake_level", "period_level", "difficulty_level"], as_index=False).mean()

        for j, mt in enumerate(["MVT", "HIT"]):
            ax = axs[i][j+2]
            r2_c = list()
            r2_b = list()

            for user in range(90):
                x = cdf[cdf["user"] == user][f"{FIELD[mt]}_user"].to_numpy()
                y = cdf[cdf["user"] == user][f"{FIELD[mt]}_simul"].to_numpy()
                if mt == "HIT":
                    x *= 100
                    y *= 100
                r2_c.append(get_r_squared(x, y))

                ax.scatter(x, y, s=5, alpha=0.1, zorder=500, linewidth=0.75, color=COND_COLOR, marker='o')
                sns.regplot(
                    x=x, y=y,
                    scatter_kws={"color": "black", "alpha": 0, "marker": 'o'},
                    line_kws={"color": COND_COLOR, "lw": 0.7, "alpha": 0.3, "zorder": 500},
                    ax=ax,
                    ci=None,
                )
            
                x = bdf[bdf["user"] == user][f"{FIELD[mt]}_user"].to_numpy()
                y = bdf[bdf["user"] == user][f"{FIELD[mt]}_simul"].to_numpy()
                if mt == "HIT":
                    x *= 100
                    y *= 100
                r2_b.append(get_r_squared(x, y))

                ax.scatter(x, y, s=5, alpha=0.1, zorder=300, linewidth=0.75, color=BASE_COLOR, marker='o')
                sns.regplot(
                    x=x, y=y,
                    scatter_kws={"color": "black", "alpha": 0, "marker": 'o'},
                    line_kws={"color": BASE_COLOR, "lw": 0.7, "alpha": 0.3, "zorder": 300},
                    ax=ax,
                    ci=None,
                )
            ax.legend([cond, base], [f"Avg $R^2$={np.mean(r2_c):.2f}", 
                                     f"Avg $R^2$={np.mean(r2_b):.2f}"], loc="lower right", fontsize=8)

            if j == 0:
                _min, _max = 0.1, 2.2
                ax.set_xlabel("User MT (s)")
                ax.set_ylabel("Simulation MT (s)")
            else:
                _min, _max = -2, 102
                ax.set_xlabel("User ACC (%)")
                ax.set_ylabel("Simulation ACC (%)")

            ax.plot([_min, _max], [_min, _max], color='gray', linewidth=0.7, linestyle='--', zorder=1)
            ax.set_xlim(_min, _max)
            ax.set_ylim(_min, _max)
            ax.grid(linestyle='--', color='gray', linewidth=0.5)
            ax.set_aspect('equal')

    fig.legend(
        [cond, base],
        ["Hierarchical amortized inference", "Non-hierarchical amortized inference"],
        fontsize=10, ncol=2,
        bbox_to_anchor=(0.75, -0.005)
    )
    figure_save(fig, os.path.join(DIR_TO_DATA, f"data/visualization/inference/pnc_infer_comparison/fitting_performance.png"), 
                dpi=300, save_svg=True)
            





if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    visualize_parameter_comparison(
        "CondAmort4_DEN_01221456",
        "BaseAmort_DEN_07251102",
        "pnc_density_estimation",
        "Density estimation"
    )

    visualize_parameter_comparison(
        "CondAmort_PTE_07251129",
        "BaseAmort_PTE_07250658",
        "pnc_point_estimation",
        "Point estimation"
    )
    visualize_fitting_performance()

    # ymin = np.minimum(n1, n2)
    # ymax = np.maximum(x1, x2)

    # print("[" + ", ".join(f"{v:.3f}" for v in ymin) + "]")
    # print("[" + ", ".join(f"{v:.3f}" for v in ymax) + "]")