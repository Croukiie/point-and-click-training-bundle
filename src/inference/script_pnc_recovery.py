import numpy as np
import pandas as pd
import os
# import torch
from tqdm import tqdm
from ..utils.myplot import figure_grid, draw_r2_plot
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from einops import rearrange

from ..inference.pnc_dataset_2step import PnCConditionalAmortizerValidationDataset
# from ..inference.pnc_inferer_2step import PnCConditionalInferer
from ..inference.pnc_inferer import PnCInferer
from ..utils.mymath import get_r_squared

##### ==================== PNC ==================== ####
# dataset = PnCConditionalAmortizerValidationDataset(verbose=True)
# param_glob, param_loc, stats, _, _ = dataset.sample(n_trial=64)
# param_glob, param_loc = dataset.simulator.convert_param_z_to_w(main_outputs=param_glob,
#                                                                sub_outputs=rearrange(param_loc, 'n s d -> (n s) d'))
# gt_params = np.concatenate([np.repeat(param_glob, 24, axis=0), param_loc], axis=1)
# param_list = dataset.simul_main_param + dataset.simul_sub_param
# gt_param_df = pd.DataFrame(gt_params, columns=param_list)
# stats = rearrange(stats, 'glob loc trial feat -> (glob loc) trial feat')


# def run_nai_inference(model_name):
#     inferer = PnCInferer(model_name)
#     inferred_df = list()
#     for i in tqdm(range(stats.shape[0])):
#         stat = stats[i]
#         w = inferer.infer(stat, return_z=False, return_w=True, return_dict=True, return_samples=False)
#         inferred_df.append(w)
#     inferred_df = pd.DataFrame(inferred_df)
#     return inferred_df


# for model in ["BaseAmort_PTE_07250658", "BaseAmort_DEN_07251102"]:
#     inferred_df = run_nai_inference(model)
#     r2_info = dict()
#     for param in param_list:
#         r2 = get_r_squared(gt_param_df[param].values, inferred_df[param].values)
#         r2_info[param] = r2
#     print(f"Model: {model}", end='')
#     for k, v in r2_info.items():
#         print(f" {k}: {v:.2f}", end=',')
#     print(f"avg={np.mean(list(r2_info.values())):.2f} ")

# Model: BaseAmort_PTE_07250658 theta_m: 0.55, theta_c: 0.78, rew_succ: 0.33, rew_fail: 0.32, time_penalty: 0.63, effort_cost: 0.62,avg=0.54 
# Model: BaseAmort_DEN_07251102 theta_m: 0.55, theta_c: 0.78, rew_succ: 0.27, rew_fail: 0.25, time_penalty: 0.60, effort_cost: 0.61,avg=0.51



#### =================== AnS (Non-hierarchical) ==================== ####
from ..inference.ans_inferer_3step import AnSTwoLevelConditionalInferer
from ..inference.ans_inferer import AnSInferer
from ..inference.ans_dataset_3step import AnSTwoLevelConditionalAmortizerValidationDataset
from ..inference.ans_dataset import AnSAmortizerValidationDataset

# Hierarchical dataset
# dataset = AnSTwoLevelConditionalAmortizerValidationDataset(verbose=True)
# param_glob, param_loc1, param_loc2, stats, _ = dataset.sample(64)
# gt_param_list = [dataset.simulator.convert_param_z_to_w(None, gtg, gtl1, gtl2) 
#                  for gtg, gtl1, gtl2 in zip(param_glob, param_loc1, param_loc2)]
# param_glob, param_loc1, param_loc2 = (
#     np.array([gt[0] for gt in gt_param_list]),
#     np.array([gt[1] for gt in gt_param_list]),
#     np.array([gt[2] for gt in gt_param_list])
# )
# param_loc1 = rearrange(param_loc1, 'n s d -> (n s) d')
# param_loc2 = rearrange(param_loc2, 'n s1 s2 d -> (n s1 s2) d')

# gt_params = np.concatenate([
#     np.repeat(param_glob, 16, axis=0),
#     np.repeat(param_loc1, 8, axis=0),
#     param_loc2
# ], axis=1)
# param_list = dataset.simul_main_param + dataset.simul_sub1_param + dataset.simul_sub2_param
# gt_param_df = pd.DataFrame(gt_params, columns=param_list)

# stats = rearrange(stats, 'glob loc1 loc2 trial feat -> (glob loc1 loc2) trial feat')

# inferer = AnSInferer("AnSBaseAmort_PTE_01140125")
# inferred_df = list()
# for i in tqdm(range(stats.shape[0])):
#     stat = stats[i]
#     w = inferer.infer(stat, return_z=False, return_w=True, return_dict=True)
#     inferred_df.append(w)
# inferred_df = pd.DataFrame(inferred_df)
# r2_info = dict()
# for param in param_list:
#     r2 = get_r_squared(gt_param_df[param].values, inferred_df[param].values)
#     r2_info[param] = r2
# for k, v in r2_info.items():
#     print(f" {k}: {v:.2f}", end=',')
# print(f"avg={np.mean(list(r2_info.values())):.2f} ")

# fig, axs = figure_grid(2, 4, size_ax=3)
# for i, p in enumerate(param_list):
#     x = gt_param_df[p].values
#     y = inferred_df[p].values

#     draw_r2_plot(
#         ax=axs[i//4][i%4],
#         xdata=x,
#         ydata=y,
#         xlabel=f"True {p}",
#         ylabel=f"Inferred {p}",
#         aspect = None
#     )
# plt.tight_layout()
# plt.show()


# Non-hierarchical dataset
dataset = AnSAmortizerValidationDataset(verbose=True)
gt_param, obs_data = dataset.sample(64)
gt_param = dataset.simulator.convert_param_z_to_w(gt_param)
param_list = ["theta_m", "theta_p", "theta_s", "theta_c", "rew_succ", "rew_fail", "decay_succ", "decay_fail"]
gt_param_df = pd.DataFrame(gt_param, columns=param_list)

inferer = AnSInferer("AnSBaseAmort_PTE_01140125")
inferred_df = list()
for i in tqdm(range(100)):
    stat = obs_data[i][0]
    w = inferer.infer(stat, return_z=False, return_w=True, return_dict=True)
    inferred_df.append(w)
inferred_df = pd.DataFrame(inferred_df)
r2_info = dict()
for param in param_list:
    r2 = get_r_squared(gt_param_df[param].values, inferred_df[param].values)
    r2_info[param] = r2
for k, v in r2_info.items():
    print(f" {k}: {v:.2f}", end=',')
print(f"avg={np.mean(list(r2_info.values())):.2f} ")

fig, axs = figure_grid(2, 4, size_ax=3)
for i, p in enumerate(param_list):
    x = gt_param_df[p].values
    y = inferred_df[p].values

    draw_r2_plot(
        ax=axs[i//4][i%4],
        xdata=x,
        ydata=y,
        xlabel=f"True {p}",
        ylabel=f"Inferred {p}",
        aspect = None
    )
plt.tight_layout()
plt.show()



#### =================== AnS (Hierarchical) ==================== ####
# stats = rearrange(stats, '(glob loc1 loc2) trial feat -> glob (loc1 loc2 trial) feat',
#                   glob=100, loc1=2, loc2=8)

# inferer = AnSTwoLevelConditionalInferer("AnSCondAmort_PTE_01151337")
# inferred_df = list()

# for i in tqdm(range(stats.shape[0])):
#     _, w, infer_glob, infer_loc1, infer_loc2 = inferer.infer(
#         stats[i],
#         cond1_index=np.repeat(np.arange(2), 8 * 64),
#         cond12_index=np.repeat(np.arange(2), 8 * 64) * 8 + np.tile(np.repeat(np.arange(8), 64), 2),
#         return_df=True
#     )
#     glob_df = pd.DataFrame([infer_glob for _ in range(16)])
#     loc1_df = pd.DataFrame([infer_loc1[j//8] for j in range(16)])
#     loc2_df = pd.DataFrame([infer_loc2[j] for j in range(16)])
#     inferred_df.append(pd.concat([glob_df, loc1_df, loc2_df], axis=1))

# inferred_df = pd.concat(inferred_df, axis=0).reset_index(drop=True)
# r2_info = dict()
# for param in param_list:
#     r2 = get_r_squared(gt_param_df[param].values, inferred_df[param].values)
#     r2_info[param] = r2
# for k, v in r2_info.items():
#     print(f" {k}: {v:.2f}", end=',')
# print(f"avg={np.mean(list(r2_info.values())):.2f} ")

# fig, axs = figure_grid(2, 4, size_ax=3)
# for i, p in enumerate(param_list):
#     x = gt_param_df[p].values
#     y = inferred_df[p].values

#     draw_r2_plot(
#         ax=axs[i//4][i%4],
#         xdata=x,
#         ydata=y,
#         xlabel=f"True {p}",
#         ylabel=f"Inferred {p}",
#         aspect = None
#     )
# plt.tight_layout()
# plt.show()