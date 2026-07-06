"""
Code upgraded by June-Seop Yoon

Trainer for amortized inference
Orignial code was written by Dr. Hee-seung Moon (https://github.com/hsmoon121/amortized-inference-hci)
"""

import os
from time import time
from copy import deepcopy
from pathlib import Path
from abc import ABC
from tqdm import tqdm
import numpy as np
import pandas as pd
from typing import List, Optional
from einops import rearrange
import seaborn as sns
from itertools import product

import torch
import torch.nn.functional as F

from ..config_ans.config import INF
from ..config_ans.constant import PARAM_SYMBOL, FIELD, FOLDER
from .ans_dataset_3step import (
    AnSTwoLevelConditionalAmortizerTrainingDataset,
    AnSTwoLevelConditionalAmortizerValidationDataset
)
from ..nets.amortizer import TwoLevelConditionalRegressionForTrialData, AmortizerForTrialData
from ..utils.myutils import get_compact_timestamp_str, save_dict_to_yaml, pickle_save
from ..utils.schedulers import CosAnnealWR
from ..utils.loggers import Logger
from ..utils.myplot import figure_grid, figure_save, draw_r2_plot
from ..utils.mymath import linear_normalize, np_groupby, kl_divergence
from ..utils.myutils import load_config

import warnings
warnings.filterwarnings("ignore", message=".*use_inf_as_na.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*is_categorical_dtype.*")

DIR_TO_DATA = Path(__file__).parent.parent.parent

class AnSTwoLevelConditionalAmortizerTrainer(ABC):
    def __init__(
        self,
        load_model=None,
        name=None,
        train_config='default_lv2_cond',         # Training hyperparameter
        simul_config='default_lv2_cond',         # Simulator model setting
        data_config='default',    # Data processing setting
    ):
        if load_model is not None:
            name = load_model
            train_config = load_config(os.path.join(DIR_TO_DATA, f"data/{FOLDER.AMORTIZER3}/models/{name}/train_setting.yaml"))
            simul_config = load_config(os.path.join(DIR_TO_DATA, f"data/{FOLDER.AMORTIZER3}/models/{name}/simulator_setting.yaml"))
            data_config = load_config(os.path.join(DIR_TO_DATA, f"data/{FOLDER.AMORTIZER3}/models/{name}/data_normalization_setting.yaml"))

        # Initialize simulator, amortizer, and dataset
        self.iter = 0
        self.train_dataset = AnSTwoLevelConditionalAmortizerTrainingDataset(
            simulator_config=simul_config, 
            normalize_config=data_config
        )
        self.valid_dataset = AnSTwoLevelConditionalAmortizerValidationDataset(
            simulator_config=simul_config, 
            normalize_config=data_config
        )
        self.simulator = deepcopy(self.train_dataset.simulator.simulator)

        # List of target parameters
        self.total_target_param = self.train_dataset.simulator.total_target_param
        self.main_target_param = self.train_dataset.simulator.main_param_list
        self.sub1_target_param = self.train_dataset.simulator.sub1_param_list
        self.sub2_target_param = self.train_dataset.simulator.sub2_param_list

        # Configuration setting
        self.train_config = INF.training[train_config] if isinstance(train_config, str) else train_config
        self.data_config = INF.normalize[data_config] if isinstance(data_config, str) else data_config

        if load_model is None:
            stat_size = len(self.data_config.stat.list)
            self.train_config.amortizer.encoder.stat_sz = stat_size
            self.train_config.amortizer.encoder.traj_sz = len(self.data_config.traj.list) \
                if not self.data_config.traj.ignore_traj else 0
            if isinstance(simul_config, str):
                self.train_config.amortizer.encoder.transformer.max_step = \
                    round(INF.simulator[simul_config].traj_downsample * self.simulator.env.user.truncate_time / 1000) + 5
            else:
                self.train_config.amortizer.encoder.transformer.max_step = \
                    round(simul_config["traj_downsample"] * self.simulator.env.user.truncate_time / 1000) + 5
                
            concat_feature_vector = self.train_config.amortizer.get("concat_feature_vector", False)

            self.train_config.amortizer.linear_main.out_sz = len(self.main_target_param)
            self.train_config.amortizer.linear_main.in_sz = \
                self.train_config.amortizer.trial_encoder.attention.out_sz
            
            self.train_config.amortizer.linear_sub1.out_sz = len(self.sub1_target_param)
            self.train_config.amortizer.linear_sub1.in_sz = \
                self.train_config.amortizer.trial_encoder.attention.out_sz + len(self.main_target_param) \
                if not concat_feature_vector else \
                self.train_config.amortizer.trial_encoder.attention.out_sz * 2
            
            self.train_config.amortizer.linear_sub2.out_sz = len(self.sub2_target_param)
            self.train_config.amortizer.linear_sub2.in_sz = \
                self.train_config.amortizer.trial_encoder.attention.out_sz + \
                    len(self.main_target_param) + len(self.sub1_target_param) \
                    if not concat_feature_vector else \
                self.train_config.amortizer.trial_encoder.attention.out_sz * 3

        # Amortizer name
        self.name = name if name is not None else f"{'PTE' if self.train_config.point_estimation else 'INN'}_{get_compact_timestamp_str()}"

        # User dataset
        self.user_dataset = pd.read_csv(os.path.join(DIR_TO_DATA, "data/ijhcs25_data/summary/general_summary.csv"))
        self.n_user = 20
        self.n_cond1 = 2
        self.n_cond12 = 16   # 2 * 8
        self.user_dataset_cond1_key = "condition1_id"
        self.user_dataset_cond2_key = "condition2_id"

        # Warmup
        self.point_estimation = self.train_config.point_estimation
        if self.point_estimation:
            amortizer_fn = TwoLevelConditionalRegressionForTrialData
        else:
            raise NotImplementedError("INN not implemented yet.")
        self.amortizer = amortizer_fn(config=self.train_config["amortizer"])

        self.lr = self.train_config["learning_rate"]
        self.lr_gamma = self.train_config["lr_gamma"]
        self.clipping = self.train_config["clipping"]
        self.optimizer = torch.optim.Adam(self.amortizer.parameters(), lr=1e-9)
        self.scheduler = CosAnnealWR(self.optimizer, T_0=10, T_mult=1, eta_max=self.lr, T_up=1, gamma=self.lr_gamma)

        # Loss weight setting
        self.loss_weight_vector = np.concatenate((
            np.array(self.train_config.amortizer.loss_weight.main_param),
            np.tile(self.train_config.amortizer.loss_weight.sub1_param, self.train_dataset.num_of_sub1param),
            np.tile(self.train_config.amortizer.loss_weight.sub2_param, self.train_dataset.num_of_sub1param * self.train_dataset.num_of_sub2param),
        ))
        self.loss_weight_vector = torch.from_numpy(self.loss_weight_vector).sqrt().to(self.amortizer.device)

        self.model_path = os.path.join(DIR_TO_DATA, f"data/{FOLDER.AMORTIZER3}/models/{self.name}/pts")
        self.board_path = os.path.join(DIR_TO_DATA, f"data/{FOLDER.AMORTIZER3}/models/{self.name}/board")
        self.result_path = os.path.join(DIR_TO_DATA, f"data/{FOLDER.AMORTIZER3}/models/{self.name}/results")

        if load_model is None:
            save_dict_to_yaml(self.train_config, os.path.join(DIR_TO_DATA, f"data/{FOLDER.AMORTIZER3}/models/{self.name}/train_setting.yaml"))
            save_dict_to_yaml(self.train_dataset.norm_config, os.path.join(DIR_TO_DATA, f"data/{FOLDER.AMORTIZER3}/models/{self.name}/data_normalization_setting.yaml"))
            save_dict_to_yaml(self.train_dataset.sim_config, os.path.join(DIR_TO_DATA, f"data/{FOLDER.AMORTIZER3}/models/{self.name}/simulator_setting.yaml"))
            
        self.clipping = float("Inf")
    
        # Prepare data process for inference
        self.stat_range = np.array([
            [self.train_dataset.norm_config.stat.range[feat].min,
            self.train_dataset.norm_config.stat.range[feat].max] for feat in self.train_dataset.norm_config.stat.list
        ])
        self.traj_range = np.array([
            [self.train_dataset.norm_config.traj.range[feat].min,
            self.train_dataset.norm_config.traj.range[feat].max] for feat in self.train_dataset.norm_config.traj.list
        ])

        if load_model is not None:
            self.load(load_model)

    
    def train(
        self,
        n_iter=100,
        step_per_iter=2048,
        batch_sz=64,
        board=True,
        save_freq=10,
    ):
        """
        Training loop

        n_iter (int): Number of training iterations
        step_per_iter (int): Number of training steps per iteration
        batch_sz (int): Batch size
        n_trial (int): Number of trials for each user data
        board (bool): Whether to use tensorboard (default: True)
        """
        iter = self.iter
        last_step = self.iter * step_per_iter

        self.logger = Logger(
            self.name,
            last_step=last_step, 
            board=board, 
            board_path=self.board_path
        )

        # Training iterations
        losses = dict()
        print(f"\n[ Training - {self.name} ]")
        for iter in range(self.iter + 1, n_iter + 1):
            losses[iter] = []

            # Training loop
            with tqdm(total=step_per_iter, desc=f" Iter {iter}") as progress:
                for step in range(step_per_iter):
                    batch_param, batch_stat, batch_traj = self.train_dataset.sample(
                        batch_sz=batch_sz,
                        sim_per_param=None,     # Load full episodes per parameter
                        stretch_param=True
                    )

                    # Training step
                    loss = self._train_step(batch_param, batch_stat, batch_traj)
                    losses[iter].append(loss)

                    # Logging
                    if step % 10 == 0:
                        self.logger.write_scalar(train_loss=loss, lr=self.scheduler.get_last_lr()[0])
                    progress.set_postfix_str(f"Avg.Loss: {np.mean(losses[iter]):.3f}")
                    progress.update(1)
                    self.logger.step()
                    self.scheduler.step((iter-1) + step/step_per_iter)
                    
                    if np.isnan(loss):
                        raise RuntimeError("Nan loss computed.")

            # Save model
            if iter % save_freq == 0:
                self.save(iter)
                valid_res = self.valid()
                self.logger.write_scalar(**valid_res)
            self.iter = iter

        print("\n[ Training Done ]")
        if iter in losses:
            print(f"  Training Loss: {np.mean(losses[iter])}\n")
        


    def _train_step(self, params, stat_data, traj_data=None):
        """
        Training step

        params (ndarray): [n_sim, n_param] array of parameters
        stat_data (list): [n_sim] list of static data
        traj_data (list): [n_sim] list of trajectories (default: None)
        """
        self.amortizer.train()
        if self.point_estimation:
            params_tensor = torch.FloatTensor(params).to(self.amortizer.device)
            loss = F.mse_loss(self.amortizer(stat_data, traj_data) * self.loss_weight_vector, 
                              params_tensor * self.loss_weight_vector)
        else:
            raise NotImplementedError("Density estimator not implemented yet")
            # z, log_det_J = self.amortizer(params, stat_data, traj_data)
            # loss = torch.mean(0.5 * torch.square(torch.norm(z, dim=-1)) - log_det_J)
        return self._optim_step(loss)
    

    def _optim_step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.amortizer.parameters():
            param.grad.data.clamp_(-self.clipping, self.clipping)
        self.optimizer.step()
        return loss.item()


    def save(self, iter, path=None):
        """
        Save model, optimizer, and scheduler with iteration number
        """
        if path is None:
            os.makedirs(self.model_path, exist_ok=True)
            ckpt_path = f"{self.model_path}/iter{iter:03d}.pt"
        else:
            os.makedirs(path, exist_ok=True)
            ckpt_path = path + f"iter{iter:03d}.pt"
        torch.save({
            "iteration": iter,
            "model_state_dict": self.amortizer.state_dict(),
            "optim_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }, ckpt_path)
        

    def load(self, model_name):
        """
        Load model, optimizer, and scheduler from the latest checkpoint
        """
        assert model_name is not None, "You must specify the model name to be loaded."
        import glob
        os.path.join(DIR_TO_DATA, f"data/{FOLDER.AMORTIZER3}/models/{self.name}/pts")
        ckpt_paths = glob.glob(os.path.join(DIR_TO_DATA, f"data/{FOLDER.AMORTIZER3}/models/{model_name}/pts/iter*.pt"))
        ckpt_paths.sort()
        ckpt_path = ckpt_paths[-1]

        self.name = model_name
        self.model_path = os.path.join(DIR_TO_DATA, f"data/{FOLDER.AMORTIZER3}/models/{self.name}/pts")
        self.board_path = os.path.join(DIR_TO_DATA, f"data/{FOLDER.AMORTIZER3}/models/{self.name}/board")
        self.result_path = os.path.join(DIR_TO_DATA, f"data/{FOLDER.AMORTIZER3}/models/{self.name}/results")

        ckpt = torch.load(ckpt_path, map_location=self.amortizer.device.type)
        self.amortizer.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optim_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        iter = ckpt["iteration"]
        self.scheduler.step(iter)

        print(f"[ amortizer - loaded checkpoint ] Model {model_name} - Iteration {iter}")

        self.iter = iter
    

    def valid(self,
        n_trial=np.array([16, 32, 64]),
        n_sample=300,   # Sample for distribution estimation
        verbose=True
    ):
        self.amortizer.eval()
        valid_res = dict()

        ### 1) Parameter recovery from simulated
        type_list = ["mode", "mean", "median"] if type(self.amortizer) is AmortizerForTrialData else ["mode"]

        start_t = time()
        
        for infer_type in type_list:
            rsq = list()
            for n in n_trial:
                (
                    sim_gt_params_main, 
                    sim_gt_params_sub1,
                    sim_gt_params_sub2,
                    sim_valid_data_stat,
                    sim_valid_data_traj
                ) = self.valid_dataset.sample(n)
                r2s = self.parameter_recovery(
                    valid_res,
                    sim_gt_params_main, 
                    sim_gt_params_sub1,
                    sim_gt_params_sub2,
                    sim_valid_data_stat,
                    sim_valid_data_traj,
                    n_sample,
                    n,
                    infer_type,
                    surfix="sim",
                )
                rsq.append(r2s)
            rsq = np.array(rsq)

            os.makedirs(f"{self.result_path}/iter{self.iter:03d}", exist_ok=True)
            pd.DataFrame(dict(zip(
                ["n", *self.main_target_param, *self.sub1_target_param, *self.sub2_target_param],
                [n_trial, *rsq.T]
            ))).to_csv(f"{self.result_path}/iter{self.iter:03d}/recovery_r2_{infer_type}.csv", index=False)

            ### 2) Parameter inference from user data
            if self.data_config.traj.ignore_traj:
                self.infer_user_parameter(n_sample, infer_type)

        if verbose:
            print(f"- parameter recovery (simulated) ({time() - start_t:.3f}s)")

        return valid_res
    


    def parameter_recovery(
        self,
        result_log,
        gt_main_params,
        gt_sub1_params,
        gt_sub2_params,
        valid_data_stat,
        valid_data_traj,
        n_sample,
        n_datasize,
        infer_type,
        surfix="",
    ):
        basedir = f"{self.result_path}/iter{self.iter:03d}/recovery_{infer_type}/" \
            if type(self.amortizer) is AmortizerForTrialData else f"{self.result_path}/iter{self.iter:03d}/"
        os.makedirs(basedir, exist_ok=True)

        # Note: all parameters are normalized: -1 ~ 1
        n_user = gt_main_params.shape[0]
        inferred_main_params, inferred_sub1_params, inferred_sub2_params = list(), list(), list()

        for user_i in range(n_user):
            stat_i = valid_data_stat[user_i]
            # traj_i = valid_data_traj[user_i]

            cond1_index = torch.arange(stat_i.shape[0]).repeat_interleave(stat_i.shape[1] * stat_i.shape[2])
            cond12_index = cond1_index * stat_i.shape[1] + torch.arange(stat_i.shape[1]).repeat_interleave(stat_i.shape[2]).tile(stat_i.shape[0])

            main_param_z, sub1_param_z, sub2_param_z = self.amortizer.infer(
                rearrange(stat_i, "s e ep d -> (s e ep) d"),
                None, 
                cond1_index=cond1_index,
                cond12_index=cond12_index,
            )

            main_param_z = self._clip_params(main_param_z)
            sub1_param_z = self._clip_params(sub1_param_z)
            sub2_param_z = self._clip_params(sub2_param_z)
            (
                main_param_w, 
                sub1_param_w,
                sub2_param_w
            ) = self.convert_param_z_to_w(
                main_outputs=main_param_z, 
                sub1_outputs=sub1_param_z,
                sub2_outputs=sub2_param_z,
            )
            inferred_main_params.append(main_param_w)
            inferred_sub1_params.append(sub1_param_w)
            inferred_sub2_params.append(sub2_param_w)

        inferred_main_params = np.array(inferred_main_params)   # (n_user, main_dim)
        inferred_sub1_params = np.array(inferred_sub1_params)   # (n_user, n_sub1, sub1_dim)
        inferred_sub2_params = np.array(inferred_sub2_params)   # (n_user, n_sub2, n_sub2, sub1_dim)

        gt_param_list = [self.convert_param_z_to_w(None, gtm, gts1, gts2) for gtm, gts1, gts2 in
                         zip(gt_main_params, gt_sub1_params, gt_sub2_params)]
        
        gt_main_params = np.array([gt[0] for gt in gt_param_list])   # (n_user, main_dim)
        gt_sub1_params = np.array([gt[1] for gt in gt_param_list])   # (n_user, n_sub1, sub1_dim)
        gt_sub2_params = np.array([gt[2] for gt in gt_param_list])   # (n_user, n_sub2, n_sub2, sub1_dim)
                

        pickle_save(
            f"{basedir}/trial{n_datasize:03d}.pkl",
            (
                gt_main_params, 
                gt_sub1_params, 
                gt_sub2_params,
                inferred_main_params, 
                inferred_sub1_params, 
                inferred_sub2_params
            )
        )
        
        # Recovery for main/sub separately
        rsq_list = list()
        
        # Main parameter
        fig, axs = figure_grid(1, len(self.main_target_param), size_ax=3)
        for i, p in enumerate(self.main_target_param):
            result = draw_r2_plot(
                ax=axs[i],
                xdata=gt_main_params[:,i],
                ydata=inferred_main_params[:,i],
                xlabel=f"True {PARAM_SYMBOL.get(p, p)}",
                ylabel=f"Inferred {PARAM_SYMBOL.get(p, p)}",
            )
            result_log[f"Parameter_Recovery/{infer_type}/r2_{n_datasize:03d}_{p}_{surfix}"] = result["R2"]
            rsq_list.append(result["R2"])
        figure_save(fig, f"{basedir}/trial{n_datasize:03d}_main.png")

        # Sub parameter 1
        fig, axs = figure_grid(1, len(self.sub1_target_param), size_ax=3)
        if len(self.sub1_target_param) == 1:
            axs = [axs]
        gt_sub1_params = rearrange(gt_sub1_params, "m s1 d -> (m s1) d")
        inferred_sub1_params = rearrange(inferred_sub1_params, "m s1 d -> (m s1) d")
        for i, p in enumerate(self.sub1_target_param):
            result = draw_r2_plot(
                ax=axs[i],
                xdata=gt_sub1_params[:,i],
                ydata=inferred_sub1_params[:,i],
                xlabel=f"True {PARAM_SYMBOL.get(p, p)}",
                ylabel=f"Inferred {PARAM_SYMBOL.get(p, p)}",
            )
            result_log[f"Parameter_Recovery/{infer_type}/r2_{n_datasize:03d}_{p}_{surfix}"] = result["R2"]
            rsq_list.append(result["R2"])
        figure_save(fig, f"{basedir}/trial{n_datasize:03d}_sub1.png")

        # Sub parameter 2
        fig, axs = figure_grid(1, len(self.sub2_target_param), size_ax=3)
        gt_sub2_params = rearrange(gt_sub2_params, "m s1 s2 d -> (m s1 s2) d")
        inferred_sub2_params = rearrange(inferred_sub2_params, "m s1 s2 d -> (m s1 s2) d")
        for i, p in enumerate(self.sub2_target_param):
            result = draw_r2_plot(
                ax=axs[i],
                xdata=gt_sub2_params[:,i],
                ydata=inferred_sub2_params[:,i],
                xlabel=f"True {PARAM_SYMBOL.get(p, p)}",
                ylabel=f"Inferred {PARAM_SYMBOL.get(p, p)}",
            )
            result_log[f"Parameter_Recovery/{infer_type}/r2_{n_datasize:03d}_{p}_{surfix}"] = result["R2"]
            rsq_list.append(result["R2"])
        figure_save(fig, f"{basedir}/trial{n_datasize:03d}_sub2.png")

        return rsq_list


    def infer_user_parameter(self, n_sample, infer_type):
        basedir = f"{self.result_path}/iter{self.iter:03d}/recovery_{infer_type}/" \
            if type(self.amortizer) is AmortizerForTrialData else f"{self.result_path}/iter{self.iter:03d}/"
        os.makedirs(basedir, exist_ok=True)

        # Inference
        infer_result = list()
        for user in tqdm(range(self.n_user), desc="User inference"):
            stat_true = self.user_dataset[(self.user_dataset["user"] == user)]
            stat_norm, traj_norm, gi1, gi2 = self._process_data(stat_true, None)
            gi1_torch = torch.from_numpy(gi1).to(self.amortizer.device)
            gi2_torch = torch.from_numpy(gi2).to(self.amortizer.device)

            main_z, sub1_z, sub2_z = self.amortizer.infer(
                stat_norm, traj_data=traj_norm,
                cond1_index=gi1_torch, cond12_index=gi2_torch
            )
            main_z = np.clip(main_z, -1, 1)
            sub1_z = np.clip(sub1_z, -1, 1)
            sub2_z = np.clip(sub2_z, -1, 1)
            main_w, sub1_w, sub2_w = self.convert_param_z_to_w(
                outputs=None, 
                main_outputs=main_z, 
                sub1_outputs=sub1_z,
                sub2_outputs=sub2_z
            )
            sub2_w = rearrange(sub2_w, "s1 s2 d -> (s1 s2) d")
            main_w = dict(zip(self.main_target_param, main_w))
            total_w = [
                {
                    **main_w,
                    **dict(zip(self.sub1_target_param, sub1_w[s // 8])),
                    **dict(zip(self.sub2_target_param, sub2_w[s])),
                    "tier": user // 10,
                    **dict(zip(["col_level", "sen_level", "spd_level", "rad_level"], [(s >> i) & 1 for i in range(4)][::-1]))
                } for s in range(self.n_cond12)
            ]
            infer_result += total_w
        infer_result = pd.DataFrame(infer_result)

        feature_order = ["tier", "sen_level", "col_level", "rad_level", "spd_level"]
        xtick_list = [
            ["Pro.", "Ama."],
            ["Habitual", "Default"],
            ["White", "Gray"],
            ["Small", "Large"],
            ["Stat.", "Moving"]
        ]
        xlabel_list = ["Group", "Sensitivity", "Color", "Radius", "Speed"]

        fig, axs = figure_grid(len(self.total_target_param), 5, size_ax=np.array([4, 2.2]) / 1.5)
        for (i, param), j in product(enumerate(self.total_target_param), range(5)):
            ax = axs[i][j]

            sns.lineplot(data=infer_result, x=feature_order[j], y=param, marker='s', color='k',
                        linewidth=1, markersize=7, 
                        errorbar=('ci', 95), err_style='bars', err_kws={'capsize': 3}, 
                        ax=ax)
            
            ax.set_xlabel("")
            ax.set_ylabel(f"Inferred {PARAM_SYMBOL[param]}")
            ax.set_xticks(np.arange(2), xtick_list[j])
            ax.grid(axis='y', alpha=0.5, linestyle='--')
        
        for j, l in enumerate(xlabel_list):
            axs[-1][j].set_xlabel(l)

        figure_save(fig, f"{basedir}/parameter_inference.png")

        
    def _clip_params(self, params):
        return np.clip(params, -1.0, 1.0)
    

    def _process_data(self, stat:pd.DataFrame, traj:Optional[List[pd.DataFrame]]=None):
        """
        stat := pandas.DataFrame with shape (n_trial, n_feature)
        traj := list of pandas.DataFrame with each of the shape (n_tct_length, n_feature), list length n_trial
        Select target features & normalize, return numpy array
        """
        if traj is not None:
            assert stat.shape[0] == len(traj)
        
        full_stat_column = list(stat.columns)
        stat_target_indices = [full_stat_column.index(feat) for feat in self.train_dataset.norm_config.stat.list]
        stat_norm = linear_normalize(stat.to_numpy()[:,stat_target_indices].astype(np.float32), 
                                     *self.stat_range.T, dtype=np.float32)

        if traj is not None:
            traj_norm = list()
            for _t in traj:
                # Downsampling
                old_timestamp = _t["timestamp"].to_numpy()
                new_timestamp = np.linspace(0, old_timestamp[-1], round(old_timestamp[-1] * self.train_dataset.sim_config.traj_downsample) + 1)
                new_traj = list()
                for key in self.train_dataset.norm_config.traj.list:
                    if key != FIELD.TIMEDIFF:
                        new_traj.append(np.interp(new_timestamp, old_timestamp, _t[key].to_numpy()))
                    else:
                        new_traj.append(np.insert(np.diff(new_timestamp), 0, 0))
                new_traj = np.array(new_traj).T
                traj_norm.append(linear_normalize(new_traj, *self.traj_range.T, dtype=np.float32))

            traj_norm = np.array(traj_norm, dtype=object)
            return stat_norm, traj_norm, stat[self.user_dataset_cond1_key].to_numpy(), stat[self.user_dataset_cond2_key].to_numpy()
        
        return stat_norm, None, stat[self.user_dataset_cond1_key].to_numpy(), stat[self.user_dataset_cond2_key].to_numpy()


    def convert_param_z_to_w(self, outputs=None, main_outputs=None, sub1_outputs=None, sub2_outputs=None):
        return self.train_dataset.simulator.convert_param_z_to_w(outputs=outputs, 
                                                                 main_outputs=main_outputs, 
                                                                 sub1_outputs=sub1_outputs,
                                                                 sub2_outputs=sub2_outputs)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Training option')

    parser.add_argument('--name', default="AnS2LvCondTest")

    parser.add_argument('--train_config', type=str, default='default_lv2_cond')
    parser.add_argument('--simul_config', type=str, default='default_lv2_cond')
    parser.add_argument('--data_config', type=str, default='default')

    parser.add_argument('--step_per_iter', type=int, default=2048)
    parser.add_argument('--batch_sz', type=int, default=64)
    parser.add_argument('--n_iter', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=10)

    parser.add_argument('--load_ckpt', type=bool, default=False)
    parser.add_argument('--load_model', default=None)

    args = parser.parse_args()

    trainer = AnSTwoLevelConditionalAmortizerTrainer(
        name=args.name,
        train_config=args.train_config,
        simul_config=args.simul_config,
        data_config=args.data_config,
    )

    if args.load_ckpt:
        trainer.load(args.load_model)
    
    trainer.train(
        n_iter=args.n_iter,
        step_per_iter=args.step_per_iter,
        batch_sz=args.batch_sz,
        save_freq=args.save_freq
    )