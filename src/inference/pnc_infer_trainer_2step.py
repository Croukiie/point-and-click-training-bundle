"""
Code modified by June-Seop Yoon

Trainer for amortized inference
Orignial code was written by Dr. Hee-seung Moon (https://github.com/hsmoon121/amortized-inference-hci)
"""

import os
import sys
from time import time
from copy import deepcopy
from pathlib import Path
from abc import ABC
from tqdm import tqdm
import numpy as np
import pandas as pd
from typing import List, Optional
from einops import rearrange

import torch
import torch.nn.functional as F

import seaborn as sns

from itertools import product

from ..config.config import INF
from ..config.constant import PARAM_SYMBOL, FIELD, NAME_ABBR, NAME_FULL, FOLDER
from .pnc_dataset_2step import (
    PnCConditionalAmortizerTrainingDataset,
    PnCConditionalAmortizerValidationDataset
)
from ..nets.amortizer import ConditionalRegressionForTrialData, ConditionalAmortizerForTrialData
from ..utils.myutils import get_compact_timestamp_str, save_dict_to_yaml, pickle_save
from ..utils.schedulers import CosAnnealWR
from ..utils.loggers import Logger
from ..utils.myutils import load_config
from ..utils.myplot import figure_grid, figure_save, draw_r2_plot
from ..utils.mymath import linear_normalize, np_groupby, kl_divergence

import warnings
warnings.filterwarnings("ignore", message=".*use_inf_as_na.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*is_categorical_dtype.*")


DIR_TO_DATA = Path(__file__).parent.parent.parent

class PnCConditionalAmortizerTrainer(ABC):
    def __init__(
        self,
        name=None,
        load_model=None,
        train_config='default_cond',         # Training hyperparameter
        simul_config='default_cond',         # Simulator model setting
        data_config='default',    # Data processing setting
        train_subparams_per_main=24,
        train_trials_per_subparam=64,
        validation_users=100,
        validation_subparams=24,
        validation_trials=100,
    ):
        if load_model is not None:
            name = load_model
            train_config = load_config(os.path.join(DIR_TO_DATA, f"data/{FOLDER.AMORTIZER2}/models/{name}/train_setting.yaml"))
            simul_config = load_config(os.path.join(DIR_TO_DATA, f"data/{FOLDER.AMORTIZER2}/models/{name}/simulator_setting.yaml"))
            data_config = load_config(os.path.join(DIR_TO_DATA, f"data/{FOLDER.AMORTIZER2}/models/{name}/data_normalization_setting.yaml"))

        # Initialize simulator, amortizer, and dataset
        self.iter = 0
        self.train_dataset = PnCConditionalAmortizerTrainingDataset(
            simulator_config=simul_config, 
            normalize_config=data_config,
            num_of_subparam_per_mainparam=train_subparams_per_main,
            num_of_epsiode_per_param=train_trials_per_subparam,
        )
        self.valid_dataset = PnCConditionalAmortizerValidationDataset(
            simulator_config=simul_config, 
            normalize_config=data_config,
            total_user=validation_users,
            total_cond_per_user=validation_subparams,
            trial_per_user_cond=validation_trials,
        )
        self.simulator = deepcopy(self.train_dataset.simulator.simulator)

        # List of target parameters
        self.total_target_param = self.train_dataset.simulator.total_target_param
        self.main_target_param = self.train_dataset.simulator.main_param_list
        self.sub_target_param = self.train_dataset.simulator.sub_param_list

        # Configuration setting
        self.train_config = INF.training[train_config] if isinstance(train_config, str) else train_config
        self.data_config = INF.normalize[data_config] if isinstance(data_config, str) else data_config
        if load_model is None:
            stat_size = len(self.data_config.stat.list)
            if "stat_mean" in self.data_config:
                if not self.data_config.stat_mean.ignore_statmean:
                    stat_size += len(self.data_config.stat_mean.list)
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

            self.train_config.amortizer.invertible_main.param_sz = len(self.main_target_param)
            self.train_config.amortizer.invertible_sub.param_sz = len(self.sub_target_param)
            self.train_config.amortizer.invertible_main.block.cond_sz = self.train_config.amortizer.trial_encoder.attention.out_sz
            self.train_config.amortizer.invertible_sub.block.cond_sz = \
                self.train_config.amortizer.trial_encoder.attention.out_sz + self.train_config.amortizer.posterior_encoder.attention.out_sz \
                if not concat_feature_vector else self.train_config.amortizer.trial_encoder.attention.out_sz * 2

            self.train_config.amortizer.linear_main.out_sz = len(self.main_target_param)
            self.train_config.amortizer.linear_main.in_sz = self.train_config.amortizer.trial_encoder.attention.out_sz
            self.train_config.amortizer.linear_sub.out_sz = len(self.sub_target_param)
            self.train_config.amortizer.linear_sub.in_sz = \
                self.train_config.amortizer.trial_encoder.attention.out_sz + len(self.main_target_param) \
                if not concat_feature_vector else self.train_config.amortizer.trial_encoder.attention.out_sz * 2

            # Amortizer name
        self.name = name if name is not None else f"{'PTE' if self.train_config.point_estimation else 'INN'}_{get_compact_timestamp_str()}"

        # User dataset
        user_dataset_path = os.path.join(
            DIR_TO_DATA, "data/prolific_data/stat/merged_summary.csv"
        )
        self.user_dataset = (
            pd.read_csv(user_dataset_path)
            if os.path.isfile(user_dataset_path)
            else None
        )
        self.n_user = 90
        self.n_cond = 24
        self.user_dataset_cond_key = "condition_id"

        # Warmup
        self.point_estimation = self.train_config.point_estimation
        amortizer_fn = ConditionalRegressionForTrialData if self.point_estimation else ConditionalAmortizerForTrialData 
        self.amortizer = amortizer_fn(config=self.train_config["amortizer"])

        self.lr = self.train_config["learning_rate"]
        self.lr_gamma = self.train_config["lr_gamma"]
        self.clipping = self.train_config["clipping"]
        self.optimizer = torch.optim.Adam(self.amortizer.parameters(), lr=1e-9)
        self.scheduler = CosAnnealWR(self.optimizer, T_0=10, T_mult=1, eta_max=self.lr, T_up=1, gamma=self.lr_gamma)

        # Loss weight setting
        if self.point_estimation:
            main_weights = np.asarray(
                self.train_config.amortizer.loss_weight.main_param,
                dtype=np.float32,
            ).reshape(-1)
            sub_weights = np.asarray(
                self.train_config.amortizer.loss_weight.sub_param,
                dtype=np.float32,
            ).reshape(-1)
            if main_weights.size == 1:
                main_weights = np.repeat(main_weights, len(self.main_target_param))
            if sub_weights.size == 1:
                sub_weights = np.repeat(sub_weights, len(self.sub_target_param))
            if main_weights.size != len(self.main_target_param):
                raise ValueError(
                    "Expected one main loss weight per shared parameter: "
                    f"got {main_weights.size} for {self.main_target_param}"
                )
            if sub_weights.size != len(self.sub_target_param):
                raise ValueError(
                    "Expected one sub loss weight per block-specific parameter: "
                    f"got {sub_weights.size} for {self.sub_target_param}"
                )
            self.loss_weight_vector = np.concatenate((
                main_weights,
                np.tile(
                    sub_weights,
                    self.train_dataset.num_of_subparam_per_mainparam,
                ),
            ))
            self.loss_weight_vector = self.loss_weight_vector / self.loss_weight_vector.mean()
            self.loss_weight_vector = torch.from_numpy(self.loss_weight_vector).sqrt().to(self.amortizer.device)
        else:
            self.loss_weight_vector = torch.tensor(self.train_config.amortizer.loss_weight.main_log_det_J_weight, dtype=torch.float32).to(self.amortizer.device)

        self.model_path = os.path.join(DIR_TO_DATA, f"data/{FOLDER.AMORTIZER2}/models/{self.name}/pts")
        self.board_path = os.path.join(DIR_TO_DATA, f"data/{FOLDER.AMORTIZER2}/models/{self.name}/board")
        self.result_path = os.path.join(DIR_TO_DATA, f"data/{FOLDER.AMORTIZER2}/models/{self.name}/results")

        if load_model is None:
            save_dict_to_yaml(self.train_config, os.path.join(DIR_TO_DATA, f"data/{FOLDER.AMORTIZER2}/models/{self.name}/train_setting.yaml"))
            save_dict_to_yaml(self.train_dataset.norm_config, os.path.join(DIR_TO_DATA, f"data/{FOLDER.AMORTIZER2}/models/{self.name}/data_normalization_setting.yaml"))
            save_dict_to_yaml(self.train_dataset.sim_config, os.path.join(DIR_TO_DATA, f"data/{FOLDER.AMORTIZER2}/models/{self.name}/simulator_setting.yaml"))
        
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
        n_trial=64,
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
            with tqdm(
                total=step_per_iter,
                desc=f" Iter {iter}/{n_iter}",
                file=sys.stdout,
                dynamic_ncols=True,
                mininterval=1.0,
            ) as progress:
                for step in range(step_per_iter):
                    batch_param, batch_stat, batch_traj, _ = self.train_dataset.sample(
                        batch_sz=batch_sz,
                        subparam_per_mainparam=None,
                        sim_per_param=n_trial,
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
        self.amortizer.train()
        if self.point_estimation:
            params_tensor = torch.FloatTensor(params).to(self.amortizer.device)
            loss = F.mse_loss(self.amortizer(stat_data, traj_data) * self.loss_weight_vector, 
                              params_tensor * self.loss_weight_vector)
        else:
            main_z, main_log_det_J, sub_z, sub_log_det_J = self.amortizer(params, stat_data, traj_data)
            losses = torch.empty(1 + sub_z.shape[0]).to(self.amortizer.device)  # 1 (main) + n_cond (sub)
            losses[0] = torch.mean(0.5 * torch.square(torch.norm(main_z, dim=-1)) - main_log_det_J) * self.loss_weight_vector
            for i, (_sub_z, _sub_J) in enumerate(zip(sub_z, sub_log_det_J)):
                losses[i + 1] = torch.mean(0.5 * torch.square(torch.norm(_sub_z, dim=-1)) - _sub_J)
            loss = torch.mean(losses)
        return self._optim_step(loss)
    

    def _optim_step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        if np.isfinite(self.clipping) and self.clipping > 0:
            torch.nn.utils.clip_grad_norm_(
                self.amortizer.parameters(), max_norm=float(self.clipping)
            )
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
        os.path.join(DIR_TO_DATA, f"data/{FOLDER.AMORTIZER2}/models/{self.name}/pts")
        ckpt_paths = glob.glob(os.path.join(DIR_TO_DATA, f"data/{FOLDER.AMORTIZER2}/models/{model_name}/pts/iter*.pt"))
        ckpt_paths.sort()
        ckpt_path = ckpt_paths[-1]

        self.name = model_name
        self.model_path = os.path.join(DIR_TO_DATA, f"data/{FOLDER.AMORTIZER2}/models/{self.name}/pts")
        self.board_path = os.path.join(DIR_TO_DATA, f"data/{FOLDER.AMORTIZER2}/models/{self.name}/board")
        self.result_path = os.path.join(DIR_TO_DATA, f"data/{FOLDER.AMORTIZER2}/models/{self.name}/results")

        ckpt = torch.load(ckpt_path, map_location=self.amortizer.device.type)
        self.amortizer.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optim_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        iter = ckpt["iteration"]
        self.scheduler.step(iter)

        print(f"[ amortizer - loaded checkpoint ] Model {model_name} - Iteration {iter}")

        self.iter = iter
    

    def valid(self,
        n_trial=None,
        n_sample=300,   # Sample for distribution estimation
        verbose=True
    ):
        self.amortizer.eval()
        valid_res = dict()

        if n_trial is None:
            available = int(self.valid_dataset.trial_per_user_cond)
            n_trial = np.array(
                [value for value in (16, 32, 64) if value <= available]
                or [available]
            )
        else:
            n_trial = np.asarray(n_trial, dtype=int)

        ### 1) Parameter recovery from simulated
        type_list = ["mode", "mean", "median"] if \
            type(self.amortizer) is ConditionalAmortizerForTrialData else ["mode"]

        start_t = time()
        
        for infer_type in type_list:
            rsq = list()
            for n in n_trial:
                (
                    sim_gt_params_main, 
                    sim_gt_params_sub,
                    sim_valid_data_stat,
                    sim_valid_data_traj,
                    _
                ) = self.valid_dataset.sample(n)
                r2s = self.parameter_recovery(
                    valid_res,
                    sim_gt_params_main, 
                    sim_gt_params_sub,
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
                ["n", *self.main_target_param, *self.sub_target_param],
                [n_trial, *rsq.T]
            ))).to_csv(f"{self.result_path}/iter{self.iter:03d}/recovery_r2_{infer_type}.csv", index=False)

            ### 2) Parameter inference from user data
            if self.data_config.traj.ignore_traj and self.user_dataset is not None:
                self.infer_user_parameter(n_sample, infer_type)

        if verbose:
            print(f"- parameter recovery (simulated) ({time() - start_t:.3f}s)")

        return valid_res
        

    def parameter_recovery(
        self,
        result_log,
        gt_main_params,
        gt_sub_params,
        valid_data_stat,
        valid_data_traj,
        n_sample,
        n_datasize,
        infer_type,
        surfix="",
    ):
        basedir = f"{self.result_path}/iter{self.iter:03d}/recovery_{infer_type}/" \
            if isinstance(self.amortizer, ConditionalAmortizerForTrialData)  else f"{self.result_path}/iter{self.iter:03d}/"
        os.makedirs(basedir, exist_ok=True)

        # Note: all parameters are normalized: -1 ~ 1
        n_user = gt_main_params.shape[0]
        inferred_main_params, inferred_sub_params = list(), list()

        for user_i in range(n_user):
            stat_i = valid_data_stat[user_i]
            traj_i = valid_data_traj[user_i]
            main_param_z, sub_param_z = self.amortizer.infer(
                rearrange(stat_i, "s e d -> (s e) d"),
                rearrange(traj_i, "s e -> (s e)"), 
                n_sample=n_sample, 
                type=infer_type,
                grouping_index=torch.arange(stat_i.shape[0]).repeat_interleave(stat_i.shape[1]).to(self.amortizer.device)
            )
            main_param_z = self._clip_params(main_param_z)
            sub_param_z = self._clip_params(sub_param_z)
            main_param_w, sub_param_w = self.convert_param_z_to_w(main_outputs=main_param_z, sub_outputs=sub_param_z)
            inferred_main_params.append(main_param_w)
            inferred_sub_params.append(sub_param_w)

        inferred_main_params = np.array(inferred_main_params)   # (n_user, main_dim)
        inferred_sub_params = np.concatenate(inferred_sub_params) # (n_user * n_subparam, sub_dim)
                
        n_subparams = gt_sub_params.shape[1]
        gt_main_params, gt_sub_params = self.convert_param_z_to_w(
            main_outputs=gt_main_params,
            sub_outputs=rearrange(gt_sub_params, 'n s d -> (n s) d'),
        )
        
        pickle_save(
            f"{basedir}/trial{n_datasize:03d}.pkl",
            (gt_main_params, gt_sub_params, inferred_main_params, inferred_sub_params)
        )
        
        # Recovery for main/sub separately
        rsq_list = list()
        
        fig, axs = figure_grid(1, len(self.main_target_param), size_ax=3)
        axs = np.atleast_1d(axs)
        for i, p in enumerate(self.main_target_param):
            result = draw_r2_plot(
                ax=axs[i],
                xdata=gt_main_params[:,i],
                ydata=inferred_main_params[:,i],
                xlabel=f"True {PARAM_SYMBOL.get(p, p)}",
                ylabel=f"Inferred {PARAM_SYMBOL.get(p, p)}",
            )
            result_log[f"Parameter_Recovery/{infer_type}/r2_{n_datasize:03d}_{p}_{surfix}"] = result["R2"]
            for metric in ("PearsonR", "Slope", "MAE", "RMSE"):
                result_log[
                    f"Parameter_Recovery/{infer_type}/{metric.lower()}_"
                    f"{n_datasize:03d}_{p}_{surfix}"
                ] = result[metric]
            rsq_list.append(result["R2"])
        fig.suptitle(f"Num. of trials: {n_datasize}", fontsize=12)
        figure_save(fig, f"{basedir}/trial{n_datasize:03d}_main.png")

        fig, axs = figure_grid(1, len(self.sub_target_param), size_ax=3)
        axs = np.atleast_1d(axs)
        for i, p in enumerate(self.sub_target_param):
            result = draw_r2_plot(
                ax=axs[i],
                xdata=gt_sub_params[:,i],
                ydata=inferred_sub_params[:,i],
                xlabel=f"True {PARAM_SYMBOL.get(p, p)}",
                ylabel=f"Inferred {PARAM_SYMBOL.get(p, p)}",
            )
            result_log[f"Parameter_Recovery/{infer_type}/r2_{n_datasize:03d}_{p}_{surfix}"] = result["R2"]
            for metric in ("PearsonR", "Slope", "MAE", "RMSE"):
                result_log[
                    f"Parameter_Recovery/{infer_type}/{metric.lower()}_"
                    f"{n_datasize:03d}_{p}_{surfix}"
                ] = result[metric]
            rsq_list.append(result["R2"])
        fig.suptitle(f"Num. of trials: {n_datasize}", fontsize=12)
        figure_save(fig, f"{basedir}/trial{n_datasize:03d}_sub.png")

        if self.sub_target_param == ["cpi"]:
            true_cpi = gt_sub_params[:, 0].reshape(n_user, n_subparams)
            inferred_cpi = inferred_sub_params[:, 0].reshape(n_user, n_subparams)
            exact_order = np.all(
                np.argsort(true_cpi, axis=1) == np.argsort(inferred_cpi, axis=1),
                axis=1,
            )
            pair_matches = []
            rank_correlations = []
            for true_values, inferred_values in zip(true_cpi, inferred_cpi):
                matches = []
                for left in range(n_subparams):
                    for right in range(left + 1, n_subparams):
                        matches.append(
                            np.sign(true_values[left] - true_values[right])
                            == np.sign(inferred_values[left] - inferred_values[right])
                        )
                pair_matches.append(float(np.mean(matches)))
                true_rank = np.argsort(np.argsort(true_values))
                inferred_rank = np.argsort(np.argsort(inferred_values))
                rank_correlations.append(
                    float(np.corrcoef(true_rank, inferred_rank)[0, 1])
                )

            ordering = pd.DataFrame({
                "simulated_participant": np.repeat(np.arange(n_user), n_subparams),
                "anonymous_block": np.tile(np.arange(n_subparams), n_user),
                "true_cpi": true_cpi.reshape(-1),
                "inferred_cpi": inferred_cpi.reshape(-1),
                "participant_exact_order": np.repeat(exact_order, n_subparams),
                "participant_pairwise_order_accuracy": np.repeat(
                    pair_matches, n_subparams
                ),
                "participant_spearman_order": np.repeat(
                    rank_correlations, n_subparams
                ),
            })
            ordering.to_csv(
                f"{basedir}/trial{n_datasize:03d}_cpi_ordering.csv",
                index=False,
            )
            result_log[
                f"Parameter_Recovery/{infer_type}/cpi_exact_order_{n_datasize:03d}_{surfix}"
            ] = float(np.mean(exact_order))
            result_log[
                f"Parameter_Recovery/{infer_type}/cpi_pairwise_order_{n_datasize:03d}_{surfix}"
            ] = float(np.mean(pair_matches))

        return rsq_list
    

    def infer_user_parameter(self, n_sample, infer_type):
        basedir = f"{self.result_path}/iter{self.iter:03d}/recovery_{infer_type}/" \
            if isinstance(self.amortizer, ConditionalAmortizerForTrialData)  else f"{self.result_path}/iter{self.iter:03d}/"
        os.makedirs(basedir, exist_ok=True)

        # Inference
        infer_result = list()
        for user in tqdm(range(self.n_user), desc="User inference"):
            stat_true = self.user_dataset[(self.user_dataset["user"] == user)]
            stat_norm, traj_norm, gi = self._process_data(stat_true, None)
            gi_torch = torch.from_numpy(gi).to(self.amortizer.device)

            main_z, sub_z = self.amortizer.infer(stat_norm, traj_norm, gi_torch, n_sample, infer_type)
            main_w, sub_w = self.convert_param_z_to_w(outputs=None, main_outputs=main_z, sub_outputs=sub_z)
            main_w = dict(zip(self.main_target_param, main_w))
            total_w = [{**main_w, **dict(zip(self.sub_target_param, sub_w[s])), **{
                "stake_level": s // 6,
                "period_level": (s % 6) // 2,
                "difficulty_level": (s % 6) % 2
            }} for s in range(self.n_cond)]

            infer_result += total_w
        infer_result = pd.DataFrame(infer_result)

        fig, axs = figure_grid(len(self.total_target_param), 3, size_ax=np.array([4, 2.2]) / 1.5)
        for i, param in enumerate(self.total_target_param):
            ax = axs[i][0]

            sns.lineplot(data=infer_result, x="stake_level", y=param, marker='s', color='k',
                        linewidth=1, markersize=7, 
                        errorbar=('ci', 95), err_style='bars', err_kws={'capsize': 3}, 
                        ax=ax)
            
            ax.set_xlabel("")
            ax.set_ylabel(f"Inferred {PARAM_SYMBOL[param]}")
            ax.set_xticks(np.arange(4), [0, 0.1, 1, 10])
            ax.grid(axis='y', alpha=0.5, linestyle='--')

            ax = axs[i][1]

            sns.lineplot(data=infer_result, x="period_level", y=param, marker='s', color='k',
                        linewidth=1, markersize=7, 
                        errorbar=('ci', 95), err_style='bars', err_kws={'capsize': 3}, 
                        ax=ax)
            
            ax.set_xlabel("")
            ax.set_ylabel(f"Inferred {PARAM_SYMBOL[param]}")
            ax.set_xticks(np.arange(3), ["Early", "Middle", "Late"])
            ax.grid(axis='y', alpha=0.5, linestyle='--')

            ax = axs[i][2]

            sns.lineplot(data=infer_result, x="difficulty_level", y=param, marker='s', color='k',
                        linewidth=1, markersize=7, 
                        errorbar=('ci', 95), err_style='bars', err_kws={'capsize': 3}, 
                        ax=ax)
            
            ax.set_xlabel("")
            ax.set_ylabel(f"Inferred {PARAM_SYMBOL[param]}")
            ax.set_xticks(np.arange(2), ["Easy", "Hard"])
            ax.grid(axis='y', alpha=0.5, linestyle='--')
        
        axs[-1][0].set_xlabel("Stake")
        axs[-1][1].set_xlabel("Period")
        axs[-1][2].set_xlabel("Difficulty")

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
 
        if "stat_mean" in self.train_dataset.norm_config:
            stat_mean_indices = [self.train_dataset.norm_config.stat.list.index(key) for key in self.train_dataset.norm_config.stat_mean.list]
            if not self.train_dataset.norm_config.stat_mean.ignore_statmean:
                for i in stat_mean_indices:
                    val = stat_norm[:,i]
                    mean_val = np.mean(val)
                    new_column = np.full((stat_norm.shape[0], 1), mean_val)
                    stat_norm = np.hstack((stat_norm, new_column))

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
            return stat_norm, traj_norm, stat[self.user_dataset_cond_key].to_numpy()
        
        return stat_norm, None, stat[self.user_dataset_cond_key].to_numpy()


    def convert_param_z_to_w(self, outputs=None, main_outputs=None, sub_outputs=None):
        return self.train_dataset.simulator.convert_param_z_to_w(outputs=outputs, 
                                                                 main_outputs=main_outputs, 
                                                                 sub_outputs=sub_outputs)

    def to_param_dict(self, main_param, sub_params):
        return self.train_dataset.simulator.to_param_dict(main_param, sub_params)
    


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Training option')

    parser.add_argument('--name', default=None)

    parser.add_argument('--train_config', type=str, default='default_cond')
    parser.add_argument('--simul_config', type=str, default='default_cond')
    parser.add_argument('--data_config', type=str, default='default')

    parser.add_argument('--step_per_iter', type=int, default=50)
    parser.add_argument('--batch_sz', type=int, default=64)
    parser.add_argument('--n_trial', type=int, default=64)
    parser.add_argument('--save_freq', type=int, default=2)
    parser.add_argument('--n_iter', type=int, default=6)

    parser.add_argument('--load_ckpt', type=bool, default=False)
    parser.add_argument('--load_model', default=None)

    args = parser.parse_args()

    trainer = PnCConditionalAmortizerTrainer(
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
        n_trial=args.n_trial,
        save_freq=args.save_freq
    )
