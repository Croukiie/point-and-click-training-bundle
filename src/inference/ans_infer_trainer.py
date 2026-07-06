"""
Code written by June-Seop Yoon

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
import seaborn as sns

from itertools import product

import torch
import torch.nn.functional as F

from ..config_ans.config import INF
from ..config_ans.constant import PARAM_SYMBOL, FIELD
from ..inference.ans_dataset import (
    AnSAmortizerTrainingDataset,
    AnSAmortizerValidationDataset
)
from ..nets.amortizer import AmortizerForTrialData, RegressionForTrialData
from ..utils.myutils import get_compact_timestamp_str, save_dict_to_yaml, pickle_save
from ..utils.schedulers import CosAnnealWR
from ..utils.loggers import Logger
from ..utils.myutils import load_config
from ..utils.myplot import figure_grid, figure_save, draw_r2_plot
from ..utils.mymath import linear_normalize, np_groupby, kl_divergence, closest_factor_pair

import warnings
warnings.filterwarnings("ignore", message=".*use_inf_as_na.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*is_categorical_dtype.*")


DIR_TO_DATA = Path(__file__).parent.parent.parent

class AnSAmortizerTrainer(ABC):
    def __init__(
        self,
        name=None,
        load_model=None,
        train_config='default',         # Training hyperparameter
        simul_config='default',         # Simulator model setting
        data_config='default',    # Data processing setting
    ):
        if load_model is not None:
            name = load_model
            train_config = load_config(os.path.join(DIR_TO_DATA, f"data/amortizer_ans/models/{name}/train_setting.yaml"))
            simul_config = load_config(os.path.join(DIR_TO_DATA, f"data/amortizer_ans/models/{name}/simulator_setting.yaml"))
            data_config = load_config(os.path.join(DIR_TO_DATA, f"data/amortizer_ans/models/{name}/data_normalization_setting.yaml"))

        # Initialize simulator, amortizer, and dataset
        self.iter = 0
        self.train_dataset = AnSAmortizerTrainingDataset(
            simulator_config=simul_config, 
            normalize_config=data_config
        )
        self.valid_dataset = AnSAmortizerValidationDataset(
            simulator_config=simul_config, 
            normalize_config=data_config
        )
        self.simulator = deepcopy(self.train_dataset.simulator.simulator)

        # List of target parameters
        self.target_param = self.train_dataset.simulator.total_target_param

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
            self.train_config.amortizer.invertible.param_sz = len(self.target_param)
            self.train_config.amortizer.invertible.block.cond_sz = self.train_config.amortizer.trial_encoder.attention.out_sz
            self.train_config.amortizer.linear.out_sz = len(self.target_param)
            self.train_config.amortizer.linear.in_sz = self.train_config.amortizer.trial_encoder.attention.out_sz

        # Amortizer name
        self.name = name if name is not None else f"{'PTE' if self.train_config.point_estimation else 'INN'}_{get_compact_timestamp_str()}"

        # User dataset
        self.user_dataset = pd.read_csv(os.path.join(DIR_TO_DATA, "data/ijhcs25_data/summary/general_summary.csv"))
        self.n_user = 20
        self.n_cond = 16
        self.user_dataset_cond_key = "condition2_id"

        # Warmup
        self.point_estimation = self.train_config.point_estimation
        amortizer_fn = RegressionForTrialData if self.point_estimation else AmortizerForTrialData
        self.amortizer = amortizer_fn(config=self.train_config["amortizer"])

        self.lr = self.train_config["learning_rate"]
        self.lr_gamma = self.train_config["lr_gamma"]
        self.clipping = self.train_config["clipping"]
        self.optimizer = torch.optim.Adam(self.amortizer.parameters(), lr=1e-9)
        self.scheduler = CosAnnealWR(self.optimizer, T_0=10, T_mult=1, eta_max=self.lr, T_up=1, gamma=self.lr_gamma)

        self.loss_weight_vector = np.array(self.train_config.amortizer.loss_weight)
        self.loss_weight_vector = self.loss_weight_vector / self.loss_weight_vector.mean()
        self.loss_weight_vector = torch.from_numpy(self.loss_weight_vector).sqrt().to(self.amortizer.device)

        self.model_path = os.path.join(DIR_TO_DATA, f"data/amortizer_ans/models/{self.name}/pts")
        self.board_path = os.path.join(DIR_TO_DATA, f"data/amortizer_ans/models/{self.name}/board")
        self.result_path = os.path.join(DIR_TO_DATA, f"data/amortizer_ans/models/{self.name}/results")

        if load_model is None:
            save_dict_to_yaml(self.train_config, os.path.join(DIR_TO_DATA, f"data/amortizer_ans/models/{self.name}/train_setting.yaml"))
            save_dict_to_yaml(self.train_dataset.norm_config, os.path.join(DIR_TO_DATA, f"data/amortizer_ans/models/{self.name}/data_normalization_setting.yaml"))
            save_dict_to_yaml(self.train_dataset.sim_config, os.path.join(DIR_TO_DATA, f"data/amortizer_ans/models/{self.name}/simulator_setting.yaml"))
            
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
        n_iter=500,
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
            with tqdm(total=step_per_iter, desc=f" Iter {iter}") as progress:
                for step in range(step_per_iter):
                    batch_args = self.train_dataset.sample(
                        batch_sz=batch_sz, 
                        sim_per_param=n_trial
                    )

                    # Training step
                    loss = self._train_step(*batch_args)
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
                # self.logger.write_scalar(**valid_res)
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
            z, log_det_J = self.amortizer(params, stat_data, traj_data)
            loss = torch.mean(0.5 * torch.square(torch.norm(z, dim=-1)) - log_det_J)
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
        os.path.join(DIR_TO_DATA, f"data/amortizer_ans/models/{self.name}/pts")
        ckpt_paths = glob.glob(os.path.join(DIR_TO_DATA, f"data/amortizer_ans/models/{model_name}/pts/iter*.pt"))
        ckpt_paths.sort()
        ckpt_path = ckpt_paths[-1]

        self.name = model_name
        self.model_path = os.path.join(DIR_TO_DATA, f"data/amortizer_ans/models/{self.name}/pts")
        self.board_path = os.path.join(DIR_TO_DATA, f"data/amortizer_ans/models/{self.name}/board")
        self.result_path = os.path.join(DIR_TO_DATA, f"data/amortizer_ans/models/{self.name}/results")

        ckpt = torch.load(ckpt_path, map_location=self.amortizer.device.type)
        self.amortizer.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optim_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        iter = ckpt["iteration"]
        self.scheduler.step(iter)

        print(f"[ amortizer - loaded checkpoint ] Model {model_name} - Iteration {iter}")

        self.iter = iter
    

    def valid(self,
        n_trial=np.array([16, 32, 64, 128, 256]),
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
                sim_gt_params, sim_valid_data = self.valid_dataset.sample(n)
                r2s = self.parameter_recovery(
                    valid_res,
                    sim_gt_params,
                    sim_valid_data,
                    n_sample,
                    n,
                    infer_type,
                    surfix="sim",
                )
                rsq.append(r2s)
            rsq = np.array(rsq)

            os.makedirs(f"{self.result_path}/iter{self.iter:03d}", exist_ok=True)
            pd.DataFrame(dict(zip(
                ["n", *self.target_param],
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
        gt_params,
        valid_data,
        n_sample,
        n_datasize,
        infer_type,
        surfix="",
    ):
        basedir = f"{self.result_path}/iter{self.iter:03d}/recovery_{infer_type}/" \
            if type(self.amortizer) is AmortizerForTrialData else f"{self.result_path}/iter{self.iter:03d}/"
        os.makedirs(basedir, exist_ok=True)

        # Note: all parameters are normalized: -1 ~ 1
        n_param = gt_params.shape[0]
        inferred_params = list()

        for param_i in range(n_param):
            stat_i, traj_i = valid_data[param_i] # valid_data := list[(stat, traj)]
            param_z = self.amortizer.infer(
                stat_i, traj_i, 
                n_sample=n_sample, 
                type=infer_type
            )
            param_z = self._clip_params(param_z)
            p = self.convert_param_z_to_w(param_z)[0]
            inferred_params.append(p)
        inferred_params = np.array(inferred_params)
                
        gt_params = self.convert_param_z_to_w(gt_params)
        
        pickle_save(
            f"{basedir}/trial{n_datasize:03d}.pkl",
            (gt_params, inferred_params)
        )
        
        ax_r, ax_c = closest_factor_pair(len(self.target_param))
        fig, axs = figure_grid(ax_r, ax_c, size_ax=3)
        axs = axs.ravel()
        rsq_list = list()

        for i, p in enumerate(self.target_param):
            rsq = draw_r2_plot(
                ax=axs[i],
                xdata=gt_params[:,i],
                ydata=inferred_params[:,i],
                xlabel=f"True {PARAM_SYMBOL.get(p, p)}",
                ylabel=f"Inferred {PARAM_SYMBOL.get(p, p)}",
            )
            result_log[f"Parameter_Recovery/{infer_type}/r2_{n_datasize:03d}_{p}_{surfix}"] = rsq
            rsq_list.append(rsq)
        fig.suptitle(f"Num. of trials: {n_datasize}", fontsize=12)

        figure_save(fig, f"{basedir}/trial{n_datasize:03d}.png")

        return rsq_list
    

    def infer_user_parameter(self, n_sample, infer_type):
        basedir = f"{self.result_path}/iter{self.iter:03d}/recovery_{infer_type}/" \
            if type(self.amortizer) is AmortizerForTrialData else f"{self.result_path}/iter{self.iter:03d}/"
        os.makedirs(basedir, exist_ok=True)

        # Inference
        infer_result = list()
        for user, cond in tqdm(product(range(self.n_user), range(self.n_cond)), total=self.n_user*self.n_cond, desc="User inference"):
            stat_true = self.user_dataset[(self.user_dataset["user"] == user) & 
                                          (self.user_dataset[self.user_dataset_cond_key] == cond)]
            stat_norm, traj_norm = self._process_data(stat_true, None)
            param_z = self.amortizer.infer(stat_norm, traj_norm, n_sample, infer_type)
            param_z = self._clip_params(param_z)
            param_w = self.convert_param_z_to_w(param_z)[0]
            param_w = dict(zip(self.target_param, param_w))

            param_w["tier"] = user // 10
            (param_w["col_level"],
            param_w["sen_level"],
            param_w["spd_level"],
            param_w["rad_level"]) = [(cond >> i) & 1 for i in range(4)][::-1]

            infer_result.append(param_w)
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

        fig, axs = figure_grid(len(self.target_param), 5, size_ax=np.array([4, 2.2]) / 1.5)
        for (i, param), j in product(enumerate(self.target_param), range(5)):
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
        return np.clip(
            params,
            np.array([-1.] * len(self.target_param)),
            np.array([1.] * len(self.target_param))
        )


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
            return stat_norm, traj_norm
        
        return stat_norm, None

    def convert_param_z_to_w(self, outputs):
        return self.train_dataset.simulator.convert_param_z_to_w(outputs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Training option')

    parser.add_argument('--name', default=None)

    parser.add_argument('--train_config', type=str, default='default')
    parser.add_argument('--simul_config', type=str, default='default')
    parser.add_argument('--data_config', type=str, default='default')

    parser.add_argument('--step_per_iter', type=int, default=2048)
    parser.add_argument('--batch_sz', type=int, default=64)
    parser.add_argument('--n_trial', type=int, default=64)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--n_iter', type=int, default=500)

    parser.add_argument('--load_ckpt', type=bool, default=False)
    parser.add_argument('--load_model', default=None)

    args = parser.parse_args()

    trainer = AnSAmortizerTrainer(
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