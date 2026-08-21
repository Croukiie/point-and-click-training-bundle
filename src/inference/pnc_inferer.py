import numpy as np
import pandas as pd
from pathlib import Path
import os
import glob
from box import Box
import torch
import time
from typing import List, Optional

from ..utils.myutils import load_config
from ..nets.amortizer import AmortizerForTrialData, RegressionForTrialData
from .pnc_dataset import PnCSimulatorForAmortizer
from ..utils.mymath import linear_normalize
from ..config.constant import FIELD
from ..utils.trajectory_preprocessing import resample_feature_frame

DIR_TO_DATA = Path(__file__).parent.parent.parent

class PnCInferer:
    def __init__(self, model_name, ckpt=None, verbose=True):
        # Configuration setting
        self.name = model_name
        self.root_path = os.path.join(DIR_TO_DATA, f"data/amortizer/models/{model_name}")

        self.data_normalization = load_config(f"{self.root_path}/data_normalization_setting.yaml")
        self.simulator_config = load_config(f"{self.root_path}/simulator_setting.yaml")
        self.train_config = load_config(f"{self.root_path}/train_setting.yaml")
        self.amortizer_config = self.train_config.amortizer
        
        # Amortizer setting
        if ckpt is None:
            model_list = glob.glob(f"{self.root_path}/pts/iter*.pt")
            model_list.sort()
            model_path = model_list[-1]
            ckpt = int(model_path[-6:-3])
        else:
            model_path = f"{self.root_path}/pts/iter{ckpt:03d}.pt"
        
        amortizer_fn = RegressionForTrialData if self.train_config.point_estimation else AmortizerForTrialData

        self.amortizer = amortizer_fn(config=self.amortizer_config)
        ckpt = torch.load(model_path, map_location=self.amortizer.device.type)
        self.amortizer.load_state_dict(ckpt["model_state_dict"])
        self.amortizer.eval()
        iter = ckpt["iteration"]

        if verbose:
            print(f"[ amortizer - loaded model ] {model_name} - Iteration {iter}")
        
        # Simulator setting
        self.simulator = PnCSimulatorForAmortizer(
            model_name = self.simulator_config.model_name,
            ckpt = self.simulator_config.ckpt,
            downsample_rate = self.simulator_config.traj_downsample,
            simulation_option = self.simulator_config.simul_option
        )

        # Data normalization setting
        self.stat_range = np.array([
            [self.data_normalization.stat.range[feat].min,
            self.data_normalization.stat.range[feat].max] for feat in self.data_normalization.stat.list
        ])
        self.traj_range = np.array([
            [self.data_normalization.traj.range[feat].min,
            self.data_normalization.traj.range[feat].max] for feat in self.data_normalization.traj.list
        ])

    
    def infer(self, stat, traj=None, n_sample=300, type='mean', return_z=True, return_w=False, return_dict=False, return_samples=False):
        if return_samples:
            assert not self.train_config.point_estimation, "return_sample only works for density estimation!"
            z, post_sampled = self.amortizer.infer(stat, traj_data=traj, n_sample=n_sample, type=type, return_samples=True)
            post_sampled = self.simulator.convert_param_z_to_w(post_sampled)

        z = self.amortizer.infer(stat, traj_data=traj, n_sample=n_sample, type=type)
        z = np.clip(z, -1, 1)
        w = self.simulator.convert_param_z_to_w(z)[0]

        if return_dict:
            z = dict(zip(self.simulator.total_target_param, z))
            w = dict(zip(self.simulator.total_target_param, w))

        if return_samples:
            if return_z and return_w:
                return z, w, post_sampled
            elif return_w:
                return w, post_sampled
            elif return_z:
                return z, post_sampled
            else:
                raise ValueError("Either return_z or reutnr_w must be True")

        if return_z and return_w:
            return z, w
        elif return_w:
            return w
        elif return_z:
            return z
        else:
            raise ValueError("Either return_z or reutnr_w must be True")
    

    def simulate(self, param_z, task_list, user_stat_list, resimulate_max_num=0, verbose=True, downsample=None):
        simulator = self.simulator.simulator
        simulator.clear_simulation()

        # user_param = {p: param_z[p] for p in self.simulator.user_target_param}
        # simul_param = {p: param_z[p] for p in self.simulator.simul_target_param} 
        # simulator.fix_user_param(param_z=user_param)
        # simulator.fix_simul_param(param_z=simul_param)

        simulator.fix_total_param(param_z=param_z)
        simulator.simulate(task_list=task_list, user_stat_list=user_stat_list,
                           resimulate_max_num=resimulate_max_num, verbose=verbose)
        
        stat, traj = simulator.get_simulation_result(return_traj=True, downsample=downsample)

        return stat, traj


    
    def process_data(self, stat:pd.DataFrame, traj:Optional[List[pd.DataFrame]]=None):
        """
        stat := pandas.DataFrame with shape (n_trial, n_feature)
        traj := list of pandas.DataFrame with each of the shape (n_tct_length, n_feature), list length n_trial
        Select target features & normalize, return numpy array
        """
        if traj is not None:
            assert stat.shape[0] == len(traj)
        
        full_stat_column = list(stat.columns)
        stat_target_indices = [full_stat_column.index(feat) for feat in self.data_normalization.stat.list]
        if "stat_mean" in self.data_normalization:
            stat_mean_indices = [self.data_normalization.stat.list.index(key) for key in self.data_normalization.stat_mean.list]
        stat_norm = linear_normalize(stat.to_numpy()[:,stat_target_indices].astype(np.float32), 
                                     *self.stat_range.T, dtype=np.float32)
        
        if "stat_mean" in self.data_normalization:
            if not self.data_normalization.stat_mean.ignore_statmean:
                for i in stat_mean_indices:
                    val = stat_norm[:,i]
                    mean_val = np.mean(val)
                    new_column = np.full((stat_norm.shape[0], 1), mean_val)
                    stat_norm = np.hstack((stat_norm, new_column))

        if traj is not None:
            traj_norm = list()
            for _t in traj:
                new_traj = resample_feature_frame(
                    _t,
                    self.data_normalization.traj.list,
                    self.simulator_config.traj_downsample,
                    timediff_key=FIELD.TIMEDIFF,
                )
                traj_norm.append(linear_normalize(new_traj, *self.traj_range.T, dtype=np.float32))

            traj_norm = np.array(traj_norm, dtype=object)
            return stat_norm, traj_norm
        
        return stat_norm




if __name__ == "__main__":
    from ..datamanagekit.prolific_data import PnCUserDatasetProlific
    amortizer_name = "PTE_24249DERL"

    inferer = PnCInferer(amortizer_name, ckpt=100)
    dataset = PnCUserDatasetProlific()

    s, t, *_ = dataset.load_dataset(user=19)
    s, t = inferer.process_data(s, t)
    print(s.shape)
    print(s[:,-2])
    print(s[:,-1])

    # for i, x in enumerate(t):
    #     if x.shape[0] > 95:
    #         print(i)
    # inferer.infer(s, t)
