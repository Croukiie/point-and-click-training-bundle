import numpy as np
import pandas as pd
from pathlib import Path
import os
import glob
from box import Box
import torch
import time
from einops import rearrange
from typing import List, Optional

from ..utils.myutils import load_config
from ..nets.amortizer import ConditionalAmortizerForTrialData, ConditionalRegressionForTrialData
from .pnc_dataset_2step import PnCSimulatorForConditionalAmortizer
from ..utils.mymath import linear_normalize
from ..config.constant import FIELD, FOLDER
from ..utils.trajectory_preprocessing import resample_feature_frame

DIR_TO_DATA = Path(__file__).parent.parent.parent

class PnCConditionalInferer:
    def __init__(self, model_name, ckpt=None, verbose=True):
        # Configuration setting
        self.name = model_name
        self.root_path = os.path.join(DIR_TO_DATA, f"data/{FOLDER.AMORTIZER2}/models/{model_name}")

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
        
        amortizer_fn = ConditionalRegressionForTrialData if self.train_config.point_estimation else ConditionalAmortizerForTrialData

        self.amortizer = amortizer_fn(config=self.amortizer_config)
        ckpt = torch.load(model_path, map_location=self.amortizer.device.type, weights_only=True)
        self.amortizer.load_state_dict(ckpt["model_state_dict"])
        self.amortizer.eval()
        iter = ckpt["iteration"]

        if verbose:
            print(f"[ amortizer - loaded model ] {model_name} - Iteration {iter}")
        
        # Simulator setting
        self.simulator = PnCSimulatorForConditionalAmortizer(
            model_name = self.simulator_config.model_name,
            ckpt = self.simulator_config.ckpt,
            downsample_rate = self.simulator_config.traj_downsample,
            simulation_option = self.simulator_config.simul_option,
            main_param_list = self.simulator_config.main_param,
            sub_param_list = self.simulator_config.sub_param
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

    
    def infer(self, stat, traj=None, grouping_index=None, n_sample=300, type='mean', return_df=False, return_samples=False):
        assert grouping_index is not None

        grouping_index = np.asarray(grouping_index, dtype=np.int64).copy()
        gi_torch = torch.from_numpy(grouping_index).to(self.amortizer.device)

        if return_samples:
            assert not self.train_config.point_estimation, "return_sample only works for density estimation!"
            main_z, sub_z, main_post_sampled, sub_post_sampled_list = \
                self.amortizer.infer(stat, traj_data=traj, n_sample=n_sample, type=type, grouping_index=gi_torch, return_samples=True)
            n_sample = self.amortizer_config.posterior_encoder.num_sample
            main_post_sampled = main_post_sampled.cpu().detach().numpy()
            sub_post_sampled_list = rearrange(sub_post_sampled_list, "t n d -> (t n) d")
            main_post_sampled, sub_post_sampled_list = self.simulator.convert_param_z_to_w(
                outputs=None, main_outputs=main_post_sampled, sub_outputs=sub_post_sampled_list)
            sub_post_sampled_list = rearrange(sub_post_sampled_list, "(t n) d -> t n d", n=n_sample)
        else:
            main_z, sub_z = self.amortizer.infer(stat, traj_data=traj, n_sample=n_sample, 
                                                 type=type, grouping_index=gi_torch)

        main_z = np.clip(main_z, -1, 1)
        sub_z = np.clip(sub_z, -1, 1)
        main_w, sub_w = self.simulator.convert_param_z_to_w(outputs=None, main_outputs=main_z, sub_outputs=sub_z)

        param_z_list = self.simulator.to_param_dict(main_z, sub_z)
        param_w_list = self.simulator.to_param_dict(main_w, sub_w)

        gi = np.sort(np.unique(grouping_index))
        main_w = {p: float(param_w_list[0][p]) for p in self.simulator.main_param_list}
        param_z_by_group = {
            group: param_z_list[position] for position, group in enumerate(gi)
        }
        param_w_by_group = {
            group: param_w_list[position] for position, group in enumerate(gi)
        }
        sub_w = {
            int(group): {
                p: float(param_w_by_group[group][p])
                for p in self.simulator.sub_param_list
            }
            for group in gi
        }

        param_z_list = [param_z_by_group[g] for g in grouping_index]
        param_w_list = [param_w_by_group[g] for g in grouping_index]

        if return_df:
            param_z_list, param_w_list = pd.DataFrame(param_z_list), pd.DataFrame(param_w_list)

        if return_samples:
            return param_z_list, param_w_list, main_w, sub_w, main_post_sampled, sub_post_sampled_list
        return param_z_list, param_w_list, main_w, sub_w
    

    def simulate(self, total_param_z_list, task_list, user_stat_list, resimulate_max_num=0, verbose=True, downsample=None):
        simulator = self.simulator.simulator
        simulator.clear_simulation()

        tpzl = total_param_z_list.to_dict(orient='records') if isinstance(total_param_z_list, pd.DataFrame) \
            else total_param_z_list

        user_param_list, sim_param_list = simulator.split_total_param_list(tpzl)
        user_param_list = [{"param_z": p} for p in user_param_list]
        sim_param_list = [{"param_z": p} for p in sim_param_list]

        simulator.simulate(user_param_list=user_param_list,
                           sim_param_list=sim_param_list,
                           task_list=task_list, 
                           user_stat_list=user_stat_list,
                           resimulate_max_num=resimulate_max_num, 
                           verbose=verbose)
        
        stat, traj = simulator.get_simulation_result(return_traj=True, downsample=downsample)

        return stat, traj

    
    def process_data(self, stat:pd.DataFrame, traj:Optional[List[pd.DataFrame]]=None, grouping_index_column="condition_id"):
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
            return stat_norm, traj_norm, stat[grouping_index_column].to_numpy()
        
        return stat_norm, None, stat[grouping_index_column].to_numpy()




if __name__ == "__main__":
    pass
