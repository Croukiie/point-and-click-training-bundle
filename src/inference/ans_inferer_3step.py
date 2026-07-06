import numpy as np
import pandas as pd
from pathlib import Path
import os
import glob
from box import Box
import torch
import time
from typing import List, Optional
from einops import rearrange

from ..utils.myutils import load_config
from ..nets.amortizer import TwoLevelConditionalRegressionForTrialData
from .ans_dataset_3step import AnSSimulatorForTwoLevelConditionalAmortizer
from ..utils.mymath import linear_normalize
from ..config_ans.constant import FIELD, FOLDER

DIR_TO_DATA = Path(__file__).parent.parent.parent

class AnSTwoLevelConditionalInferer:
    def __init__(self, model_name, ckpt=None, verbose=True):
        # Configuration setting
        self.name = model_name
        self.root_path = os.path.join(DIR_TO_DATA, f"data/{FOLDER.AMORTIZER3}/models/{model_name}")

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
        
        if self.train_config.point_estimation:
            amortizer_fn = TwoLevelConditionalRegressionForTrialData
        else:
            raise NotImplementedError("Density estimation for 3-step cond. amortizer not implemented!")

        self.amortizer = amortizer_fn(config=self.amortizer_config)
        ckpt = torch.load(model_path, map_location=self.amortizer.device.type, weights_only=True)
        self.amortizer.load_state_dict(ckpt["model_state_dict"])
        self.amortizer.eval()
        iter = ckpt["iteration"]

        if verbose:
            print(f"[ amortizer - loaded model ] {model_name} - Iteration {iter}")
        
        # Simulator setting
        self.simulator = AnSSimulatorForTwoLevelConditionalAmortizer(
            model_name = self.simulator_config.model_name,
            ckpt = self.simulator_config.ckpt,
            downsample_rate = self.simulator_config.traj_downsample,
            main_param_list = self.simulator_config.main_param,
            sub1_param_list = self.simulator_config.sub1_param,
            sub2_param_list = self.simulator_config.sub2_param,
            repeat_task_cond = self.simulator_config.repeat_task_cond
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

    
    def infer(self, stat, traj=None, cond1_index=None, cond12_index=None, return_df=True):
        assert cond1_index is not None and cond12_index is not None

        cond1_idx_torch = torch.from_numpy(cond1_index).to(self.amortizer.device)
        cond12_idx_torch = torch.from_numpy(cond12_index).to(self.amortizer.device)

        main_z, sub1_z, sub2_z = self.amortizer.infer(
            stat, traj_data=traj,
            cond1_index=cond1_idx_torch, cond12_index=cond12_idx_torch
        )
        main_z = np.clip(main_z, -1, 1)
        sub1_z = np.clip(sub1_z, -1, 1)
        sub2_z = np.clip(sub2_z, -1, 1)
        main_w, sub1_w, sub2_w = self.simulator.convert_param_z_to_w(
            outputs=None, 
            main_outputs=main_z, 
            sub1_outputs=sub1_z,
            sub2_outputs=sub2_z
        )

        # param_z_list = self.simulator.to_param_dict(main_z, sub1_z, sub2_z)
        # param_w_list = self.simulator.to_param_dict(main_w, sub1_w, sub2_w)

        gi1 = np.sort(np.unique(cond1_index))
        gi2 = np.sort(np.unique(cond12_index))

        main_z = dict(zip(self.simulator.main_param_list, main_z))
        sub1_z = {int(i): dict(zip(self.simulator.sub1_param_list, sub1_z[i])) for i in gi1}
        sub2_z = rearrange(sub2_z, "s1 s2 d -> (s1 s2) d")
        sub2_z = {int(i): dict(zip(self.simulator.sub2_param_list, sub2_z[i])) for i in gi2}

        main_w = dict(zip(self.simulator.main_param_list, main_w))
        sub1_w = {int(i): dict(zip(self.simulator.sub1_param_list, sub1_w[i])) for i in gi1}
        sub2_w = rearrange(sub2_w, "s1 s2 d -> (s1 s2) d")
        sub2_w = {int(i): dict(zip(self.simulator.sub2_param_list, sub2_w[i])) for i in gi2}

        param_z_list = [{**main_z, **sub1_z[g1], **sub2_z[g2]} for g1, g2 in zip(cond1_index, cond12_index)]
        param_w_list = [{**main_w, **sub1_w[g1], **sub2_w[g2]} for g1, g2 in zip(cond1_index, cond12_index)]

        if return_df:
            param_z_list, param_w_list = pd.DataFrame(param_z_list), pd.DataFrame(param_w_list)

        return param_z_list, param_w_list, main_w, sub1_w, sub2_w
    

    def simulate(self, total_param_z_list, task_list, user_stat_list, resimulate_max_num=0, verbose=True, downsample=None):
        simulator = self.simulator.simulator
        simulator.clear_simulation()

        tpzl = total_param_z_list.to_dict(orient='records') if isinstance(total_param_z_list, pd.DataFrame) \
            else total_param_z_list

        simulator.simulate(param_list=tpzl,
                           task_list=task_list, 
                           user_stat_list=user_stat_list,
                           resimulate_max_num=resimulate_max_num, 
                           verbose=verbose)
        
        stat, traj = simulator.get_simulation_result(return_traj=True, downsample=downsample)

        return stat, traj

    
    def process_data(
        self, 
        stat:pd.DataFrame, 
        traj:Optional[List[pd.DataFrame]]=None,
        cond1_index_column="condition1_id",
        cond12_index_column="condition2_id"
    ):
        """
        stat := pandas.DataFrame with shape (n_trial, n_feature)
        traj := list of pandas.DataFrame with each of the shape (n_tct_length, n_feature), list length n_trial
        Select target features & normalize, return numpy array
        """
        if traj is not None:
            assert stat.shape[0] == len(traj)
        
        full_stat_column = list(stat.columns)
        stat_target_indices = [full_stat_column.index(feat) for feat in self.data_normalization.stat.list]
        stat_norm = linear_normalize(stat.to_numpy()[:,stat_target_indices].astype(np.float32), 
                                     *self.stat_range.T, dtype=np.float32)

        if traj is not None:
            traj_norm = list()
            for _t in traj:
                # Downsampling
                old_timestamp = _t["timestamp"].to_numpy()
                new_timestamp = np.linspace(0, old_timestamp[-1], round(old_timestamp[-1] * self.simulator_config.traj_downsample) + 1)
                new_traj = list()
                for key in self.data_normalization.traj.list:
                    if key != FIELD.TIMEDIFF:
                        new_traj.append(np.interp(new_timestamp, old_timestamp, _t[key].to_numpy()))
                    else:
                        new_traj.append(np.insert(np.diff(new_timestamp), 0, 0))
                new_traj = np.array(new_traj).T
                traj_norm.append(linear_normalize(new_traj, *self.traj_range.T, dtype=np.float32))

            traj_norm = np.array(traj_norm, dtype=object)
            return stat_norm, traj_norm, stat[cond1_index_column].to_numpy(), stat[cond12_index_column].to_numpy()
        
        return stat_norm, None, stat[cond1_index_column].to_numpy(), stat[cond12_index_column].to_numpy()
    

    def extract_task_and_user_stat(self, summary_data:pd.DataFrame):
        return self.simulator.simulator.extract_task_and_user_stat(summary_data)




if __name__ == "__main__":
    from ..datamanagekit.ijhcs25_data import ExperimentIJHCS
    amortizer_name = "PTE_25210DJMY"

    inferer = AnSTwoLevelConditionalInferer(amortizer_name, ckpt=2)
    dataset = ExperimentIJHCS()

    s, _ = dataset.load_dataset(user=0)
    tl, ul = inferer.extract_task_and_user_stat(s)
    sn, _, c1, c2 = inferer.process_data(s)

    pz, pw, w1, w2, w3 = inferer.infer(sn, None, c1, c2)
    print(w1)
    print(w2)
    print(w3)
    # stat, _ = inferer.simulate(pz, tl, ul, verbose=True)

    # stat.to_csv("simul.csv", index=False)
