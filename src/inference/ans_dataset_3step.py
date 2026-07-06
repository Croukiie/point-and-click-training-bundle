"""
Dataset generator/loader for amortized inference
Orignial code written by Hee-seung Moon (https://github.com/hsmoon121/amortized-inference-hci)

Code modified by June-Seop Yoon
"""

import os
from pathlib import Path
from joblib import Parallel, delayed
import psutil
import numpy as np
from tqdm import tqdm
from datetime import datetime
import glob, gc
from einops import rearrange

from ..config_ans.config import INF
from ..config_ans.constant import FOLDER, FIELD
from ..utils.myutils import get_compact_timestamp_str, npz_load, npz_save, save_dict_to_yaml, load_config
from ..utils.mymath import linear_denormalize, linear_normalize, log_denormalize
from ..inference.ans_dataset import AnSSimulatorForAmortizer, AnSAmortizerTrainingDataset, AnSAmortizerValidationDataset

DIR_TO_DATA = Path(__file__).parent.parent.parent


class AnSSimulatorForTwoLevelConditionalAmortizer(AnSSimulatorForAmortizer):
    def __init__(self, model_name, ckpt, downsample_rate,
                 main_param_list, sub1_param_list, sub2_param_list, repeat_task_cond,
                 task_cond=dict(), user_stat_cond=dict()):
        super().__init__(
            model_name=model_name,
            ckpt=ckpt,
            downsample_rate=downsample_rate,
            task_cond=task_cond,
            user_stat_cond=user_stat_cond
        )
        self.main_param_list = main_param_list
        self.sub1_param_list = sub1_param_list
        self.sub2_param_list = sub2_param_list

        assert set(self.total_target_param) == \
            set(self.main_param_list) | set(self.sub1_param_list) | set(self.sub2_param_list)

        self.main_param_indices = list()
        self.sub1_param_indices = list()
        self.sub2_param_indices = list()

        for param in self.total_target_param:
            if param in self.main_param_list:
                self.main_param_indices.append(self.total_target_param.index(param))
            elif param in self.sub1_param_list:
                self.sub1_param_indices.append(self.total_target_param.index(param))
            elif param in self.sub2_param_list:
                self.sub2_param_indices.append(self.total_target_param.index(param))
            else:
                raise ValueError("parameter configuration incorrect.")
        
        # Set task condition in advance, if post model-fitting use task condition for
        # dataset binning
        self.repeat_task_cond = repeat_task_cond
        if len(self.repeat_task_cond.keys()):
            assert all(len(self.repeat_task_cond[key]) == len(self.repeat_task_cond[list(self.repeat_task_cond.keys())[0]]) 
                       for key in self.repeat_task_cond)
            self.n_sub2_param = len(self.repeat_task_cond[list(self.repeat_task_cond.keys())[0]])
            self.repeat_task_cond = [{key: self.repeat_task_cond[key][i] for key in self.repeat_task_cond} for i in range(self.n_sub2_param)]
        else:
            self.n_sub2_param = None


    def simulate(
        self,
        n_main_param=1,     # Will be fixed - cognitive parameter
        n_sub1_param=2,      # Will be sampled per cognitive parameter set
        n_sub2_param=8,
        sim_per_param_set=1,    # No. of simulation per (parameter, task) condition,
        verbose=False,
    ):
        if self.n_sub2_param is not None:
            n_sub2_param = self.n_sub2_param
        else:
            self.repeat_task_cond = [dict() for _ in range(n_sub2_param)]

        # Change sampling method
        n_param = n_main_param * n_sub1_param * n_sub2_param

        main_param_z = np.array([self.simulator.env._z_sampler() for _ in range(n_main_param)])
        sub1_param_z = np.array([self.simulator.env._z_sampler() for _ in range(n_main_param * n_sub1_param)])
        param_z = np.array([self.simulator.env._z_sampler() for _ in range(n_param)])

        # Replace with main parameter sample result
        for param in self.main_param_list:
            param_z[:,self.total_target_param.index(param)] = \
                np.repeat(main_param_z[:,self.total_target_param.index(param)], n_sub1_param * n_sub2_param)

        # Replace with sub1 parameter sample result
        for param in self.sub1_param_list:
            param_z[:,self.total_target_param.index(param)] = \
                np.repeat(sub1_param_z[:,self.total_target_param.index(param)], n_sub2_param)

        # Not modified below
        stats, trajs = list(), list()
        for i in (tqdm(range(n_param)) if verbose else range(n_param)):
            user_z = param_z[i]
            self.simulator.fix_user_param(param_z=dict(zip(self.total_target_param, user_z)))
            task_list = self.game.sample_task_condition(sim_per_param_set, **self.repeat_task_cond[i%n_sub2_param])
            user_stat_list = self.simulator.sample_user_stat_conditions(sim_per_param_set, **self.user_stat_cond)
            self.simulator.simulate(task_list=task_list, user_stat_list=user_stat_list, verbose=False)
            stat_result, traj_result = self.simulator.get_simulation_result(return_traj=True, downsample=self.downsample_rate)
            self.simulator.clear_simulation()

            stats.append(stat_result)   # pd.DataFrame
            trajs.append(traj_result)   # list of pd.DataFrame
        
        return (
            param_z, # np.ndarray with shape (n_param, num of target param)
            stats,   # [pd.df (sim_per_param x n_beh_feat), pd.df, ...] 1D list
            trajs    # [[pd.df traj 1, ... pd.df traj sim_per_param], [ ... ], ..., [ ... ]] 2D nested list
        )


    def convert_param_z_to_w(self, outputs=None, main_outputs=None, sub1_outputs=None, sub2_outputs=None):
        if outputs is not None:
            return super().convert_param_z_to_w(outputs)
        
        assert main_outputs is not None and sub1_outputs is not None and sub2_outputs is not None, \
            "Specify all parameters!"
        
        assert len(main_outputs.shape) == 1 and len(sub1_outputs.shape) == 2 and len(sub2_outputs.shape) == 3

        main_in = np.zeros(len(self.total_target_param), dtype=np.float32)
        for i in range(len(self.main_param_list)):
            main_in[self.main_param_indices[i]] = main_outputs[i]
        main_out = super().convert_param_z_to_w(main_in)[0]
        main_out = main_out[self.main_param_indices]


        sub1_in = np.zeros((sub1_outputs.shape[0], len(self.total_target_param)), dtype=np.float32)
        for i in range(len(self.sub1_param_list)):
            sub1_in[:,self.sub1_param_indices[i]] = sub1_outputs[:,i]
        sub1_out = super().convert_param_z_to_w(sub1_in)
        sub1_out = sub1_out[:, self.sub1_param_indices]

        sub2_outputs_copy = rearrange(sub2_outputs, "s1 s2 dim -> (s1 s2) dim")
        sub2_in = np.zeros((sub2_outputs_copy.shape[0], len(self.total_target_param)), dtype=np.float32)
        for i in range(len(self.sub2_param_list)):
            sub2_in[:,self.sub2_param_indices[i]] = sub2_outputs_copy[:,i]
        sub2_out = super().convert_param_z_to_w(sub2_in)
        sub2_out = sub2_out[:, self.sub2_param_indices]
        sub2_out = rearrange(sub2_out, "(s1 s2) dim -> s1 s2 dim", s1=sub2_outputs.shape[0])
        
        return main_out, sub1_out, sub2_out
    

    def to_param_dict(self, main_param, sub1_params, sub2_params):
        raise NotImplementedError

        



class AnSTwoLevelConditionalAmortizerTrainingDataset(AnSAmortizerTrainingDataset):
    def __init__(
        self, 
        simulator_config="default_lv2_cond",
        normalize_config="default",
        num_of_sub1param=2,
        num_of_sub2param=8,
        num_of_epsiode_per_param=64,
        load_existing_data=True,
        verbose=True
    ):
        sim_cfg = INF["simulator"][simulator_config] if isinstance(simulator_config, str) else simulator_config
        
        for necessary_field in ["model_name", "ckpt", "traj_downsample", "main_param", "sub1_param", "sub2_param", "repeat_task_cond"]:
            assert necessary_field in sim_cfg, f"{necessary_field} missing in configuration!"

        self.simulator = AnSSimulatorForTwoLevelConditionalAmortizer(
            model_name=sim_cfg["model_name"], 
            ckpt=sim_cfg["ckpt"],
            downsample_rate=sim_cfg["traj_downsample"],
            main_param_list=sim_cfg["main_param"],
            sub1_param_list=sim_cfg["sub1_param"],
            sub2_param_list=sim_cfg["sub2_param"],
            repeat_task_cond=sim_cfg["repeat_task_cond"]
        )
        self.sim_config = sim_cfg
        self.norm_config = INF["normalize"][normalize_config] if isinstance(normalize_config, str) else normalize_config

        self.num_of_sub1param = num_of_sub1param
        self.num_of_sub2param = num_of_sub2param
        self.num_of_epsiode_per_param = num_of_epsiode_per_param

        self.simul_model = sim_cfg["model_name"]
        self.simul_model_ckpt = sim_cfg["ckpt"]
        self.simul_model_traj_downsample = sim_cfg["traj_downsample"]

        self.simul_main_param = sim_cfg["main_param"]
        self.simul_sub1_param = sim_cfg["sub1_param"]
        self.simul_sub2_param = sim_cfg["sub2_param"]

        self.main_param_indices = np.array(self.simulator.main_param_indices)
        self.sub1_param_indices = np.array(self.simulator.sub1_param_indices)
        self.sub2_param_indices = np.array(self.simulator.sub2_param_indices)

        self.param_setting_code = '_'.join([''.join(map(str, l)) for l in [self.main_param_indices, 
                                                                           self.sub1_param_indices, 
                                                                           self.sub2_param_indices]])
        basename = "_".join([
            self.sim_config.model_name,
            str(self.sim_config.ckpt),
            str(self.sim_config.traj_downsample),
            self.param_setting_code
        ])
        self.data_path = os.path.join(DIR_TO_DATA, f"data/{FOLDER.AMORTIZER3}/dataset/", basename)
        os.makedirs(self.data_path, exist_ok=True)

        if load_existing_data:
            self._get_dataset(verbose)
    
    
    def _get_dataset(self, verbose=False, do_not_load_traj=True, **kwargs):
        filename = "_".join([
            "tr", f"*mp",
            f"{self.num_of_sub1param}sp1",
            f"{self.num_of_sub2param}sp2",
            f"{self.num_of_epsiode_per_param}ep",
            f"*.npz"
        ])
        datafilelist = glob.glob(f"{self.data_path}/{filename}")

        if not len(datafilelist):
            raise FileNotFoundError("No data file found ... generate the dataset first.")

        n_main_param = 0
        n_simul = 0

        param, stat, traj = list(), list(), list()

        datafilelist.sort()
        for f in datafilelist if not verbose else tqdm(datafilelist, desc="Processing training data simulations ... "):
            ff = os.path.basename(f).split('_')
            n_main_param += int(ff[1][:-2])
            n_simul += int(ff[-2])

            data = npz_load(f)
            param.append(data["params"])

            # Feature order
            data_feat_order = load_config(f.replace('.npz', '.yaml'))
            stat_target_indices = [data_feat_order.stat.index(key) for key in self.norm_config.stat.list]
            traj_target_indices = [data_feat_order.traj.index(key) for key in self.norm_config.traj.list]

            stat_value_range = np.array([
                [self.norm_config.stat.range[key].min, 
                self.norm_config.stat.range[key].max] for key in self.norm_config.stat.list]
            )
            traj_value_range = np.array([
                [self.norm_config.traj.range[key].min, 
                self.norm_config.traj.range[key].max] for key in self.norm_config.traj.list]
            )

            # Normalize and re-organize
            def norm_stat(stat):    # stat := (n_trial x n_feature) np.ndarray
                return linear_normalize(stat[:,stat_target_indices], *stat_value_range.T, dtype=np.float32)
            
            def norm_traj(traj):    # traj := (n_length x n_feature) np.ndarray
                if "timestamp" in data_feat_order.traj:
                    arr = traj[:,data_feat_order.traj.index("timestamp")]
                    arr = np.insert(np.diff(arr), 0, 0)
                    traj[:,data_feat_order.traj.index("timestamp")] = arr
                
                return linear_normalize(traj[:,traj_target_indices], *traj_value_range.T, dtype=np.float32)
            

            # Normalization
            for s, traj_list in zip(data["stat_data"], data["traj_data"]):
                stat.append(norm_stat(s))
                sub_traj = list()
                if not do_not_load_traj:
                    for t in traj_list:
                        sub_traj.append(norm_traj(t))
                    traj.append(sub_traj)
            
            # Garbage collection for memory efficiency
            del data
            gc.collect()

        self.n_main_param = n_main_param

        params = np.concatenate(param, dtype=np.float32)
        stats = np.array(stat, dtype=np.float32)
        trajs = np.array(traj, dtype=object) if not do_not_load_traj else None

        self.dataset = dict(
            params = params,
            stat_data = stats,
            traj_data = trajs
        )

        if verbose:
            print(f"[ simulated AnS dataset ({self.simul_model}-{self.simul_model_ckpt}-{self.simul_model_traj_downsample}) loaded for training ]", end=' ') 
            print(f"{n_simul} simulations in total ({self.num_of_epsiode_per_param} eps per param, {self.n_main_param} main params, {self.num_of_sub1param} sub1 params, {self.num_of_sub2param} sub2 params)")


    def _generate_raw_dataset(self, n_main_param=1, repeat=1, num_cpu=None):
        # Settings
        num_cpu = psutil.cpu_count(logical=False) if num_cpu is None else min(num_cpu, psutil.cpu_count(logical=False))

        def get_simul_res(simulator: AnSSimulatorForTwoLevelConditionalAmortizer, i):
            np.random.seed(datetime.now().microsecond + i)
            result = simulator.simulate(
                n_main_param=1,
                n_sub1_param=self.num_of_sub1param,
                n_sub2_param=self.num_of_sub2param,
                sim_per_param_set=self.num_of_epsiode_per_param,
                verbose=False
            )
            return result   # Normalized parameter, summary (nested list of pd.DataFrame), trajectory (nested list of pd.DataFrame)

        for _ in range(repeat):
            num_of_sim = n_main_param * self.num_of_sub1param * self.num_of_sub2param * self.num_of_epsiode_per_param
            filename = "_".join([
                "tr",
                f"{n_main_param}mp",
                f"{self.num_of_sub1param}sp1",
                f"{self.num_of_sub2param}sp2",
                f"{self.num_of_epsiode_per_param}ep",
                f"{num_of_sim}", 
                f"{get_compact_timestamp_str()}"
            ])
            filepath = f"{self.data_path}/{filename}"

            eps = Parallel(n_jobs=num_cpu)(delayed(get_simul_res)(self.simulator, i) for i in tqdm(range(n_main_param)))

            # Save column
            column_info = dict(
                stat = list(eps[0][1][0].columns),
                traj = list(eps[0][2][0][0].columns)
            )
            save_dict_to_yaml(column_info, f"{filepath}.yaml")

            params_arr = np.concatenate([eps[i][0] for i in range(n_main_param)], axis=0, dtype=np.float32)
            stat_data = np.array([pd_beh.to_numpy() for i in range(n_main_param) for pd_beh in eps[i][1]], dtype=np.float32)
            traj_data = np.array([[df.to_numpy(dtype=np.float32) for df in traj_list] for i in range(n_main_param) for traj_list in eps[i][2]], dtype=object)

            npz_save(
                filename=f"{filepath}.npz",
                params=params_arr,
                stat_data=stat_data,
                traj_data=traj_data
            )

    
    def sample(self, batch_sz, sim_per_param=None, stretch_param=False):
        """
        Returns a random sample from the dataset with
        the specified number of main parameter sets (batch size) and sub parameter sets (subparam_per_mainparam),
        and number of simulated trials per parameter-condition pair (sim_per_param).
        """
        if sim_per_param is None:
            sim_per_param = self.num_of_epsiode_per_param

        mp_indices = np.random.choice(self.n_main_param, batch_sz, replace=batch_sz > self.n_main_param)
        ep_indices = np.random.choice(self.num_of_epsiode_per_param, sim_per_param)

        num_of_sub_comb = self.num_of_sub1param * self.num_of_sub2param

        indices = np.repeat(mp_indices * num_of_sub_comb, num_of_sub_comb) + np.tile(np.arange(num_of_sub_comb), batch_sz)

        rows = np.repeat(indices, sim_per_param).reshape((-1, sim_per_param))
        cols = np.tile(ep_indices, (batch_sz * num_of_sub_comb, 1))

        stats = self.dataset["stat_data"][rows, cols]

        const_shape = (batch_sz, self.num_of_sub1param, self.num_of_sub2param)

        if self.dataset["traj_data"] is None:
            traj_data = None
        else:
            traj_data = self.dataset["traj_data"][rows, cols].squeeze(1).reshape(const_shape) if sim_per_param == 1 \
                else self.dataset["traj_data"][rows, cols].reshape((*const_shape, sim_per_param))

        params = self.dataset["params"][indices].reshape((batch_sz, self.num_of_sub1param, self.num_of_sub2param, -1))
        if stretch_param:
            params = np.concatenate((params[:, 0, 0, self.main_param_indices],
                                     params[:, :, 0, self.sub1_param_indices].reshape((batch_sz, -1)),
                                     params[:, :, :, self.sub2_param_indices].reshape((batch_sz, -1))),
                                     axis=1)

        return (
            params, 
            stats.squeeze(1).reshape((*const_shape, -1)) if sim_per_param == 1 else stats.reshape((*const_shape, sim_per_param, -1)), 
            traj_data
        )




class AnSTwoLevelConditionalAmortizerValidationDataset(AnSAmortizerValidationDataset):
    def __init__(
        self, 
        simulator_config="default_lv2_cond",
        normalize_config="default",
        total_user=100, # Main parameter
        total_cond_lv1=2,
        total_cond_lv2=8,
        trial_per_user_cond=128, # Num. of trial per user x cond
        load_existing_data=True,
        verbose=True
    ):
        sim_cfg = INF["simulator"][simulator_config] if isinstance(simulator_config, str) else simulator_config
        
        for necessary_field in ["model_name", "ckpt", "traj_downsample", "main_param", "sub1_param", "sub2_param"]:
            assert necessary_field in sim_cfg, f"{necessary_field} missing in configuration!"

        self.simulator = AnSSimulatorForTwoLevelConditionalAmortizer(
            model_name=sim_cfg["model_name"], 
            ckpt=sim_cfg["ckpt"],
            downsample_rate=sim_cfg["traj_downsample"],
            main_param_list=sim_cfg["main_param"],
            sub1_param_list=sim_cfg["sub1_param"],
            sub2_param_list=sim_cfg["sub2_param"],
            repeat_task_cond=sim_cfg["repeat_task_cond"]
        )

        self.total_user = total_user
        self.total_cond_lv1 = total_cond_lv1
        self.total_cond_lv2 = total_cond_lv2
        self.trial_per_user_cond = trial_per_user_cond

        self.sim_config = sim_cfg
        self.norm_config = INF["normalize"][normalize_config] if isinstance(normalize_config, str) else normalize_config

        self.simul_model = sim_cfg["model_name"]
        self.simul_model_ckpt = sim_cfg["ckpt"]
        self.simul_model_traj_downsample = sim_cfg["traj_downsample"]

        self.simul_main_param = sim_cfg["main_param"]
        self.simul_sub1_param = sim_cfg["sub1_param"]
        self.simul_sub2_param = sim_cfg["sub2_param"]

        self.main_param_indices = np.array(self.simulator.main_param_indices)
        self.sub1_param_indices = np.array(self.simulator.sub1_param_indices)
        self.sub2_param_indices = np.array(self.simulator.sub2_param_indices)

        self.param_setting_code = '_'.join([''.join(map(str, l)) for l in [self.main_param_indices, self.sub1_param_indices, self.sub2_param_indices]])
        basename = "_".join([
            self.sim_config.model_name,
            str(self.sim_config.ckpt),
            str(self.sim_config.traj_downsample),
            self.param_setting_code
        ])
        self.data_path = os.path.join(DIR_TO_DATA, f"data/{FOLDER.AMORTIZER3}/dataset", basename)
        os.makedirs(self.data_path, exist_ok=True)

        if load_existing_data:
            self._get_dataset(verbose)
    

    def _get_dataset(self, verbose=False, **kwargs):
        filename = "_".join([
            "vd",
            f"{self.total_user}users",
            f"{self.total_cond_lv1}cond1",
            f"{self.total_cond_lv2}cond2",
            f"*.npz"
        ])
        filepath = os.path.join(self.data_path, filename)
        datafilelist = glob.glob(filepath)

        if not len(datafilelist):
            raise FileNotFoundError("No data file found for validation ... generate the dataset first.")
        
        datafilelist.sort()
        for f in datafilelist:
        # Feature order
            data = npz_load(f)

            data_feat_order = load_config(f.replace('.npz', '.yaml'))
            stat_target_indices = [data_feat_order.stat.index(key) for key in self.norm_config.stat.list]
            traj_target_indices = [data_feat_order.traj.index(key) for key in self.norm_config.traj.list]
            if "stat_mean" in self.norm_config:
                self.stat_mean_indices = [self.norm_config.stat.list.index(key) for key in self.norm_config.stat_mean.list]

            stat_value_range = np.array([
                [self.norm_config.stat.range[key].min, 
                self.norm_config.stat.range[key].max] for key in self.norm_config.stat.list]
            )
            traj_value_range = np.array([
                [self.norm_config.traj.range[key].min, 
                self.norm_config.traj.range[key].max] for key in self.norm_config.traj.list]
            )

            # Normalize and re-organize
            def norm_stat(stat):
                return linear_normalize(stat[:,stat_target_indices], *stat_value_range.T, dtype=np.float32)
            
            # def norm_traj(traj):    # traj := (n_length x n_feature) np.ndarray
            #     if "timestamp" in data_feat_order.traj:
            #         arr = traj[:,data_feat_order.traj.index("timestamp")]
            #         arr = np.insert(np.diff(arr), 0, 0)
            #         traj[:,data_feat_order.traj.index("timestamp")] = arr
                
            #     return linear_normalize(traj[:,traj_target_indices], *traj_value_range.T, dtype=np.float32)

            param = data["params"]
            stats = np.array([norm_stat(s) for s in data["stat_data"]], dtype=np.float32)
            # trajs = np.array([[norm_traj(t) for t in traj_list] for traj_list in data["traj_data"]], dtype=object)
            trajs = np.array(data["traj_data"], dtype=np.float32)

            self.dataset = dict(
                params = param,
                stat_data = stats,
                traj_data = trajs
            )
            break

        if verbose:
            print(f"[ simulated AnS dataset ({self.simul_model}-{self.simul_model_ckpt}-{self.simul_model_traj_downsample}) loaded for validation ]", end=' ')
            print(f"{self.total_user} users x {self.total_cond_lv1} conditions (1) x {self.total_cond_lv2} conditions (2) x {self.trial_per_user_cond} trials")


    def _generate_raw_dataset(self, num_cpu=None):
        # Settings
        num_cpu = psutil.cpu_count(logical=False) if num_cpu is None else min(num_cpu, psutil.cpu_count(logical=False))

        def get_simul_res(simulator: AnSSimulatorForTwoLevelConditionalAmortizer, i):
            np.random.seed(datetime.now().microsecond + i)
            result = simulator.simulate(
                n_main_param=1,
                n_sub1_param=self.total_cond_lv1,
                n_sub2_param=self.total_cond_lv2,
                sim_per_param_set=self.trial_per_user_cond,
                verbose=False
            )
            return result   # Normalized parameter, summary (nested list of pd.DataFrame), trajectory (nested list of pd.DataFrame)

        eps = Parallel(n_jobs=num_cpu)(delayed(get_simul_res)(self.simulator, i) for i in tqdm(range(self.total_user)))

        filename = "_".join([
            "vd",
            f"{self.total_user}users",
            f"{self.total_cond_lv1}cond1",
            f"{self.total_cond_lv2}cond2",
            f"{get_compact_timestamp_str()}"
        ])
        filepath = os.path.join(self.data_path, filename)

        # Save column
        column_info = dict(
            stat = list(eps[0][1][0].columns),
            traj = list(eps[0][2][0][0].columns),
        )
        save_dict_to_yaml(column_info, f"{filepath}.yaml")

        params_arr = np.concatenate([eps[i][0] for i in range(self.total_user)], axis=0, dtype=np.float32)
        stat_data = np.array([pd_beh.to_numpy() for i in range(self.total_user) for pd_beh in eps[i][1]], dtype=np.float32)
        traj_data = np.array([[df.to_numpy(dtype=np.float32) for df in traj_list] for i in range(self.total_user) for traj_list in eps[i][2]], dtype=object)

        npz_save(
            filename=f"{filepath}.npz",
            params=params_arr,
            stat_data=stat_data,
            traj_data=traj_data
        )
    

    def sample(self, n_trial, n_user=None, random_sample=False):
        """
        Sample the dataset for the given number of trials and users.
        It returns the sampled data based on the specified options (indiv_user and cross_valid).
        If indiv_user is set to True, the data for each user is returned individually.
        If cross_valid is set to True, the data is divided into two parts for cross-validation purposes.
        """
        if n_user is None:
            n_user = self.total_user
        else:
            assert n_user <= self.total_user, f"{n_user} must not exceed total user:{self.total_user}"

        assert n_trial <= self.trial_per_user_cond

        params = list()
        n_total_cond_per_user = self.total_cond_lv1 * self.total_cond_lv2

        if not random_sample:
            user_cond_indices = np.repeat(np.arange(n_user) * n_total_cond_per_user, n_total_cond_per_user) + \
                np.tile(np.arange(n_total_cond_per_user), n_user)
        else:
            selected_user = np.random.choice(np.arange(self.total_user), size=n_user, replace=False)
            user_cond_indices = np.repeat(selected_user * n_total_cond_per_user, n_total_cond_per_user) + \
                np.tile(n_total_cond_per_user, n_user)

        user_stats = list()
        # user_trajs = list()

        for sample_i, user_cond in enumerate(user_cond_indices):
            params.append(self.dataset["params"][user_cond])
            if not random_sample:
                stats = self.dataset["stat_data"][user_cond][:n_trial]
                # trajs = self.dataset["traj_data"][user_cond][:n_trial] if self.dataset["traj_data"] is not None else [None] * n_trial
            else:
                sample_indices = np.random.choice(np.arange(self.trial_per_user_cond), size=n_trial, replace=False)
                stats = self.dataset["stat_data"][user_cond][sample_indices]
                # trajs = self.dataset["traj_data"][user_cond][sample_indices] if self.dataset["traj_data"] is not None else [None] * n_trial

            user_stats.append(stats)
            # user_trajs.append(trajs)

        params = np.array(params, dtype=np.float32).reshape((n_user, self.total_cond_lv1, self.total_cond_lv2, -1))
        params_main = params[:, 0, 0, self.main_param_indices]      # (n_user, dim)
        params_sub1 = params[:, :, 0, self.sub1_param_indices]      # (n_user, n_sub1param, dim)
        params_sub2 = params[:, :, :, self.sub2_param_indices]      # (n_user, n_sub1param, n_sub2param, dim)

        user_stats = np.array(user_stats, dtype=np.float32).reshape((n_user, self.total_cond_lv1, self.total_cond_lv2, n_trial, -1))  # (n_user, *n_subparam, n_trial, dim)
        # user_trajs = np.array(user_trajs, dtype=object).reshape((n_user, self.total_cond_lv1, self.total_cond_lv2, n_trial))  # (n_user, *n_subparam, n_trial)

        return params_main, params_sub1, params_sub2, user_stats, None



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Data synthesis')

    dataset = AnSTwoLevelConditionalAmortizerTrainingDataset(load_existing_data=False)
    dataset._generate_raw_dataset(num_cpu=10, n_main_param=16, repeat=1)

    # dataset = AnSTwoLevelConditionalAmortizerValidationDataset(load_existing_data=False)
    # dataset._generate_raw_dataset(10)
    # pm, ps1, ps2, u, t = dataset.sample(32)
    # print(pm.shape, ps1.shape, ps2.shape, u.shape, t.shape)
