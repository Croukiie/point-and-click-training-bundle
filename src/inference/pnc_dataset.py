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
import pandas as pd
from box import Box
from tqdm import tqdm
import copy
from datetime import datetime
import glob, gc

from ..config.config import INF
from ..config.constant import FOLDER, FIELD
from ..agent.pnc_simulator import PnCSimulator
from ..utils.myutils import get_compact_timestamp_str, npz_load, npz_save, save_dict_to_yaml, load_config
from ..utils.mymath import linear_denormalize, linear_normalize, log_denormalize

DIR_TO_DATA = Path(__file__).parent.parent.parent


class PnCSimulatorForAmortizer:
    def __init__(self, model_name, ckpt, downsample_rate, simulation_option,
                 fix_per_param=list(), task_cond=dict(), user_stat_cond=dict()):
        self.simulator = PnCSimulator(model_name=model_name, ckpt=ckpt, simul_option_name=simulation_option)
        self.user_target_param, self.simul_target_param, self.param_range = self.simulator.modulated_param_list()
        self.total_target_param = self.user_target_param + self.simul_target_param

        # Trajectory data downsampling rate
        self.downsample_rate = downsample_rate
        self.task_cond = task_cond
        self.user_stat_cond = user_stat_cond
        self.game = copy.deepcopy(self.simulator.env.pnc_env)
        self.fixed_condition_per_param = fix_per_param



    def simulate(
        self,
        n_param=1,
        sim_per_param=1,    # No. of simulation per (parameter, task) condition,
        verbose=False,
    ):
        # param_z = np.random.uniform(low=-1, high=1, size=(n_param, len(self.user_target_param)))
        user_param_z = np.array([self.simulator.env._z_sampler() for _ in range(n_param)])
        simul_param_z = np.random.uniform(low=-1, high=1, size=(n_param, len(self.simul_target_param)))

        stats, trajs = list(), list()
        for i in (tqdm(range(n_param)) if verbose else range(n_param)):
            user_z = user_param_z[i]
            simul_z = simul_param_z[i]
            self.simulator.fix_user_param(param_z=dict(zip(self.user_target_param, user_z)))
            self.simulator.fix_simul_param(param_z=dict(zip(self.simul_target_param, simul_z)))
            task_list = self.game.sample_task_conditions(sim_per_param, 
                                                         fix_cond_list=self.fixed_condition_per_param, 
                                                         **self.task_cond)
            user_stat_list = self.simulator.sample_user_stat_conditions(sim_per_param, **self.user_stat_cond)
            self.simulator.simulate(task_list=task_list, user_stat_list=user_stat_list, verbose=False)
            stat_result, traj_result = self.simulator.get_simulation_result(return_traj=True, downsample=self.downsample_rate)
            self.simulator.clear_simulation()

            stats.append(stat_result)   # pd.DataFrame
            trajs.append(traj_result)   # list of pd.DataFrame
        
        return (
            np.concatenate((user_param_z, simul_param_z), axis=1), # np.ndarray with shape (n_param, num of target param)
            stats,   # [pd.df (sim_per_param x n_beh_feat), pd.df, ...] 1D list
            trajs    # [[pd.df traj 1, ... pd.df traj sim_per_param], [ ... ], ..., [ ... ]] 2D nested list
        )

    
    def convert_param_z_to_w(self, outputs):
        param_in = copy.deepcopy(outputs)
        if len(np.array(param_in).shape) == 1:
            param_in = np.expand_dims(param_in, axis=0)

        param_out = np.zeros_like((param_in))

        # User param
        for i, p in enumerate(self.user_target_param):
            if self.param_range[p].type == 'uniform':
                param_out[:,i] = linear_denormalize(
                    param_in[:,i], 
                    self.param_range[p]["min"], 
                    self.param_range[p]["max"]
                )
            elif self.param_range[p].type == 'log-uniform':
                raise NotImplementedError(f"Not implemented: log-uniform denormlization (file {Path(__file__)})")
        
        # Simul param
        for i, p in enumerate(self.simul_target_param):
            if self.param_range[p].type == 'uniform':
                param_out[:,i + len(self.user_target_param)] = linear_denormalize(
                    param_in[:,i + len(self.user_target_param)], 
                    self.param_range[p]["min"], 
                    self.param_range[p]["max"]
                )
            elif self.param_range[p].type == 'log-uniform':
                raise NotImplementedError(f"Not implemented: log-uniform denormlization (file {Path(__file__)})")

        return param_out  




class PnCAmortizerTrainingDataset(object):
    def __init__(
        self, 
        simulator_config="default",
        normalize_config="default",
        num_of_epsiode_per_param=64,
        load_existing_data=True,
        verbose=True
    ):
        sim_cfg = INF["simulator"][simulator_config] if isinstance(simulator_config, str) else simulator_config
        
        for necessary_field in ["model_name", "ckpt", "traj_downsample", "simul_option", "fix_per_param"]:
            assert necessary_field in sim_cfg, f"{necessary_field} missing in configuration!"

        self.simulator = PnCSimulatorForAmortizer(
            model_name=sim_cfg["model_name"], 
            ckpt=sim_cfg["ckpt"],
            downsample_rate=sim_cfg["traj_downsample"],
            simulation_option=sim_cfg["simul_option"],
            fix_per_param=sim_cfg["fix_per_param"]
        )
        self.sim_config = sim_cfg
        self.norm_config = INF["normalize"][normalize_config] if isinstance(normalize_config, str) else normalize_config
        self.num_of_epsiode_per_param = num_of_epsiode_per_param

        basename = "_".join([
            self.sim_config.model_name,
            str(self.sim_config.ckpt),
            str(self.sim_config.traj_downsample),
            self.sim_config.simul_option
        ])
        self.data_path = os.path.join(DIR_TO_DATA, f"data/{FOLDER.AMORTIZER}/dataset/", basename)
        os.makedirs(self.data_path, exist_ok=True)

        self.simul_model = sim_cfg["model_name"]
        self.simul_model_ckpt = sim_cfg["ckpt"]
        self.simul_model_traj_downsample = sim_cfg["traj_downsample"]
        self.simul_option = sim_cfg["simul_option"]

        if load_existing_data:
            self._get_dataset(
                verbose,
                do_not_load_traj=self.norm_config.traj.ignore_traj,
                remove_reaction=self.norm_config.option.remove_reaction,
            )
    

    def _get_dataset(self, verbose=False, do_not_load_traj=True, **kwargs):
        filename = "_".join([
            "tr", f"*p",
            f"{self.num_of_epsiode_per_param}ep",
            f"*.npz"
        ])
        datafilelist = glob.glob(f"{self.data_path}/{filename}")

        if not len(datafilelist):
            raise FileNotFoundError("No data file found ... generate the dataset first.")

        n_param = 0
        n_simul = 0

        param, stat, traj = list(), list(), list()

        datafilelist.sort()
        incompatible_data = 0
        for f in datafilelist if not verbose else tqdm(datafilelist, desc="Processing training data simulations ... "):
            data_feat_order = load_config(f.replace('.npz', '.yaml'))
            if set(data_feat_order.fixed_cond) != set(self.simulator.fixed_condition_per_param):
                incompatible_data += 1
                if incompatible_data == len(datafilelist):
                    raise FileNotFoundError("There is no compatible data. Check the configuration of fixed condition.")
                continue
            
            ff = os.path.basename(f).split('_')
            n_param += int(ff[1][:-1])
            n_simul += int(ff[-2])

            data = npz_load(f)
            param.append(data["params"])

            # Feature order
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
            def norm_stat(stat):    # stat := (n_trial x n_feature) np.ndarray
                return linear_normalize(stat[:,stat_target_indices], *stat_value_range.T, dtype=np.float32)
            
            def norm_traj(traj):    # traj := (n_length x n_feature) np.ndarray
                traj = np.asarray(traj, dtype=np.float32)
                return linear_normalize(
                    traj[:, traj_target_indices],
                    *traj_value_range.T,
                    dtype=np.float32,
                )

            # Normalization
            for s, traj_list in zip(data["stat_data"], data["traj_data"]):
                stat.append(norm_stat(s))
                sub_traj = list()
                if not do_not_load_traj:
                    for t in traj_list:
                        if "remove_reaction" in kwargs and kwargs["remove_reaction"]:
                            rt_flag = (t[:,data_feat_order.traj.index(FIELD.TIMERTFLAG)] == 1)
                            if np.sum(rt_flag) <= 1:
                                t = t[:2]
                            else:
                                t = t[rt_flag]
                        sub_traj.append(norm_traj(t))
                    traj.append(sub_traj)
            
            # Garbage collection for memory efficiency
            del data
            gc.collect()

        self.n_param = n_param

        params = np.concatenate(param, dtype=np.float32)
        stats = np.array(stat, dtype=np.float32)
        trajs = np.array(traj, dtype=object) if not do_not_load_traj else None

        self.dataset = dict(
            params = params,
            stat_data = stats,
            traj_data = trajs
        )

        if verbose:
            print(f"[ simulated PnC dataset ({self.simul_model}-{self.simul_model_ckpt}-{self.simul_model_traj_downsample}) loaded for training ] {n_simul} simulations in total ({self.num_of_epsiode_per_param} eps per param, {self.n_param} params)")


    def _generate_raw_dataset(self, n_param=2**14, repeat=1, num_cpu=None):
        # Settings
        num_cpu = psutil.cpu_count(logical=False) if num_cpu is None else min(num_cpu, psutil.cpu_count(logical=False))

        def get_simul_res(simulator: PnCSimulatorForAmortizer, i):
            np.random.seed(datetime.now().microsecond + i)
            result = simulator.simulate(
                n_param=1,
                sim_per_param=self.num_of_epsiode_per_param,
                verbose=False
            )
            return result   # Normalized parameter, summary (nested list of pd.DataFrame), trajectory (nested list of pd.DataFrame)

        for _ in range(repeat):
            num_of_sim = n_param * self.num_of_epsiode_per_param
            filename = "_".join([
                "tr", 
                f"{n_param}p",
                f"{self.num_of_epsiode_per_param}ep",
                f"{num_of_sim}", 
                f"{get_compact_timestamp_str()}"
            ])
            filepath = f"{self.data_path}/{filename}"

            eps = Parallel(n_jobs=num_cpu)(delayed(get_simul_res)(self.simulator, i) for i in tqdm(range(n_param)))

            # Save column
            column_info = dict(
                stat = list(eps[0][1][0].columns),
                traj = list(eps[0][2][0][0].columns),
                fixed_cond = list(self.simulator.fixed_condition_per_param)
            )
            save_dict_to_yaml(column_info, f"{filepath}.yaml")

            params_arr = np.concatenate([eps[i][0] for i in range(n_param)], axis=0, dtype=np.float32)
            stat_data = np.array([pd_beh.to_numpy() for i in range(n_param) for pd_beh in eps[i][1]], dtype=np.float32)
            traj_data = np.array([[df.to_numpy(dtype=np.float32) for df in traj_list] for i in range(n_param) for traj_list in eps[i][2]], dtype=object)

            npz_save(
                filename=f"{filepath}.npz",
                params=params_arr,
                stat_data=stat_data,
                traj_data=traj_data
            )

    
    def sample(self, batch_sz, sim_per_param=None):
        """
        Returns a random sample from the dataset with
        the specified number of parameter sets (batch size),
        and number of simulated trials per parameter-condition pair (sim_per_param).
        """
        if sim_per_param is None:
            sim_per_param = self.num_of_epsiode_per_param

        indices = np.random.choice(self.n_param, batch_sz, replace=batch_sz > self.n_param)
        ep_indices = np.random.choice(self.num_of_epsiode_per_param, sim_per_param)
        rows = np.repeat(indices, sim_per_param).reshape((-1, sim_per_param))
        cols = np.tile(ep_indices, (batch_sz, 1))
        

        stats = self.dataset["stat_data"][rows, cols]
        if "stat_mean" in self.norm_config:
            if not self.norm_config.stat_mean.ignore_statmean:
                new_stats = list()
                for b in range(batch_sz):
                    s = stats[b]
                    for i in self.stat_mean_indices:
                        val = s[:,i]
                        mean_val = np.mean(val)
                        new_column = np.full((s.shape[0], 1), mean_val)
                        s = np.hstack((s, new_column))
                    new_stats.append(s)
                stats = np.array(new_stats, dtype=np.float32)

        if self.dataset["traj_data"] is None:
            traj_data = None
        else:
            traj_data = self.dataset["traj_data"][rows, cols]

        return (
            self.dataset["params"][indices], 
            stats,
            traj_data
        )


class PnCAmortizerValidationDataset(object):
    def __init__(
        self, 
        simulator_config="default",
        normalize_config="default",
        total_user=100,
        trial_per_user=256,
        load_existing_data=True,
        verbose=True
    ):
        sim_cfg = INF["simulator"][simulator_config] if isinstance(simulator_config, str) else simulator_config
        
        for necessary_field in ["model_name", "ckpt", "traj_downsample", "simul_option", "fix_per_param"]:
            assert necessary_field in sim_cfg, f"{necessary_field} missing in configuration!"

        self.simulator = PnCSimulatorForAmortizer(
            model_name=sim_cfg["model_name"], 
            ckpt=sim_cfg["ckpt"],
            downsample_rate=sim_cfg["traj_downsample"],
            simulation_option=sim_cfg["simul_option"],
            fix_per_param=sim_cfg["fix_per_param"]
        )
        self.total_user = total_user
        self.trial_per_user = trial_per_user
        self.sim_config = sim_cfg
        self.norm_config = INF["normalize"][normalize_config] if isinstance(normalize_config, str) else normalize_config

        basename = "_".join([
            self.sim_config.model_name,
            str(self.sim_config.ckpt),
            str(self.sim_config.traj_downsample),
            self.sim_config.simul_option
        ])
        self.data_path = os.path.join(DIR_TO_DATA, f"data/{FOLDER.AMORTIZER}/dataset/", basename)
        os.makedirs(self.data_path, exist_ok=True)

        self.simul_model = sim_cfg["model_name"]
        self.simul_model_ckpt = sim_cfg["ckpt"]
        self.simul_model_traj_downsample = sim_cfg["traj_downsample"]
        self.simul_option = sim_cfg["simul_option"]

        if load_existing_data:
            self._get_dataset(verbose, remove_reaction=self.norm_config.option.remove_reaction)
    

    def _get_dataset(self, verbose=False, **kwargs):
        filepath = os.path.join(
            self.data_path, 
            f"vd_{self.total_user}users_{self.trial_per_user}trials_*.npz"
        )
        datafilelist = glob.glob(filepath)

        if not len(datafilelist):
            raise FileNotFoundError("No data file found for validation ... generate the dataset first.")
        
        datafilelist.sort()
        incompatible_data = 0
        for f in datafilelist:
        # Feature order
            data_feat_order = load_config(f.replace('.npz', '.yaml'))
            if set(data_feat_order.fixed_cond) != set(self.simulator.fixed_condition_per_param):
                incompatible_data += 1
                if incompatible_data == len(datafilelist):
                    raise FileNotFoundError("There is no compatible data. Check the configuration of fixed condition.")
                continue
        
            data = npz_load(f)
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
            
            def norm_traj(traj):    # traj := (n_length x n_feature) np.ndarray
                traj = np.asarray(traj, dtype=np.float32)
                return linear_normalize(
                    traj[:, traj_target_indices],
                    *traj_value_range.T,
                    dtype=np.float32,
                )

            param = data["params"]
            stats = np.array([norm_stat(s) for s in data["stat_data"]], dtype=np.float32)

            if "remove_reaction" in kwargs and kwargs["remove_reaction"]:
                trajs = list()
                for traj_list in data["traj_data"]:
                    sub_traj = list()
                    for t in traj_list:
                        rt_flag = (t[:,data_feat_order.traj.index(FIELD.TIMERTFLAG)] == 1)
                        if np.sum(rt_flag) <= 1:
                            t = t[:2]
                        else:
                            t = t[rt_flag]
                        sub_traj.append(norm_traj(t))
                    trajs.append(sub_traj)
                trajs = np.array(trajs, dtype=object)
            else:
                trajs = np.array([[norm_traj(t) for t in traj_list] for traj_list in data["traj_data"]], dtype=object)

            self.dataset = dict(
                params = param,
                stat_data = stats,
                traj_data = trajs
            )
            break

        if verbose:
            print(f"[ simulated PnC dataset ({self.simul_model}-{self.simul_model_ckpt}-{self.simul_model_traj_downsample}) loaded for validation ] {self.total_user} users x {self.trial_per_user} trials")


    def _generate_raw_dataset(self, num_cpu=None):
        # Settings
        num_cpu = psutil.cpu_count(logical=False) if num_cpu is None else min(num_cpu, psutil.cpu_count(logical=False))

        def get_simul_res(simulator: PnCSimulatorForAmortizer, i):
            np.random.seed(datetime.now().microsecond + i)
            result = simulator.simulate(
                n_param=1,
                sim_per_param=self.trial_per_user,
                verbose=False
            )
            return result   # Normalized parameter, summary (nested list of pd.DataFrame), trajectory (nested list of pd.DataFrame)

        eps = Parallel(n_jobs=num_cpu)(
            delayed(get_simul_res)(self.simulator, i) for i in tqdm(range(self.total_user))
        )

        filepath = os.path.join(
            self.data_path,
            f"vd_{self.total_user}users_{self.trial_per_user}trials_{get_compact_timestamp_str()}"
        )

        # Save column
        column_info = dict(
            stat = list(eps[0][1][0].columns),
            traj = list(eps[0][2][0][0].columns),
            fixed_cond = list(self.simulator.fixed_condition_per_param)
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
    

    def sample(self, n_trial, n_user=None, indiv_user=False, cross_valid=False, random_sample=False):
        """
        Sample the dataset for the given number of trials and users.
        It returns the sampled data based on the specified options (indiv_user and cross_valid).
        If indiv_user is set to True, the data for each user is returned individually.
        If cross_valid is set to True, the data is divided into two parts for cross-validation purposes.
        """
        if n_user is None:
            n_user = self.total_user
        user_data, valid_user_data = list(), list()
        params = list()
        for user in range(n_user):
            params.append(self.dataset["params"][user])
            if not random_sample:
                stats = self.dataset["stat_data"][user][:n_trial]
                trajs = self.dataset["traj_data"][user][:n_trial] if self.dataset["traj_data"] is not None else [None] * n_trial
            else:
                sample_indices = np.random.choice(np.arange(self.trial_per_user), size=n_trial, replace=False)
                stats = self.dataset["stat_data"][user][sample_indices]
                trajs = self.dataset["traj_data"][user][sample_indices] if self.dataset["traj_data"] is not None else [None] * n_trial
            
            if "stat_mean" in self.norm_config:
                if not self.norm_config.stat_mean.ignore_statmean:
                    for i in self.stat_mean_indices:
                        val = stats[:,i]
                        mean_val = np.mean(val)
                        new_column = np.full((stats.shape[0], 1), mean_val)
                        stats = np.hstack((stats, new_column))


            if indiv_user:
                if cross_valid:
                    half = n_trial // 2
                    user_data.append([stats[:half], trajs[:half]])
                    valid_user_data.append([stats[half:], trajs[half:]])
                else:
                    user_data.append([stats, trajs])
            else:
                if user >= n_user // 2 and cross_valid:
                    valid_user_data.append([stats, trajs])
                else:
                    user_data.append([stats, trajs])

        if cross_valid:
            return np.array(params, dtype=np.float32), user_data, valid_user_data
        else:
            return np.array(params, dtype=np.float32), user_data





if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Data synthesis')
    parser.add_argument('--simul_config', default='mouse_converge_best')
    parser.add_argument('--data_config', default='default')
    parser.add_argument('--n_param', type=int, default=2**14)
    parser.add_argument('--n_trial', type=int, default=64)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--valid_users', type=int, default=100)
    parser.add_argument('--valid_trials', type=int, default=256)
    parser.add_argument('--num_cpu', type=int, default=None)
    parser.add_argument('--skip_train', action='store_true')
    parser.add_argument('--skip_valid', action='store_true')
    args = parser.parse_args()

    if not args.skip_train:
        dataset = PnCAmortizerTrainingDataset(
            simulator_config=args.simul_config,
            normalize_config=args.data_config,
            num_of_epsiode_per_param=args.n_trial,
            load_existing_data=False,
        )
        dataset._generate_raw_dataset(
            num_cpu=args.num_cpu,
            repeat=args.repeat,
            n_param=args.n_param,
        )

    if not args.skip_valid:
        dataset = PnCAmortizerValidationDataset(
            simulator_config=args.simul_config,
            normalize_config=args.data_config,
            total_user=args.valid_users,
            trial_per_user=args.valid_trials,
            load_existing_data=False,
        )
        dataset._generate_raw_dataset(num_cpu=args.num_cpu)
