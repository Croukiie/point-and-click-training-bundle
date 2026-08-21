"""
Dataset generator/loader for amortized inference
Orignial code written by Hee-seung Moon (https://github.com/hsmoon121/amortized-inference-hci)

Code modified by June-Seop Yoon
"""

import os
import sys
from pathlib import Path
from joblib import Parallel, delayed
import psutil
import numpy as np
import pandas as pd
from box import Box
from tqdm import tqdm
import copy
import glob, gc
import hashlib

from ..config.config import INF
from ..config.constant import FOLDER, FIELD
from ..agent.pnc_simulator import PnCSimulator
from ..utils.myutils import get_compact_timestamp_str, npz_load, npz_save, save_dict_to_yaml, load_config
from ..utils.mymath import linear_denormalize, linear_normalize, log_denormalize
from ..inference.pnc_dataset import PnCSimulatorForAmortizer, PnCAmortizerTrainingDataset, PnCAmortizerValidationDataset

DIR_TO_DATA = Path(__file__).parent.parent.parent


def _spawn_worker_seeds(count: int) -> np.ndarray:
    """Create deterministic worker seeds from the caller's seeded RNG state."""
    return np.random.randint(
        0, np.iinfo(np.uint32).max, size=count, dtype=np.uint32
    )


def _seed_digest(seeds: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(seeds, dtype=np.uint32).tobytes()).hexdigest()


class PnCSimulatorForConditionalAmortizer(PnCSimulatorForAmortizer):
    def __init__(self, model_name, ckpt, downsample_rate, simulation_option, 
                 main_param_list, sub_param_list,
                 fix_per_param=None, task_cond=None, user_stat_cond=None):
        super().__init__(
            model_name=model_name,
            ckpt=ckpt,
            downsample_rate=downsample_rate,
            simulation_option=simulation_option,
            fix_per_param=fix_per_param,
            task_cond=task_cond,
            user_stat_cond=user_stat_cond
        )
        self.main_param_list = main_param_list
        self.sub_param_list = sub_param_list

        # Specify the index location of fixed main parameters
        assert set(self.main_param_list).isdisjoint(set(self.sub_param_list)), \
            "Main parameters and sub parameters should not contain common parameter!"
        assert set(self.main_param_list) | set(self.sub_param_list) == set(self.total_target_param), \
            "It seems some main/sub parameters are not included in total parameter set."
        
        self.main_param_indices = list()
        self.sub_param_indices = list()

        for param in self.total_target_param:
            if param in self.main_param_list:
                self.main_param_indices.append(self.total_target_param.index(param))
            elif param in self.sub_param_list:
                self.sub_param_indices.append(self.total_target_param.index(param))
            else:
                raise ValueError("parameter configuration incorrect.")


    def simulate(
        self,
        n_main_param=1,     # Will be fixed - cognitive parameter
        n_sub_param=24,      # Will be sampled per cognitive parameter set
        sim_per_param_set=1,    # No. of simulation per (parameter, task) condition,
        verbose=False,
    ):
        # Change sampling method
        user_main_param_z = np.array([self.simulator.env._z_sampler() for _ in range(n_main_param)])
        simul_main_param_z = np.random.uniform(low=-1, high=1, size=(n_main_param, len(self.simul_target_param)))

        n_param = n_main_param * n_sub_param

        user_param_z = np.array([self.simulator.env._z_sampler() for _ in range(n_param)])
        simul_param_z = np.random.uniform(low=-1, high=1, size=(n_param, len(self.simul_target_param)))

        for param in self.main_param_list:
            if param in self.user_target_param:
                user_param_z[:,self.user_target_param.index(param)] = \
                    np.repeat(user_main_param_z[:,self.user_target_param.index(param)], n_sub_param)
            elif param in self.simul_target_param:
                simul_param_z[:,self.simul_target_param.index(param)] = \
                    np.repeat(simul_main_param_z[:,self.simul_target_param.index(param)], n_sub_param)
            else:
                raise ValueError(f"{param} is not included in parameter set. Possibly configuration error.")

        # Not modified below
        stats, trajs, completion_attempts = list(), list(), list()
        for i in (tqdm(range(n_param)) if verbose else range(n_param)):
            user_z = user_param_z[i]
            simul_z = simul_param_z[i]
            self.simulator.fix_user_param(param_z=dict(zip(self.user_target_param, user_z)))
            self.simulator.fix_simul_param(param_z=dict(zip(self.simul_target_param, simul_z)))
            stat_result, traj_result, attempt_count = self._simulate_completed_trials(
                sim_per_param_set,
                context=f"shared parameter set {i}",
            )

            stats.append(stat_result)   # pd.DataFrame
            trajs.append(traj_result)   # list of pd.DataFrame
            completion_attempts.append(attempt_count)
        
        return (
            np.concatenate((user_param_z, simul_param_z), axis=1), # np.ndarray with shape (n_param, num of target param)
            stats,   # [pd.df (sim_per_param x n_beh_feat), pd.df, ...] 1D list
            trajs,   # [[pd.df traj 1, ... pd.df traj sim_per_param], [ ... ], ..., [ ... ]] 2D nested list
            np.asarray(completion_attempts, dtype=np.int64),
        )

    def convert_param_z_to_w(self, outputs=None, main_outputs=None, sub_outputs=None):
        if outputs is not None:
            return super().convert_param_z_to_w(outputs)
        
        assert main_outputs is not None and sub_outputs is not None, "Specify both parameters!"

        if len(main_outputs.shape) == 1:
            main_in = np.zeros(len(self.total_target_param), dtype=np.float32)
            for i in range(len(self.main_param_list)):
                main_in[self.main_param_indices[i]] = main_outputs[i]
            main_out = super().convert_param_z_to_w(main_in)[0]
            main_out = main_out[self.main_param_indices]
        else:
            main_in = np.zeros((main_outputs.shape[0], len(self.total_target_param)), dtype=np.float32)
            for i in range(len(self.main_param_list)):
                main_in[:,self.main_param_indices[i]] = main_outputs[:,i]
            main_out = super().convert_param_z_to_w(main_in)
            main_out = main_out[:, self.main_param_indices]
        

        if len(sub_outputs.shape) == 1:
            sub_in = np.zeros(len(self.total_target_param), dtype=np.float32)
            for i in range(len(self.sub_param_list)):
                sub_in[self.sub_param_indices[i]] = sub_outputs[i]
            sub_out = super().convert_param_z_to_w(sub_in)[0]
            sub_out = sub_out[self.sub_param_indices]
        else:
            sub_in = np.zeros((sub_outputs.shape[0], len(self.total_target_param)), dtype=np.float32)
            for i in range(len(self.sub_param_list)):
                sub_in[:,self.sub_param_indices[i]] = sub_outputs[:,i]
            sub_out = super().convert_param_z_to_w(sub_in)
            sub_out = sub_out[:, self.sub_param_indices]
        
        return main_out, sub_out
    

    def to_param_dict(self, main_param, sub_params):
        if len(sub_params.shape) == 1:
            return [{**dict(zip(self.main_param_list, main_param)), **dict(zip(self.sub_param_list, sub_params))}]
        return [{**dict(zip(self.main_param_list, main_param)), **dict(zip(self.sub_param_list, sp))} for sp in sub_params]
    


class PnCConditionalAmortizerTrainingDataset(PnCAmortizerTrainingDataset):
    def __init__(
        self, 
        simulator_config="default_cond",
        normalize_config="default",
        num_of_subparam_per_mainparam=24,
        num_of_epsiode_per_param=64,
        load_existing_data=True,
        verbose=True
    ):
        sim_cfg = INF["simulator"][simulator_config] if isinstance(simulator_config, str) else simulator_config
        
        for necessary_field in ["model_name", "ckpt", "traj_downsample", "simul_option", "main_param", "sub_param"]:
            assert necessary_field in sim_cfg, f"{necessary_field} missing in configuration!"

        self.simulator = PnCSimulatorForConditionalAmortizer(
            model_name=sim_cfg["model_name"], 
            ckpt=sim_cfg["ckpt"],
            downsample_rate=sim_cfg["traj_downsample"],
            simulation_option=sim_cfg["simul_option"],
            fix_per_param=sim_cfg["fix_per_param"],
            main_param_list=sim_cfg["main_param"],
            sub_param_list=sim_cfg["sub_param"]
        )
        
        self.sim_config = sim_cfg
        self.norm_config = INF["normalize"][normalize_config] if isinstance(normalize_config, str) else normalize_config
        self.num_of_subparam_per_mainparam = num_of_subparam_per_mainparam
        self.num_of_epsiode_per_param = num_of_epsiode_per_param

        self.simul_model = sim_cfg["model_name"]
        self.simul_model_ckpt = sim_cfg["ckpt"]
        self.simul_model_traj_downsample = sim_cfg["traj_downsample"]
        self.simul_option = sim_cfg["simul_option"]

        self.simul_main_param = sim_cfg["main_param"]
        self.simul_sub_param = sim_cfg["sub_param"]

        self.main_param_indices = np.array(self.simulator.main_param_indices)
        self.sub_param_indices = np.array(self.simulator.sub_param_indices)

        self.param_setting_code = '_'.join([''.join(map(str, l)) for l in [self.main_param_indices, self.sub_param_indices]])
        basename = "_".join([
            self.sim_config.model_name,
            str(self.sim_config.ckpt),
            str(self.sim_config.traj_downsample),
            self.sim_config.simul_option,
            self.param_setting_code,
            str(self.sim_config.get("dataset_contract", "legacy")),
        ])
        self.data_path = os.path.join(DIR_TO_DATA, f"data/{FOLDER.AMORTIZER2}/dataset/", basename)
        os.makedirs(self.data_path, exist_ok=True)

        if load_existing_data:
            self._get_dataset(verbose, remove_reaction=self.norm_config.option.remove_reaction)
    
    
    def _get_dataset(self, verbose=False, do_not_load_traj=None, **kwargs):
        if do_not_load_traj is None:
            do_not_load_traj = bool(self.norm_config.traj.ignore_traj)
        datafilelist = glob.glob(f"{self.data_path}/tr_*mp_{self.num_of_subparam_per_mainparam}sp_{self.num_of_epsiode_per_param}ep_*.npz")

        if not len(datafilelist):
            raise FileNotFoundError("No data file found ... generate the dataset first.")

        n_main_param = 0
        n_simul = 0

        param, stat, traj = list(), list(), list()

        datafilelist.sort(key=os.path.getmtime)
        if len(datafilelist) > 1:
            if verbose:
                print(
                    f"Warning: {len(datafilelist)} matching training datasets "
                    f"were found; using only the newest: {datafilelist[-1]}",
                    flush=True,
                )
            datafilelist = datafilelist[-1:]
        incompatible_data = 0
        for f in datafilelist if not verbose else tqdm(
            datafilelist,
            desc="Loading shared-CPI training data",
            file=sys.stdout,
            dynamic_ncols=True,
            mininterval=1.0,
        ):
            data_feat_order = load_config(f.replace('.npz', '.yaml'))
            if set(data_feat_order.fixed_cond) != set(self.simulator.fixed_condition_per_param):
                incompatible_data += 1
                if incompatible_data == len(datafilelist):
                    raise FileNotFoundError("There is no compatible data. Check the configuration of fixed condition.")
                continue
            
            ff = os.path.basename(f).split('_')
            n_main_param += int(ff[1][:-2])
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
                return linear_normalize(traj[:,traj_target_indices], *traj_value_range.T, dtype=np.float32)

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
            print(f"[ simulated PnC dataset ({self.simul_model}-{self.simul_model_ckpt}-{self.simul_model_traj_downsample}-{self.simul_option}) loaded for training ]", end=' ') 
            print(f"{n_simul} simulations in total ({self.num_of_epsiode_per_param} eps per param, {self.n_main_param} main params, {self.num_of_subparam_per_mainparam} sub params)")


    def _generate_raw_dataset(self, n_main_param=1, repeat=1, num_cpu=None):
        # Settings
        num_cpu = psutil.cpu_count(logical=False) if num_cpu is None else min(num_cpu, psutil.cpu_count(logical=False))
        # Load once on CPU before Joblib serializes worker batches. Inference
        # remains checkpoint-lazy, while large dataset jobs avoid reloading the
        # policy for every simulated participant.
        self.simulator.simulator._ensure_policy_loaded()

        def get_simul_res(simulator: PnCSimulatorForConditionalAmortizer, seed):
            np.random.seed(int(seed))
            result = simulator.simulate(
                n_main_param=1,
                n_sub_param=self.num_of_subparam_per_mainparam,
                sim_per_param_set=self.num_of_epsiode_per_param,
                verbose=False
            )
            return result   # Normalized parameter, summary (nested list of pd.DataFrame), trajectory (nested list of pd.DataFrame)

        for _ in range(repeat):
            worker_seeds = _spawn_worker_seeds(n_main_param)
            num_of_sim = n_main_param * self.num_of_subparam_per_mainparam * self.num_of_epsiode_per_param
            filename = "_".join([
                "tr",
                f"{n_main_param}mp",
                f"{self.num_of_subparam_per_mainparam}sp",
                f"{self.num_of_epsiode_per_param}ep",
                f"{num_of_sim}", 
                f"{get_compact_timestamp_str()}"
            ])
            filepath = f"{self.data_path}/{filename}"
            
            eps = Parallel(n_jobs=num_cpu)(
                delayed(get_simul_res)(self.simulator, worker_seeds[i])
                for i in tqdm(
                    range(n_main_param),
                    desc="Simulating shared participants",
                    file=sys.stdout,
                    dynamic_ncols=True,
                    mininterval=1.0,
                )
            )

            # Save column
            column_info = dict(
                stat = list(eps[0][1][0].columns),
                traj = list(eps[0][2][0][0].columns),
                fixed_cond = list(self.simulator.fixed_condition_per_param),
                schema_version = 3,
                worker_seed_sha256 = _seed_digest(worker_seeds),
                dataset_contract = str(
                    self.sim_config.get("dataset_contract", "legacy")
                ),
                completion_sampling = "fresh_task_and_user_state_after_timeout",
                completion_trials_requested = int(num_of_sim),
                completion_attempts = int(
                    sum(np.sum(worker[3]) for worker in eps)
                ),
            )
            column_info["completion_acceptance_rate"] = float(
                column_info["completion_trials_requested"]
                / column_info["completion_attempts"]
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

    
    def sample(self, batch_sz, subparam_per_mainparam=None, sim_per_param=None, stretch_param=False):
        """
        Returns a random sample from the dataset with
        the specified number of main parameter sets (batch size) and sub parameter sets (subparam_per_mainparam),
        and number of simulated trials per parameter-condition pair (sim_per_param).
        """
        if subparam_per_mainparam is None:
            subparam_per_mainparam = self.num_of_subparam_per_mainparam
        if sim_per_param is None:
            sim_per_param = self.num_of_epsiode_per_param
            
        mp_indices = np.random.choice(self.n_main_param, batch_sz, replace=batch_sz > self.n_main_param)
        sp_indices = np.random.choice(self.num_of_subparam_per_mainparam, subparam_per_mainparam, replace=subparam_per_mainparam > self.num_of_subparam_per_mainparam)
        ep_indices = np.random.choice(
            self.num_of_epsiode_per_param,
            sim_per_param,
            replace=sim_per_param > self.num_of_epsiode_per_param,
        )

        sample_info = dict(
            n_sample_main_param = batch_sz,
            n_sample_sub_param = subparam_per_mainparam,
            n_sample_episode = sim_per_param,

            main_param_indices = self.main_param_indices,
            sub_param_indices=self.sub_param_indices,

            main_param_sample_indices = mp_indices,
            sub_param_sample_indices = sp_indices,
            episode_indices = ep_indices,
        )

        indices = np.repeat(mp_indices * self.num_of_subparam_per_mainparam, subparam_per_mainparam) + np.tile(sp_indices, batch_sz)

        rows = np.repeat(indices, sim_per_param).reshape((-1, sim_per_param))
        cols = np.tile(ep_indices, (batch_sz * subparam_per_mainparam, 1))

        stats = self.dataset["stat_data"][rows, cols]
        if "stat_mean" in self.norm_config:
            if not self.norm_config.stat_mean.ignore_statmean:
                new_stats = list()
                for b in range(batch_sz * subparam_per_mainparam):
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
            traj_data = self.dataset["traj_data"][rows, cols].squeeze(1).reshape((batch_sz, subparam_per_mainparam)) if sim_per_param == 1 \
                else self.dataset["traj_data"][rows, cols].reshape((batch_sz, subparam_per_mainparam, sim_per_param))

        params = self.dataset["params"][indices].reshape((batch_sz, subparam_per_mainparam, -1))
        if stretch_param:
            params = np.concatenate((params[:, 0, self.main_param_indices],
                                     params[:, :, self.sub_param_indices].reshape((batch_sz, -1))),
                                     axis=1)

        return (
            params, 
            stats.squeeze(1).reshape((batch_sz, subparam_per_mainparam, -1)) if sim_per_param == 1 else stats.reshape((batch_sz, subparam_per_mainparam, sim_per_param, -1)), 
            traj_data, 
            sample_info
        )




class PnCConditionalAmortizerValidationDataset(PnCAmortizerValidationDataset):
    def __init__(
        self, 
        simulator_config="default_cond",
        normalize_config="default",
        total_user=100, # Main parameter
        total_cond_per_user=24, # Sub parameter
        trial_per_user_cond=100, # Num. of trial per user x cond
        load_existing_data=True,
        verbose=True
    ):
        sim_cfg = INF["simulator"][simulator_config] if isinstance(simulator_config, str) else simulator_config
        
        for necessary_field in ["model_name", "ckpt", "traj_downsample", "simul_option", "main_param"]:
            assert necessary_field in sim_cfg, f"{necessary_field} missing in configuration!"

        self.simulator = PnCSimulatorForConditionalAmortizer(
            model_name=sim_cfg["model_name"], 
            ckpt=sim_cfg["ckpt"],
            downsample_rate=sim_cfg["traj_downsample"],
            simulation_option=sim_cfg["simul_option"],
            fix_per_param=sim_cfg["fix_per_param"],
            main_param_list=sim_cfg["main_param"],
            sub_param_list=sim_cfg["sub_param"]
        )

        self.total_user = total_user
        self.total_cond_per_user = total_cond_per_user
        self.trial_per_user_cond = trial_per_user_cond

        self.sim_config = sim_cfg
        self.norm_config = INF["normalize"][normalize_config] if isinstance(normalize_config, str) else normalize_config

        self.simul_model = sim_cfg["model_name"]
        self.simul_model_ckpt = sim_cfg["ckpt"]
        self.simul_model_traj_downsample = sim_cfg["traj_downsample"]
        self.simul_option = sim_cfg["simul_option"]

        self.simul_main_param = sim_cfg["main_param"]
        self.simul_sub_param = sim_cfg["sub_param"]

        self.main_param_indices = np.array(self.simulator.main_param_indices)
        self.sub_param_indices = np.array(self.simulator.sub_param_indices)

        self.param_setting_code = '_'.join([''.join(map(str, l)) for l in [self.main_param_indices, self.sub_param_indices]])
        basename = "_".join([
            self.sim_config.model_name,
            str(self.sim_config.ckpt),
            str(self.sim_config.traj_downsample),
            self.sim_config.simul_option,
            self.param_setting_code,
            str(self.sim_config.get("dataset_contract", "legacy")),
        ])
        self.data_path = os.path.join(DIR_TO_DATA, f"data/{FOLDER.AMORTIZER2}/dataset/", basename)
        os.makedirs(self.data_path, exist_ok=True)

        if load_existing_data:
            self._get_dataset(verbose, remove_reaction=self.norm_config.option.remove_reaction)
    

    def _get_dataset(self, verbose=False, **kwargs):
        filepath = os.path.join(
            self.data_path, 
            f"vd_{self.total_user}users_{self.total_cond_per_user}_conds_{self.trial_per_user_cond}trials_*.npz"
        )
        datafilelist = glob.glob(filepath)

        if not len(datafilelist):
            raise FileNotFoundError("No data file found for validation ... generate the dataset first.")
        
        datafilelist.sort(key=os.path.getmtime)
        if len(datafilelist) > 1 and verbose:
            print(
                f"Warning: {len(datafilelist)} matching validation datasets "
                f"were found; using only the newest compatible file.",
                flush=True,
            )
        datafilelist = list(reversed(datafilelist))
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
                return linear_normalize(traj[:,traj_target_indices], *traj_value_range.T, dtype=np.float32)

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
            print(f"[ simulated PnC dataset ({self.simul_model}-{self.simul_model_ckpt}-{self.simul_model_traj_downsample}-{self.simul_option}) loaded for validation ]", end=' ')
            print(f"{self.total_user} users x {self.total_cond_per_user} conditions x {self.trial_per_user_cond} trials")


    def _generate_raw_dataset(self, num_cpu=None):
        # Settings
        num_cpu = psutil.cpu_count(logical=False) if num_cpu is None else min(num_cpu, psutil.cpu_count(logical=False))
        self.simulator.simulator._ensure_policy_loaded()

        def get_simul_res(simulator: PnCSimulatorForConditionalAmortizer, seed):
            np.random.seed(int(seed))
            result = simulator.simulate(
                n_main_param=1,
                n_sub_param=self.total_cond_per_user,
                sim_per_param_set=self.trial_per_user_cond,
                verbose=False
            )
            return result   # Normalized parameter, summary (nested list of pd.DataFrame), trajectory (nested list of pd.DataFrame)

        worker_seeds = _spawn_worker_seeds(self.total_user)
        eps = Parallel(n_jobs=num_cpu)(
            delayed(get_simul_res)(self.simulator, worker_seeds[i])
            for i in tqdm(
                range(self.total_user),
                desc="Simulating validation participants",
                file=sys.stdout,
                dynamic_ncols=True,
                mininterval=1.0,
            )
        )

        filepath = os.path.join(
            self.data_path,
            f"vd_{self.total_user}users_{self.total_cond_per_user}_conds_{self.trial_per_user_cond}trials_{get_compact_timestamp_str()}"
        )

        # Save column
        column_info = dict(
            stat = list(eps[0][1][0].columns),
            traj = list(eps[0][2][0][0].columns),
            fixed_cond = list(self.simulator.fixed_condition_per_param),
            schema_version = 3,
            worker_seed_sha256 = _seed_digest(worker_seeds),
            dataset_contract = str(
                self.sim_config.get("dataset_contract", "legacy")
            ),
            completion_sampling = "fresh_task_and_user_state_after_timeout",
            completion_trials_requested = int(
                self.total_user
                * self.total_cond_per_user
                * self.trial_per_user_cond
            ),
            completion_attempts = int(
                sum(np.sum(worker[3]) for worker in eps)
            ),
        )
        column_info["completion_acceptance_rate"] = float(
            column_info["completion_trials_requested"]
            / column_info["completion_attempts"]
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
    

    def sample(self, n_trial, n_user=None, n_cond=None, indiv_user=False, random_sample=False):
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
        if n_cond is None:
            n_cond = self.total_cond_per_user
        else:
            assert n_cond <= self.total_cond_per_user, f"{n_cond} must not exceed condition per user:{self.total_cond_per_user}"

        # user_data = list()
        # valid_user_data = list()
        params = list()
        sample_info = dict(n_user=n_user, n_cond=n_cond, 
                           main_param_indices=self.main_param_indices,
                           sub_param_indices=self.sub_param_indices)

        if not random_sample:
            user_cond_indices = np.repeat(np.arange(n_user) * self.total_cond_per_user, n_cond) + np.tile(np.arange(n_cond), n_user)
            sample_info["selected_user"] = np.arange(n_user)
            sample_info["selected_cond"] = np.arange(n_cond)
        else:
            selected_user = np.random.choice(np.arange(self.total_user), size=n_user, replace=False)
            selected_cond = np.random.choice(np.arange(self.total_cond_per_user), size=n_cond, replace=False)
            user_cond_indices = np.repeat(selected_user * self.total_cond_per_user, n_cond) + np.tile(selected_cond, n_user)
            sample_info["selected_user"] = selected_user
            sample_info["selected_cond"] = selected_cond

        user_stats = list()
        user_trajs = list()

        for sample_i, user_cond in enumerate(user_cond_indices):
            params.append(self.dataset["params"][user_cond])
            if not random_sample:
                stats = self.dataset["stat_data"][user_cond][:n_trial]
                trajs = self.dataset["traj_data"][user_cond][:n_trial] if self.dataset["traj_data"] is not None else [None] * n_trial
                sample_info["sample_trial_indices"] = np.arange(n_trial)
            else:
                sample_indices = np.random.choice(np.arange(self.trial_per_user_cond), size=n_trial, replace=False)
                stats = self.dataset["stat_data"][user_cond][sample_indices]
                trajs = self.dataset["traj_data"][user_cond][sample_indices] if self.dataset["traj_data"] is not None else [None] * n_trial
                sample_info["sample_trial_indices"] = sample_indices
            
            if "stat_mean" in self.norm_config:
                if not self.norm_config.stat_mean.ignore_statmean:
                    for i in self.stat_mean_indices:
                        val = stats[:,i]
                        mean_val = np.mean(val)
                        new_column = np.full((stats.shape[0], 1), mean_val)
                        stats = np.hstack((stats, new_column))

            # if indiv_user:
            #     if cross_valid:
            #         half = n_trial // 2
            #         user_data.append([stats[:half], trajs[:half]])
            #         valid_user_data.append([stats[half:], trajs[half:]])
            #     else:
            #         user_data.append([stats, trajs])
            # else:
            #     if sample_i >= (n_user // 2) * n_cond and cross_valid:
            #         valid_user_data.append([stats, trajs])
            #     else:
            #         user_data.append([stats, trajs])
            user_stats.append(stats)
            user_trajs.append(trajs)

        params = np.array(params, dtype=np.float32).reshape((n_user, n_cond, -1))
        params_main = params[:, 0, self.main_param_indices] # (n_user, dim)
        params_sub = params[:, :, self.sub_param_indices]   # (n_user, n_subparam, dim)

        user_stats = np.array(user_stats, dtype=np.float32).reshape((n_user, n_cond, n_trial, -1))  # (n_user, n_subparam, n_trial, dim)
        user_trajs = np.array(user_trajs, dtype=object).reshape((n_user, n_cond, n_trial))  # (n_user, n_subparam, n_trial)

        # if cross_valid:
        #     return params_main, params_sub, user_data, valid_user_data, sample_info
        # else:
        # return params_main, params_sub, user_data, sample_info
        return params_main, params_sub, user_stats, user_trajs, sample_info



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Data synthesis')

    dataset = PnCConditionalAmortizerTrainingDataset(load_existing_data=False)
    dataset._generate_raw_dataset(num_cpu=10, n_main_param=70, repeat=1)

    dataset = PnCConditionalAmortizerValidationDataset(load_existing_data=False)
    dataset._generate_raw_dataset(num_cpu=10)

    # dataset = PnCConditionalAmortizerTrainingDataset(load_existing_data=True)
    # p, u, t, i = dataset.sample(10, 24, 50, stretch_param=True)
    # print(p.shape)
