"""
Point-and-Click Simulator
This code implements the simulation queries.

Code written by June-Seop Yoon
"""
import stat
import numpy as np
import pandas as pd
import cv2

import os
from tqdm import tqdm
from pathlib import Path
from box import Box

import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# import imageio

from stable_baselines3 import SAC
from typing import List, Union, Optional

from ..agent import pnc_agent
from ..agent.module_motor import Motor
from ..agent.pnc_task import PnCGame
from ..config.constant import FOLDER, VIDEO, COLOR, CONST, FIELD
from ..config.config import SIM
from ..utils.myutils import load_config, get_compact_timestamp_str
from ..utils.render import RENDER
from ..utils.myplot import figure_grid, figure_save
from ..utils.mymath import linear_normalize, np_interp_nd, cos_sin_array, derivative_central, linear_denormalize


class PnCEpisodeRecordDefault:
    def __init__(self, env):
        self.user = env.user
        self.intv = env.intv
        self.param = env.user_param.copy()
        self.initial_task_cond = env.initial_env_cond.copy()
        self.task_config = Box(env.env_setting_info["task_config"])
        self.max_window_qt = np.array([
            self.task_config.range.window_width.max,
            self.task_config.range.window_width.max * RENDER.ASPECT_RATIO,
        ]) # / 2

        self.actions = list()
        self.perceive = Box(dict(
            current_time = list(),
            cpos_hat_1 = list(),
            cvel_hat_1 = list(),
            tpos_hat_1 = list(),
            tvel_hat_1 = list(),
            tpos_hat_2 = list(),
            tvel_hat_2 = list(),
            hpos_hat_1 = list(),
            hvel_hat_1 = list(),
            cpos_true_0 = list()
        ))
        self.result = Box(dict(
            success = False,
            time = 0,
            click_distance = 0,
            motor_effort = 0,
            click_endpt = None,
            target_endpt = None,
            truncated = False
        ))

    def record(self, env, action, done, truncated, info):
        self.actions.append(env._unpack_action(action))
        for k in self.perceive:
            self.perceive[k].append(info["step_state"][k])

        if done or truncated:
            self.actions = pd.DataFrame(self.actions)
            for k in self.perceive:
                self.perceive[k] = np.array(self.perceive[k])
            self.result.success = info["step_state"].click_success
            self.result.time = int(info["step_state"].click_moment)
            self.result.click_distance = info["step_state"].click_distance
            self.result.truncated = info["step_state"].truncated
            self.result.click_endpt = info["step_state"].click_endpt
            self.result.target_endpt = info["step_state"].target_endpt
            self.result.motor_effort = info["motor_effort"]

            self.param = {**self.param, **info["simul_param"]}

            self.interpolate_action(env)
    

    def interpolate_action(self, env):
        hrt, hp, hv = env.episode_trajectory()
        hp, hv = Motor.interpolate_plan(hp, hv, self.intv.muscle, self.intv.interp)
        self.htraj_p = np.concatenate((np.zeros((hrt // self.intv.interp - self.intv.bump, 2)), hp))[:self.result.time + 1]
        self.htraj_v = np.concatenate((np.zeros((hrt // self.intv.interp - self.intv.bump, 2)), hv))[:self.result.time + 1]
        self.replay = self.replay_and_save()


    def get_summary_result(self):
        target_vel = self.initial_task_cond["target_spd"] * cos_sin_array(self.initial_task_cond["target_dir"])
        traverse_length = np.sum(np.linalg.norm(np.diff(self.replay["cursor_pos"], axis=0), axis=1))
        click_rel_endpt = self.result.target_endpt - self.result.click_endpt
        target_distance = np.linalg.norm(self.initial_task_cond["target_pos"] - self.initial_task_cond["cursor_pos"])
        iod = np.log2(target_distance / (2 * self.initial_task_cond["target_rad"]) + 1)
        # max_cursor_spd = np.max(np.linalg.norm(np.diff(self.replay["cursor_pos"], axis=0), axis=1) / (self.intv.interp / 1000))
        timestamp = np.arange(self.replay["target_pos"].shape[0]) * self.intv.interp / 1000
        max_cursor_spd = np.max(derivative_central(timestamp[1:], 
                                            np.cumsum(np.linalg.norm(np.diff(self.replay["cursor_pos"], axis=0), axis=1))))

        action_kc = self.actions["kc"].to_numpy().astype(int)
        act_click_indices = np.where((action_kc == 1) | (action_kc == 2))[0]
        action_click = int(action_kc[act_click_indices[0]]) if len(act_click_indices) else 0

        return pd.DataFrame([{
            # Task condition
            FIELD.WINDOW_WIDTH: self.initial_task_cond["window_width"],
            FIELD.CURSOR_POS_X0: self.initial_task_cond["cursor_pos"][CONST.X],
            FIELD.CURSOR_POS_Y0: self.initial_task_cond["cursor_pos"][CONST.Y],
            FIELD.TARGET_POS_X0: self.initial_task_cond["target_pos"][CONST.X],
            FIELD.TARGET_POS_Y0: self.initial_task_cond["target_pos"][CONST.Y],
            FIELD.TARGET_VEL_X0: target_vel[CONST.X],
            FIELD.TARGET_VEL_Y0: target_vel[CONST.Y],
            FIELD.TARGET_POS_X0_REL: self.initial_task_cond["target_pos"][CONST.X] - self.initial_task_cond["cursor_pos"][CONST.X],
            FIELD.TARGET_POS_Y0_REL: self.initial_task_cond["target_pos"][CONST.Y] - self.initial_task_cond["cursor_pos"][CONST.Y],
            FIELD.TARGET_RAD: self.initial_task_cond["target_rad"],
            FIELD.TARGET_REW: self.initial_task_cond["target_rew"],
            FIELD.HAND_POS_X0: self.initial_task_cond["hand_pos"][CONST.X],
            FIELD.HAND_POS_Y0: self.initial_task_cond["hand_pos"][CONST.Y],
            FIELD.HAND_VEL_X0: self.initial_task_cond["hand_vel"][CONST.X],
            FIELD.HAND_VEL_Y0: self.initial_task_cond["hand_vel"][CONST.Y],
            FIELD.TARGET_DIST: target_distance,
            FIELD.TARGET_DIFF_ID: iod,
            FIELD.MOUSE_GAIN: self.initial_task_cond["mouse_gain"],

            # Model parameter
            **self.param,

            # Behavioral result
            FIELD.TCT: self.result.time / 1000,     # ms -> s
            FIELD.MVT: max(0, (self.result.time - self.initial_task_cond[FIELD.HAND_RT]) / 1000),
            FIELD.HAND_RT: self.initial_task_cond[FIELD.HAND_RT] / 1000,    # ms -> s
            FIELD.HIT: int(self.result.success),           # bool -> binary
            FIELD.CLICK_DIST: self.result.click_distance,
            FIELD.CLICK_DIST_NORM: self.result.click_distance / self.initial_task_cond["target_rad"],
            FIELD.CLICK_ENDPT_X: self.result.click_endpt[CONST.X],
            FIELD.CLICK_ENDPT_Y: self.result.click_endpt[CONST.Y],
            FIELD.TARGET_ENDPT_X: self.result.target_endpt[CONST.X],
            FIELD.TARGET_ENDPT_Y: self.result.target_endpt[CONST.Y],
            FIELD.CLICK_REL_ENDPT_X: click_rel_endpt[CONST.X],
            FIELD.CLICK_REL_ENDPT_Y: click_rel_endpt[CONST.Y],
            FIELD.TRAV_LEN: traverse_length,
            FIELD.MAX_CURSOR_SPD: max_cursor_spd,
            FIELD.MOTOR_EFFORT: self.result.motor_effort,
            FIELD.TRUNC: int(self.result.truncated),    # bool -> binary

            # Simulation related information
            FIELD.ACTION_KC: action_click
        }])

    def get_trajectory_result(self, downsample=None):
        timestamp = np.arange(self.replay["target_pos"].shape[0]) * self.intv.interp / 1000
        target_pos = self.replay["target_pos"]
        cursor_pos = self.replay["cursor_pos"]
        hand_pos = self.replay["hand_pos"]

        if downsample is not None:
            assert type(downsample) is int
            timestamp_new = np.linspace(0, timestamp[-1], round(timestamp[-1] * downsample) + 1)
            target_pos = np_interp_nd(timestamp_new, timestamp, target_pos)
            cursor_pos = np_interp_nd(timestamp_new, timestamp, cursor_pos)
            hand_pos = np_interp_nd(timestamp_new, timestamp, hand_pos)
            timestamp = timestamp_new
        
        reaction_flag = np.array(timestamp >= (self.initial_task_cond[FIELD.HAND_RT] / 1000), dtype=int)

        return pd.DataFrame({
            FIELD.TIMESTAMP: timestamp,
            FIELD.TIMEDIFF: np.insert(np.diff(timestamp), 0, 0),
            FIELD.TIMERTFLAG: reaction_flag,
            FIELD.TARGET_POS_X: target_pos[:, CONST.X],
            FIELD.TARGET_POS_Y: target_pos[:, CONST.Y],
            FIELD.CURSOR_POS_X: cursor_pos[:, CONST.X],
            FIELD.CURSOR_POS_Y: cursor_pos[:, CONST.Y],
            FIELD.HAND_POS_X: hand_pos[:, CONST.X],
            FIELD.HAND_POS_Y: hand_pos[:, CONST.Y],
            FIELD.TARGET_POS_X_REL: target_pos[:, CONST.X] - cursor_pos[:, CONST.X],
            FIELD.TARGET_POS_Y_REL: target_pos[:, CONST.Y] - cursor_pos[:, CONST.Y]
        })
    

    def cursor_speed_profile(self):
        data = self.get_trajectory_result()
        timestamp = data[FIELD.TIMESTAMP]
        cursor_pos = np.vstack((data[FIELD.CURSOR_POS_X], data[FIELD.CURSOR_POS_Y])).T
        cursor_mdist = np.cumsum(np.linalg.norm(np.diff(cursor_pos, axis=0), axis=1))
        cursor_mdist = np.insert(cursor_mdist, 0, 0)
        speed = derivative_central(timestamp, cursor_mdist)

        return timestamp, speed


    def action_list(self):
        return pd.DataFrame(dict(zip(self.user.action.list, self.actions.T)))


    def replay_and_save(self):
        # keys=["cursor_pos", "target_pos", "hand_pos", "cursor_vel", "target_vel", "hand_vel"]
        pnc_replay = PnCGame(config=self.task_config)
        pnc_replay.reset(**self.initial_task_cond)
        return pnc_replay.replay_and_save(self.htraj_p, self.htraj_v, self.intv.interp, unit='ms')
    

    def export_single_replay(self, filepath=None, filename=None, scale=0.7):

        fig, axs = figure_grid(2, 2, size_ax=np.array([8.0, 4.5])*scale)
        if filepath is None:
            filepath = os.path.join(
                Path(__file__).parent.parent.parent, 
                f"data/{FOLDER.VISUALIZATION}/{FOLDER.GIF}/{self.model_name}"
            )
        if filename is None:
            filename = get_compact_timestamp_str()

        # Animated gif
        # 2D replay, cursor speed, actions (th, kc)
        
        # plt.plot(*replay["target_pos"].T, linewidth=1, color='k')
        plt.plot(*self.replay["cursor_pos"][:int(self.result.time)].T, linewidth=0.5, color='k')
        plt.scatter(*self.perceive["cpos_true_0"].T, s=15, color='r', marker='x')
        plt.scatter(*self.result.click_endpt, s=50, color='r', marker='.')

        # plt.plot(*replay["target_pos"].T, linewidth=0.5, color='r')
        # plt.scatter(*replay["target_pos"][-1], s=50, color='r')
        plt.xlim(-self.initial_task_cond["window_qt"][0], self.initial_task_cond["window_qt"][0])
        plt.ylim(-self.initial_task_cond["window_qt"][1], self.initial_task_cond["window_qt"][1])
        plt.gca().set_aspect('equal')
        plt.show()
    

    def render_episode(
        self,
        videowriter: cv2.VideoWriter,
        framerate:int,
        resolution:int,
        hold_interval:int,   # Frame
    ):
        # Interpolate to draw in given framerate
        timestamp_original = np.arange(self.replay["target_pos"].shape[0]) * self.intv.interp
        timestamp = np.linspace(0, self.result.time, int(round(self.result.time * framerate / 1000)) + 1)

        traj_m = dict()
        for key in self.replay:
            traj_m[key] = np_interp_nd(timestamp, timestamp_original, self.replay[key])

        frame_num = timestamp.size
        for i in range(frame_num):
            scene = self.__draw_task_scene(
                resolution,
                traj_m["cursor_pos"][i],
                traj_m["target_pos"][i],
                self.initial_task_cond["target_rad"]
            )
            scene = RENDER.put_text_at(
                scene,
                f"Cursor: ({traj_m['cursor_pos'][i][0]:.3f}, {traj_m['cursor_pos'][i][1]:.3f})",
                (0.05, 0.2),
                0.5,
                COLOR.BLACK
            )
            scene = RENDER.put_text_at(
                scene,
                f"Target: ({traj_m['target_pos'][i][0]:.3f}, {traj_m['target_pos'][i][1]:.3f})",
                (0.05, 0.25),
                0.5,
                COLOR.BLACK
            )
            # scene = RENDER.draw_trajectory(scene, traj["cursor_pos"][:i+1], COLOR.BLACK)
            # scene = RENDER.draw_trajectory(scene, traj["target_pos"][:i+1], COLOR.RED)

            ### Add hand information later (maybe)
            ### Add text information
            videowriter.write(scene)
        
        scene = RENDER.put_text_at(
            scene,
            f"Click distance: {self.result['click_distance']:.3f} m, Target radius: {self.initial_task_cond['target_rad']:.3f} m",
            (0.05, 0.05),
            0.5,
            COLOR.BLACK
        )
        scene = RENDER.put_text_at(
            scene,
            "Hit" if self.result["success"] else "Fail",
            (0.05, 0.1),
            0.5,
            COLOR.BLACK
        )
        for _ in range(hold_interval): videowriter.write(scene)
    

    def __draw_task_scene(self, resolution, cursor_pos, target_pos, target_rad, cursor_rad=0.002):
        # Black screen
        scene = RENDER.empty_scene(resolution, COLOR.WHITE)

        # Border
        edgeX, edgeY = self.initial_task_cond["window_qt"]
        (p1, p2, p3, p4) = (
            RENDER.convert_meter_to_pixel(np.array([edgeX, edgeY]), resolution, self.max_window_qt),
            RENDER.convert_meter_to_pixel(np.array([-edgeX, edgeY]), resolution, self.max_window_qt),
            RENDER.convert_meter_to_pixel(np.array([-edgeX, -edgeY]), resolution, self.max_window_qt),
            RENDER.convert_meter_to_pixel(np.array([edgeX, -edgeY]), resolution, self.max_window_qt)
        )
        scene = RENDER.draw_line(scene, p1, p2, COLOR.DARKGRAY)
        scene = RENDER.draw_line(scene, p2, p3, COLOR.DARKGRAY)
        scene = RENDER.draw_line(scene, p3, p4, COLOR.DARKGRAY)
        scene = RENDER.draw_line(scene, p4, p1, COLOR.DARKGRAY)

        # Maximum window width = Image width
        tp_pixel = RENDER.convert_meter_to_pixel(target_pos, resolution, self.max_window_qt)
        cp_pixel = RENDER.convert_meter_to_pixel(cursor_pos, resolution, self.max_window_qt)
        tr_pixel = max(1, int(round(target_rad / (2*self.max_window_qt[CONST.Y]) * resolution)))
        cr_pixel = max(1, int(round(cursor_rad / (2*self.max_window_qt[CONST.Y]) * resolution)))

        # Draw both objects
        scene = RENDER.draw_circle(scene, tp_pixel, tr_pixel, COLOR.RED)
        scene = RENDER.draw_circle(scene, cp_pixel, cr_pixel, COLOR.BLACK)

        # Flip horizontally (opencv uses flipped starting position in Y-axis)
        scene = cv2.flip(scene, 0)

        return scene
    

class PnCSimulator:
    def __init__(
        self, 
        model_name: str, 
        ckpt: Optional[int] | str = 'best', 
        record_type=PnCEpisodeRecordDefault,
        simul_option_name = "default"
    ):
        ### Load PnC model

        # Policy file
        if ckpt == 'best':
            self.policy = os.path.join(
                Path(__file__).parent.parent.parent, 
                f"data/{FOLDER.DATA_RL_MODEL}/{model_name}/{FOLDER.MODEL_BEST}/best_model.zip"
            )
        elif type(ckpt) is int:
            self.policy = os.path.join(
                Path(__file__).parent.parent.parent, 
                f"data/{FOLDER.DATA_RL_MODEL}/{model_name}/{FOLDER.MODEL_CHECKPT}/rl_model_{ckpt}_steps.zip"
            )
        else:
            raise TypeError("Argument ckpt must be either int or 'best'.")
        
        self.model = SAC.load(self.policy)
        
        # Configurations
        env_config = load_config(os.path.join(
            Path(__file__).parent.parent.parent, 
            f"data/{FOLDER.DATA_RL_MODEL}/{model_name}/env_info.yaml"
        ))

        self.model_name = f"{model_name}_{ckpt}"

        env_class = getattr(pnc_agent, env_config.env_class)
        env_intv = env_config.interval
        env_task = env_config.task_config
        env_user = env_config.user_config

        # Load environment
        self.env = env_class(
            agent_cfg=env_user,
            pnc_env_cfg=env_task,
            interval_cfg=env_intv,
        )

        self.simul_option = SIM.simul_option[simul_option_name]
        self._setup_simultor_only_param()

        # List of simulated result
        self.simulation = list()

        self.user_param_fixed = False
        self.simul_param_fixed = False
        self.record_type = record_type

        self.user_param_list = self.env.user.param_modul.list
        # simul_param_list is set in below function.
    

    def _setup_simultor_only_param(self):
        # These were not included in RL agent,
        # but explicitly added for extended simulation
        self.simul_param_list = self.simul_option.param.list
        self.simul_param_range = {p: self.simul_option.param[p] for p in self.simul_param_list}
        self.current_simul_param = self._sample_simul_param()   # Not normalized

    
    def _sample_simul_param(self):
        sample = dict()

        for p in self.simul_param_list:
            if self.simul_param_range[p].type == "uniform":
                v = np.random.uniform(
                    self.simul_param_range[p].min,
                    self.simul_param_range[p].max,
                )
            else:
                raise NotImplementedError("PnCSimulator._sample_simul_param not implemented for current sample option.")
            sample[p] = v

        return sample
    
    def reset_simul_param(self):
        self.current_simul_param = self._sample_simul_param()
    

    def modulated_param_list(self):
        param_range = Box({p: self.env.user.param_modul[p] for p in self.user_param_list})
        return (
            self.user_param_list, 
            self.simul_param_list, 
            Box({**param_range, **self.simul_param_range})
        )
    

    def simulation_param_list(self):
        return self.simul_param_list, self.simul_param_range

    
    def fix_user_param(self, param_z=None, param_w=None):
        # assert not (param_z is None and param_w is None), "No parameter information given."
        if param_z is not None:
            # if isinstance(param_z, dict):
            self.env.fix_z(param=param_z)
            # else:
            #     self.env.fix_z(param=dict(zip(self.modulated_param_list()[0], param_z)))
            self.user_param_fixed = True
        elif param_w is not None:
            z = dict()
            for v, w in param_w.items():
                # if self.env.user.param_modul[v].type == 'uniform':
                z[v] = linear_normalize(w, self.env.user.param_modul[v].min, self.env.user.param_modul[v].max)
            self.env.fix_z(param=z)
            self.user_param_fixed = True
    

    def unfix_user_param(self):
        self.env.unfix_z()
        self.user_param_fixed = False
    

    def fix_simul_param(self, param_z=None, param_w=None):
        if param_z is not None:
            w = dict()
            for v, z in param_z.items():
                w[v] = linear_denormalize(z, self.simul_param_range[v].min, self.simul_param_range[v].max)
            self.current_simul_param.update(w)
            self.simul_param_fixed = True
        
        elif param_w is not None:
            self.current_simul_param.update(param_w)
            self.simul_param_fixed = True
            

    def unfix_simul_param(self):
        self.simul_param_fixed = False

    
    def fix_total_param(self, param_z=None, param_w=None):
        if param_z is not None:
            user_z = {p: param_z[p] for p in self.user_param_list}
            simul_z = {p: param_z[p] for p in self.simul_param_list}
            self.fix_user_param(param_z=user_z)
            self.fix_simul_param(param_z=simul_z)
        
        elif param_w is not None:
            user_w = {p: param_w[p] for p in self.user_param_list}
            simul_w = {p: param_w[p] for p in self.simul_param_list}
            self.fix_user_param(param_w=user_w)
            self.fix_simul_param(param_w=simul_w)

    def unfix_total_param(self):
        self.unfix_user_param()
        self.unfix_simul_param()

    
    def split_total_param_list(self, param_list):
        # param_list -> user_param_list, simul_param_list
        user_param_list = [{p: param[p] for p in self.user_param_list} for param in param_list]
        simul_param_list = [{p: param[p] for p in self.simul_param_list} for param in param_list]
        return user_param_list, simul_param_list


    def set_pnc_task(self, setting:dict=None):
        if setting is not None:
            self.env.update_task_setting(pnc_env_setting=setting)
    

    def set_user_stat(self, setting:dict=None):
        if setting is not None:
            self.env.update_user_status(user_stat_setting=setting)


    def sample_user_stat_conditions(self, n, **conds):
        return [self.env.user_stat_sampler(**conds) for _ in range(n)]


    def clear_simulation(self):
        self.simulation = list()
    

    def simulate(
        self,
        user_param_list: Optional[List[dict]] = None,
        sim_param_list: Optional[List[dict]] = None,
        task_list: Optional[List[dict]] = None,
        user_stat_list: Optional[List[dict]] = None,
        num_simul: Optional[int] = None,
        overwrite_existing_simul: bool=False,
        resimulate_max_num: int=0,
        verbose=True
    ):
        if self.user_param_fixed and user_param_list is not None:
            print("Warning - The user parameters are currently fixed, but they will be overwritten by input param list...")

        
        if num_simul is None:
            assert not (user_param_list is None and task_list is None and user_stat_list is None), \
            "num_simul is not specified. Specify at least one simulation condition list."

            assertions = [len(x) for x in [user_param_list, sim_param_list, task_list, user_stat_list] if x is not None]
            assert all(length == assertions[0] for length in assertions), \
            "All specified simul conditions must have the same length."

            num_simul = assertions[0]
            user_param_list = user_param_list if user_param_list is not None else [{} for _ in range(num_simul)]
            sim_param_list = sim_param_list if sim_param_list is not None else [{} for _ in range(num_simul)]
            task_list = task_list if task_list is not None else [None for _ in range(num_simul)]
            user_stat_list = user_stat_list if user_stat_list is not None else [None for _ in range(num_simul)]

        else:
            assert user_param_list is None and task_list is None and user_stat_list is None, \
            "num_simul is specified. The rest should be set as None."
            user_param_list = [{} for _ in range(num_simul)]
            sim_param_list = [{} for _ in range(num_simul)]
            task_list = [None for _ in range(num_simul)]
            user_stat_list = [None for _ in range(num_simul)]
        
        n_resim = 0

        if overwrite_existing_simul:
            self.clear_simulation()

        iterator = zip(
            tqdm(user_param_list, desc="Simulations: ") if verbose else user_param_list,
            sim_param_list,
            task_list,
            user_stat_list
        )

        for n_ep, (cond_param, cond_sim_param, cond_task, cond_user_stat) in enumerate(iterator):
            self.fix_user_param(**cond_param)
            self.fix_simul_param(**cond_sim_param)
            self.set_pnc_task(cond_task)
            self.set_user_stat(cond_user_stat)

            if not self.simul_param_fixed:
                self.reset_simul_param()

            while True: # Simulation loop
                obs, _ = self.env.reset()
                ep_rec = self.record_type(self.env)
                done = False
                truncated = False

                while not done:
                    action, state = self.model.predict(getattr(self, self.simul_option.func)(obs), deterministic=True)
                    obs, rew, done, truncated, info = self.env.step(action)
                    ep_rec.record(self.env, action, done, truncated, {**info, "simul_param": self.current_simul_param})
                    if done or truncated:
                        break
                
                if not truncated or (n_resim == resimulate_max_num):
                    n_resim = 0
                    self.simulation.append(ep_rec)
                    break
                else:
                    n_resim += 1

    ### ----------------------------------------------------------
    # SIMULATION OPTION
    # OBSERVATION STATE MANIPULATION
    def obs_manipulation_none(self, obs):
        return obs
    
    def obs_manipulation_reward_perception(self, obs):
        new_obs = obs.copy()
        if self.env.timer < self.current_simul_param["theta_r"]:
            for param in ["rew_succ_0", "rew_fail_0"]:
                if param in self.simul_option.param.list and \
                    self.simul_option.param[param].target in self.env.user.param_modul.list:
                    param_to_be_distorted = self.simul_option.param[param].target
                    obs_index = self.env.user.param_modul.list.index(param_to_be_distorted)
                    new_obs[obs_index] = linear_normalize(
                        self.current_simul_param[param],
                        self.env.user.param_modul[param_to_be_distorted].min,
                        self.env.user.param_modul[param_to_be_distorted].max,
                    )
        return new_obs
    ### ----------------------------------------------------------

    
    def get_simulation_result(self, return_traj=False, downsample=None):
        summ, traj = list(), list()
        for rec in self.simulation:
            summ.append(rec.get_summary_result())
            if return_traj:
                traj.append(rec.get_trajectory_result(downsample=downsample))
        summ = pd.concat(summ)
        summ.reset_index()
        
        if return_traj:
            return summ, traj
        return summ
    

    def export_simulation_video(
        self,
        framerate: int = 60,
        resolution: int = 720,
        interval: Union[float, List[float]] = 0.6,    # sec
        video_path: Optional[str] = None,
        video_name: Optional[str] = None,
        warning_threshold: int = 100,
        verbose=True
    ):
        framerate = min(300, framerate)

        num_simul = len(self.simulation)
        if num_simul > warning_threshold:
            print(f"Warning: there are {num_simul} episode saved.")
            print("Are you sure to render all of these?")
            ans = input("Type y to proceed: ")
            if ans != 'y' and ans != 'Y':
                return
        
        # Interval between episodes (frame)
        if type(interval) is list:
            interval = (np.array(interval) * framerate).astype(int)
            assert num_simul == interval.size
        else:
            interval = [int(interval * framerate)] * num_simul

        if video_path is None:
            video_path = os.path.join(
                Path(__file__).parent.parent.parent, 
                f"data/{FOLDER.VISUALIZATION}/{FOLDER.VIDEO}/{self.model_name}"
            )
        os.makedirs(video_path, exist_ok=True)

        if video_name is None:
            video_name = get_compact_timestamp_str()

        video = cv2.VideoWriter(
            f"{video_path}/{video_name}.mp4",
            VIDEO.CODEC,
            framerate, 
            RENDER.img_resolution(resolution)
        )
    
        print(f"Rendering {num_simul} simulation(s) ...")

        for episode, rec in enumerate(tqdm(self.simulation)) if verbose else enumerate(self.simulation):
            rec.render_episode(
                video,
                framerate,
                resolution,
                interval[episode],    # Number of frame
            )
        video.release()


if __name__ == "__main__":
    simulator = PnCSimulator("PnCClassicClick_25197EXSJ", 10000000)
    # simulator.fix_user_param(param_z=dict(
    #     theta_m=0,
    #     theta_c=0,
    # ))
    # simulator.fix_simul_param(param_w=dict(
    #     theta_r=150,
    #     rew_succ_0=32,
    #     rew_fail_0=16
    # ))
    from time import time
    start_time = time()
    simulator.simulate(num_simul=1000)
    end_time = time()
    print(f"Simulation time for 1000 episodes: {end_time - start_time:.2f} sec")
    # r = simulator.simulation[0]
    # print(r.actions)
    # s = simulator.get_simulation_result()
    # s.to_csv("asdf.csv", index=False)

    # p, r = simulator.modulated_param_list()
    # print(p)
    # print(r)