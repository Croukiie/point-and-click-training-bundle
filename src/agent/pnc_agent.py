"""
Point-and-Click base agent

Code written by June-Seop Yoon
Original: https://github.com/dodoseung/point-and-click-simulation-model
"""

import numpy as np
import scipy as sp
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from gymnasium.envs.registration import EnvSpec
# gym.logger.set_level(40)

from box import Box

from .pnc_task import PnCGame
from .module_motor import Motor
from .module_perceive import Perceive
from .module_click import Click
from ..config.config import SIM, AGN
from ..utils.mymath import (
    cos_sin_array,
    linear_denormalize,
    linear_normalize,
)
from ..config.constant import FIELD, BASE_SCORE, TOTAL_TIME_SEC, RING_TIMER


class PnCEnvDefault(gym.Env):
    def __init__(
        self,
        spec_name: str="pnc-base",
        agent_name: str="default",
        pnc_env_name: str="default",
        interval_name: str="default",
        agent_cfg=None,
        pnc_env_cfg=None,
        interval_cfg=None,
    ):
        self.viewer = None
        self.spec = EnvSpec(spec_name)
        
        # Configuration settings
        self.intv = SIM.interval[interval_name] if interval_cfg is None else interval_cfg
        self.user = AGN[agent_name] if agent_cfg is None else agent_cfg

        # User parameter settings
        self.param_mod_z = {p: 0.0 for p in self.user.param_modul.list}
        self.user_param = dict()
        if self.user.param_const is not None:
            for v, w in self.user.param_const.items():
                self.user_param[v] = w
        self.sample_mod_z = True
        self.z_sample_min_prob = list()
        for p in self.user.param_modul.list:
            if "sample_min_prob" in self.user.param_modul[p]:
                self.z_sample_min_prob.append(self.user.param_modul[p].sample_min_prob)
            else:
                self.z_sample_min_prob.append(0)
        self.z_sample_min_prob = np.array(self.z_sample_min_prob)

        ### Observation and Action space
        self.observation_space = spaces.Box(    # Modulated parameter + environment state
            -np.ones(len(self.user.param_modul.list) + np.sum([len(self.user.observation[key].max) for key in self.user.observation.list]), dtype=np.float32),
            np.ones( len(self.user.param_modul.list) + np.sum([len(self.user.observation[key].max) for key in self.user.observation.list]), dtype=np.float32),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            -np.ones(len(self.user.action.list), dtype=np.float32),
             np.ones(len(self.user.action.list), dtype=np.float32),
            dtype=np.float32
        )
        self.observation_range = np.vstack((
            np.concatenate([self.user.observation[key].min for key in self.user.observation.list]),
            np.concatenate([self.user.observation[key].max for key in self.user.observation.list])
        )).T
        self.action_range = np.vstack((
            [self.user.action[key].min for key in self.user.action.list],
            [self.user.action[key].max for key in self.user.action.list]
        )).T

        # Point-and-Click environment
        self.pnc_env_setting = dict(
            window_width = None,
            cursor_pos = None,
            cursor_vel = None,
            hand_pos = None,
            hand_vel = None,
            target_pos = None,
            target_rad = None,
            target_spd = None,
            target_dir = None,
            mouse_gain = None,
            mouse_ori = None,
        )
        self.pnc_env = PnCGame(config_name=pnc_env_name, config=pnc_env_cfg)

        # User status that is non-cognitive parameter
        self.user_status_setting = {
            FIELD.HAND_RT: None,     # Random sample hand reaction time. To fix, specify value,
            FIELD.CLICK_PRD: None,
        }

        # Cognitive state
        self.cstate = dict(
            target_pos = None,
            target_pos_rel = None,
            target_vel = None,
            target_rad = None,
            cursor_pos = None,
            cursor_vel = None,
            window_qt = None,
            hand_pos = None,
            hand_vel = None,
            mouse_ori = None,
            target_rew = None,
            target_rew_weighted = None,
            elapsed_time = None
        )
        pnc_current_state = self.pnc_env.get_current_state()
        self.cstate.update(pnc_current_state)
        self.cstate["elapsed_time"] = 0
        self.cstate["target_rew_weighted"] = self.cstate["target_rew"] * self.__rew_succ_scale()

        # Motor plans
        self.ongoing_mp_ideal = dict(
            p = np.zeros((self.intv.bump // self.intv.muscle + 1, 2)),
            v = np.zeros((self.intv.bump // self.intv.muscle + 1, 2)),
        )
        self.ongoing_mp_actual = dict(
            p = np.zeros((self.intv.bump // self.intv.muscle + 1, 2)),
            v = np.zeros((self.intv.bump // self.intv.muscle + 1, 2)),
        )
        self.pending_mp_ideal = dict(p=None, v=None)
        self.pending_mp_actual = dict(p=None, v=None)

        self.click_mp_determined = False
        self.click_timing_determined = False

        self.click_timing = 100000  # 100 s
        self.truncation = False
        self.hand_reaction_time = 2 * self.intv.bump

        self.timer = 0
        self.result = 0
        self.motor_effort = 0

        # Behavior history
        self.h_traj_p_delta = list()
        self.h_traj_v = list()
        self.initial_env_cond = dict()

        self.env_setting_info = dict(
            user_config = self.user.to_dict(),
            task_config = self.pnc_env.config.to_dict(),
            interval = self.intv.to_dict(),
        )
    

    def __rew_succ_scale(self):
        for key in ["money_weight", "w_earn"]:
            if key in self.user_param:
                return self.user_param[key]
        return 1


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    

    def update_task_setting(self, pnc_env_setting=dict()):
        self.pnc_env_setting.update(pnc_env_setting)

    def update_user_status(self, user_stat_setting=dict()):
        self.user_status_setting.update(user_stat_setting)
    

    def np_cstate(self):
        param_z = np.array([self.param_mod_z[v] for v in self.user.param_modul.list])
        obs = np.concatenate([np.atleast_1d(self.cstate[key]) for key in self.user.observation.list])
        obs = linear_normalize(obs, *self.observation_range.T, clip=False)
        return np.concatenate((param_z, obs)).astype(np.float32)
    

    def episode_trajectory(self):
        return self.hand_reaction_time, np.cumsum(self.h_traj_p_delta, axis=0), self.h_traj_v


    def update_params(self):
        for v in self.user.param_modul.list:
            parameter_range = self.user.param_modul[v]
            value = linear_denormalize(
                self.param_mod_z[v],
                parameter_range.min,
                parameter_range.max,
            )
            if "step" in parameter_range:
                step = float(parameter_range.step)
                value = parameter_range.min + round(
                    (float(value) - parameter_range.min) / step
                ) * step
                value = float(np.clip(value, parameter_range.min, parameter_range.max))
                # Keep the policy conditioning value exactly aligned with the
                # quantized physical value used by the environment.
                self.param_mod_z[v] = float(
                    linear_normalize(value, parameter_range.min, parameter_range.max)
                )
            self.user_param[v] = value
        for param, attr in [
            ("cpi", "cpi"),
            ("mouse_rotation_gain_x", "mouse_rotation_gain_x"),
            ("mouse_rotation_gain_y", "mouse_rotation_gain_y"),
            ("mouse_plan_correction", "mouse_plan_correction"),
        ]:
            if param in self.user_param:
                setattr(self.pnc_env, attr, self.user_param[param])


    def fix_z(self, param:dict=None):
        self.sample_mod_z = False
        if param is not None:
            for v, z in param.items():
                self.param_mod_z[v] = z
            self.update_params()
    

    def unfix_z(self):
        self.sample_mod_z = True


    def _z_sampler(self, sample_min_prob=0.0):
        ext = self.z_sample_min_prob / (1 - self.z_sample_min_prob)
        sampled = np.clip(
            np.random.uniform(
                -ext - 1,
                np.ones(len(self.user.param_modul.list), dtype=np.float32),
            ),
            -1,
            1,
        )
        for index, name in enumerate(self.user.param_modul.list):
            parameter_range = self.user.param_modul[name]
            if "step" not in parameter_range:
                continue
            step = float(parameter_range.step)
            n_steps = int(
                round((parameter_range.max - parameter_range.min) / step)
            )
            value = parameter_range.min + np.random.randint(0, n_steps + 1) * step
            sampled[index] = linear_normalize(
                value,
                parameter_range.min,
                parameter_range.max,
            )
        return sampled.astype(np.float32)


    def sample_z(self, sample_min_prob=0.0):
        new_z = self._z_sampler(sample_min_prob=sample_min_prob)
        for i, v in enumerate(self.param_mod_z):
            self.param_mod_z[v] = new_z[i]
    
    def np_z(self):
        return np.array([self.param_mod_z[v] for v in self.user.param_modul.list])

    
    def user_stat_sampler(self):
        """
        Sample hand reaction time only... for now.
        """
        stat = dict()
        if self.user_status_setting[FIELD.HAND_RT] is not None:
            stat[FIELD.HAND_RT] = np.clip(self.user_status_setting[FIELD.HAND_RT], 
                                          self.user.reaction.hand.min, self.user.reaction.hand.max, dtype=int)
        elif self.user.reaction.hand.type == 'uniform':
            stat[FIELD.HAND_RT] = np.random.randint(self.user.reaction.hand.min, self.user.reaction.hand.max+1)
        elif self.user.reaction.hand.type == 'skewnorm':
            while True:
                hrt = round(sp.stats.skewnorm(
                    self.user.reaction.hand.alpha,
                    self.user.reaction.hand.mean,
                    self.user.reaction.hand.std
                ).rvs(1)[0])
                if self.user.reaction.hand.min <= hrt <= self.user.reaction.hand.max:
                    break
            stat[FIELD.HAND_RT] = hrt
        
        if "click_periods" in self.user:
            if self.user_status_setting[FIELD.CLICK_PRD] is not None:
                stat[FIELD.CLICK_PRD] = np.clip(self.user_status_setting[FIELD.CLICK_PRD], 
                                            self.user.click_periods.min, 
                                            self.user.click_periods.max, dtype=int)
            elif self.user.click_periods.type == 'normal':
                stat[FIELD.CLICK_PRD] = round(np.clip(np.random.normal(
                    self.user.click_periods.mean, 
                    self.user.click_periods.std
                ), self.user.click_periods.min, self.user.click_periods.max))
            else:
                raise NotImplementedError(f"click period sampling for {self.user.click_periods.type} incomplete.")
        else:
            stat[FIELD.CLICK_PRD] = 0   # Will not be used, just for dummy 

        return stat
    

    def reset(
        self, 
        seed=None, 
        sample_min_prob=0.0,
    ):
        if self.sample_mod_z:
            self.sample_z(sample_min_prob)
        self.update_params()

        self.initial_env_cond = self.pnc_env.reset(**self.pnc_env_setting, return_reset_condition=True)

        # Customizable information
        current_user_stat = self.user_stat_sampler()
        self.hand_reaction_time = current_user_stat[FIELD.HAND_RT]
        self.click_period_time = current_user_stat[FIELD.CLICK_PRD]
        self.initial_env_cond[FIELD.HAND_RT] = self.hand_reaction_time

        # Set initial state at the moment when the reaction time is 100 ms left
        self.timer = self.hand_reaction_time - self.intv.bump
        self.pnc_env.update_target_move(self.timer, unit='ms')
        self.result = 0
        self.motor_effort = 0

        self.click_timing = 100_000
        self.click_mp_determined = False
        self.click_timing_determined = False
        self.click_give_up = False

        self.truncation = False

        self.ongoing_mp_ideal = dict(
            p = np.zeros((self.intv.bump // self.intv.muscle + 1, 2)),
            v = np.zeros((self.intv.bump // self.intv.muscle + 1, 2)),
        )
        self.ongoing_mp_actual = dict(
            p = np.zeros((self.intv.bump // self.intv.muscle + 1, 2)),
            v = np.zeros((self.intv.bump // self.intv.muscle + 1, 2)),
        )
        self.pending_mp_ideal = dict(p=None, v=None)
        self.pending_mp_actual = dict(p=None, v=None)
        
        pnc_current_state = self.pnc_env.get_current_state()
        self.cstate.update(pnc_current_state)
        self.cstate["elapsed_time"] = self.timer / 1000

        # Reward perception
        self._update_reward_cstate()

        # Elapsed time perception
        self.cstate["elapsed_time"] = Perceive.timing_perception(self.timer / 1000, self.user_param["theta_c"])
        
        self.h_traj_p_delta = np.array([self.pnc_env.hand_pos]) # np.cumsum will be applied at the last call
        self.h_traj_v = np.array([self.pnc_env.hand_vel])

        info = dict(
            time = self.timer,
            is_success = bool(self.result)
        )

        return self.np_cstate(), info


    def _update_reward_cstate(self):
        self.cstate["target_rew"] = self.pnc_env.target_rew if \
            self.timer >= self.user_param.get("rew_proc_t", -1) else \
                self.user_param.get("init_stake_belief", self.pnc_env.target_rew)
        
        self.cstate["target_rew_weighted"] = self.pnc_env.target_rew * self.__rew_succ_scale() if \
            self.timer >= self.user_param.get("rew_proc_t", -1) else \
                self.user_param.get("r_future", self.pnc_env.target_rew * self.__rew_succ_scale())


    def step(self, action_norm):
        action = self._unpack_action(action_norm)
        state = self._step_init_state()

        self._perceive_and_predict(state, action)
        self._plan_hand_movement(state, action)
        self._plan_click(state, action)
        self._check_click_result(state, action)
        self._update_to_next_step(state, action)

        return self._terminate_or_truncate(state, action)


    def _unpack_action(self, action_norm):
        action_val = linear_denormalize(action_norm, *self.action_range.T)
        _action = {a: v for a, v in zip(self.user.action.list, action_val)}
        
        # Processing action values
        _action["th"] = int(round(_action["th"]) // self.intv.muscle) * self.intv.muscle
        _action["th"] = min(max(_action["th"], int(round(self.user.action.th.min))), int(round(self.user.action.th.max)))
        _action["kc"] = bool(_action["kc"] >= self.user.action.kc.threshold)

        return _action


    def _step_init_state(self):
        return Box(dict(
            current_time = self.timer / 1000,
            tpos_hat_0 = self.pnc_env.target_pos.copy(),    # t = t0
            cpos_hat_0 = self.pnc_env.cursor_pos.copy(),
            cpos_true_0 = self.pnc_env.cursor_pos.copy(),
            target_rad = self.pnc_env.target_rad
            # tdir_true_0 = self.pnc_env.target_dir,
            # tpos_true_0 = self.pnc_env.target_pos.copy(),
        ))


    def _perceive_and_predict(self, state, action):
        # Cursor and hand future state prediction
        # state.cpos_hat_0 = Perceive.position_perception(...)
        (
            state.cpos_hat_1,
            state.cvel_hat_1,
            state.hpos_hat_1,
            state.hvel_hat_1,
            state.mouse_ori_hat_1,
        ) = self.pnc_env.get_cursor_and_hand_update(
            self.ongoing_mp_ideal["p"],
            self.ongoing_mp_ideal["v"],
            cursor_pos=state.cpos_hat_0,
            return_mouse_ori=True,
        )
        # Target state prediction
        # state.tpos_hat_0 = Perceive.position_perception(...)
        state.tvel_hat_0 = Perceive.speed_perception(
            self.pnc_env.target_vel, state.tpos_hat_0, self.user_param["theta_s"]
        )
        state.tpos_hat_1, tspd_hat_1, tdir_hat_1 = self.pnc_env.get_target_update(
            self.intv.bump, unit='ms', 
            target_pos=state.tpos_hat_0, 
            target_vel=state.tvel_hat_0
        )
        state.tvel_hat_1 = tspd_hat_1 * cos_sin_array(tdir_hat_1)
        state.tpos_hat_2, tspd_hat_2, tdir_hat_2 = self.pnc_env.get_target_update(
            self.intv.bump + action["th"], unit='ms', 
            target_pos=state.tpos_hat_0, 
            target_vel=state.tvel_hat_0
        )
        state.tvel_hat_2 = tspd_hat_2 * cos_sin_array(tdir_hat_2)


    def _plan_hand_movement(self, state, action):
        if not self.click_mp_determined:
            # Click motor plan not determined, and BUMP continues
            if not action["kc"]:
                cursor_plan_p, cursor_plan_v = Motor.plan_hand_movement(
                    state.cpos_hat_1,
                    state.cvel_hat_1,
                    state.tpos_hat_2,
                    state.tvel_hat_2,
                    plan_duration=action["th"],
                    execute_duration=self.intv.bump,    # Cut at BUMP interval
                    interval=self.intv.muscle,
                    maximum_hand_speed=self.user.max_hand_speed * self.pnc_env.cursor_gain
                )
                state.ideal_plan_p, state.ideal_plan_v = self.pnc_env.convert_cursor_plan_to_hand_plan(
                    cursor_plan_p,
                    cursor_plan_v,
                    hand_pos=self.ongoing_mp_actual["p"][-1],
                    lead_in_htraj_p=self.ongoing_mp_actual["p"],
                    interval=self.intv.muscle,
                )
                state.noisy_plan_p, state.noisy_plan_v = Motor.add_motor_noise(
                    state.ideal_plan_p[0],
                    state.ideal_plan_v,
                    self.intv.muscle,
                    self.user_param["theta_m"],
                    n_p = self.user_param["theta_m"] * self.user_param["theta_n"],
                )
            # Create fixed motor plan for click
            else:
                self.click_mp_determined = True
                self.click_give_up = False
                # if isinstance(action["kc"], bool):
                #     self.click_give_up = False
                # elif isinstance(action["kc"], int): # This branch works for extended ICP mechanism
                #     # self.click_give_up = (action["kc"] == 1)    # Don't give up if 2, give up if 1
                #     self.click_give_up = False
                # else:
                #     raise RuntimeError(f"Something gone wrong... check action kc value: {action['kc']}")

                cursor_plan_p, cursor_plan_v = Motor.plan_hand_movement(
                    state.cpos_hat_1,
                    state.cvel_hat_1,
                    state.tpos_hat_2,
                    state.tvel_hat_2,
                    plan_duration=action["th"],
                    execute_duration=action["th"],  # Full length
                    interval=self.intv.muscle,
                    maximum_hand_speed=self.user.max_hand_speed * self.pnc_env.cursor_gain
                )
                expand_length = self.user.truncate_time // self.intv.muscle + 1
                cursor_plan_v = np.pad(cursor_plan_v, ((0, expand_length), (0, 0)), mode='edge')
                cursor_plan_p = np.concatenate((
                    cursor_plan_p, 
                    cursor_plan_p[-1] + (cursor_plan_v[-1][:, np.newaxis] * np.arange(1, expand_length+1)).T * (self.intv.muscle/1000)
                ))
                ideal_plan_p, ideal_plan_v = self.pnc_env.convert_cursor_plan_to_hand_plan(
                    cursor_plan_p,
                    cursor_plan_v,
                    hand_pos=self.ongoing_mp_actual["p"][-1],
                    lead_in_htraj_p=self.ongoing_mp_actual["p"],
                    interval=self.intv.muscle,
                )
                noisy_plan_p, noisy_plan_v = Motor.add_motor_noise(
                    ideal_plan_p[0],
                    ideal_plan_v,
                    self.intv.muscle,
                    self.user_param["theta_m"],
                    n_p = self.user_param["theta_m"] * self.user_param["theta_n"],
                )

                bump_length = self.intv.bump // self.intv.muscle

                # Store motor plans on pending plan storage (queue)
                self.pending_mp_ideal["p"] = ideal_plan_p[bump_length:]
                self.pending_mp_ideal["v"] = ideal_plan_v[bump_length:]
                self.pending_mp_actual["p"] = noisy_plan_p[bump_length:]
                self.pending_mp_actual["v"] = noisy_plan_v[bump_length:]

                # Maintain "next motor plan length" as tp
                state.ideal_plan_p = ideal_plan_p[:bump_length + 1]
                state.ideal_plan_v = ideal_plan_v[:bump_length + 1]
                state.noisy_plan_p = noisy_plan_p[:bump_length + 1]
                state.noisy_plan_v = noisy_plan_v[:bump_length + 1]
                
        else:
            # Pop motor plan from queue
            bump_length = self.intv.bump // self.intv.muscle

            state.noisy_plan_p = self.pending_mp_actual["p"][:bump_length + 1]
            state.noisy_plan_v = self.pending_mp_actual["v"][:bump_length + 1]

            ideal_plan_p = self.pending_mp_ideal["p"][:bump_length + 1]
            state.ideal_plan_p = ideal_plan_p + (state.noisy_plan_p[0] - ideal_plan_p[0])
            state.ideal_plan_v = self.pending_mp_ideal["v"][:bump_length + 1]

            self.pending_mp_ideal["p"] = self.pending_mp_ideal["p"][bump_length:]
            self.pending_mp_ideal["v"] = self.pending_mp_ideal["v"][bump_length:]
            self.pending_mp_actual["p"] = self.pending_mp_actual["p"][bump_length:]
            self.pending_mp_actual["v"] = self.pending_mp_actual["v"][bump_length:]
            
    

    def _plan_click(self, state, action):
        if self.click_mp_determined and not self.click_timing_determined:
            self.click_timing_determined = True
            self.click_timing = np.clip(
                int(round(Perceive.timing_perception(self.intv.bump + action["th"] * action["tc"], self.user_param["theta_c"]))),
                self.intv.bump + self.intv.interp, np.inf
            )
            # print(f"Click determined: {self.click_timing} (current time: {self.timer})")


    def _check_click_result(self, state, action):
        if self.click_timing <= self.intv.bump:
            # Interpolate motor plan and compute the accurate (1ms unit) click point
            interp_plan_p, interp_plan_v = Motor.interpolate_plan(
                self.ongoing_mp_actual["p"],
                self.ongoing_mp_actual["v"],
                self.intv.muscle,
                self.intv.interp
            )
            ending_cursor_pos, *_ = self.pnc_env.get_cursor_and_hand_update(
                interp_plan_p[:int(np.ceil(self.click_timing / self.intv.interp)) + 1],
                interp_plan_v[:int(np.ceil(self.click_timing / self.intv.interp)) + 1],
            )
            ending_target_pos, *_ = self.pnc_env.get_target_update(self.click_timing, unit='ms')
            state.click_distance = np.linalg.norm(ending_cursor_pos - ending_target_pos)
            state.click_success = state.click_distance <= self.pnc_env.target_rad
            state.click_moment = self.timer + self.click_timing
            state.elapsed_time_step = self.click_timing
            state.click_endpt = ending_cursor_pos
            state.target_endpt = ending_target_pos
        
        state.done = self.click_timing <= self.intv.bump
        state.truncated = False
            

    def _update_to_next_step(self, state, action):
        # Current time step passed
        self.timer += self.intv.bump
        self.click_timing -= self.intv.bump
        self.pnc_env.update_target_move(self.intv.bump, unit='ms')
        self.pnc_env.update_cursor_and_hand_move(self.ongoing_mp_actual["p"], self.ongoing_mp_actual["v"])

        # Record history
        self.h_traj_p_delta = np.append(self.h_traj_p_delta, np.diff(self.ongoing_mp_actual["p"], axis=0), axis=0)
        self.h_traj_v = np.append(self.h_traj_v, self.ongoing_mp_actual["v"][1:], axis=0)

        prev_time_hat = self.cstate["elapsed_time"]

        # Update cognitive state
        self.cstate.update(
            dict(
                target_pos = state.tpos_hat_1,
                target_pos_rel = state.tpos_hat_1 - state.cpos_hat_1,
                target_vel = state.tvel_hat_1,
                cursor_pos = state.cpos_hat_1,
                cursor_vel = state.cvel_hat_1,
                hand_pos = state.hpos_hat_1,
                hand_vel = state.hvel_hat_1,
                mouse_ori = np.array([np.cos(state.mouse_ori_hat_1), np.sin(state.mouse_ori_hat_1)]),
                elapsed_time = self.timer / 1000
            )
        )
        self._update_reward_cstate()
        state.perceived_elapsed_time = Perceive.timing_perception(self.intv.bump / 1000, 
                                                                  self.user_param["theta_c"])
        if not state.done:
            self.cstate["elapsed_time"] = prev_time_hat + state.perceived_elapsed_time
        else:
            self.cstate["elapsed_time"] = prev_time_hat + state.elapsed_time_step / 1000

        # Update motor effort cumsum
        state.motor_effort = Motor.accel_sum(self.ongoing_mp_actual["v"])
        self.motor_effort += state.motor_effort

        # Update plan
        self.ongoing_mp_ideal.update(dict(p=state.ideal_plan_p, v=state.ideal_plan_v))
        self.ongoing_mp_actual.update(dict(p=state.noisy_plan_p, v=state.noisy_plan_v))


    def _terminate_or_truncate(self, state, action):
        if self.timer > self.user.truncate_time + self.hand_reaction_time:
            state.truncated = True
        
        if not state.done and state.truncated:
            state.click_distance = self.pnc_env.get_cursor_target_distance()
            state.click_success = False
            state.click_moment = self.timer
            state.elapsed_time_step = self.intv.bump
            state.click_endpt = self.pnc_env.cursor_pos.copy()
            state.target_endpt = self.pnc_env.target_pos.copy()

        reward = self._reward_function(state, action)
        info = dict(
            time = state.click_moment if state.done else self.timer,
            result = state.click_success if state.done else False,
            is_success = state.click_success if state.done else False,
            motor_effort = self.motor_effort,
            step_state = state
        )
        if state.done or state.truncated:
            info["episode_parameters"] = self._episode_parameter_metadata(state)

        return self.np_cstate(), reward, state.done, state.truncated, info


    def _episode_parameter_metadata(self, state):
        """Serializable parameters and task conditions for evaluation logging."""
        initial = self.initial_env_cond
        cursor_pos = np.asarray(initial["cursor_pos"], dtype=float)
        target_pos = np.asarray(initial["target_pos"], dtype=float)
        target_radius = float(initial["target_rad"])
        target_distance = float(np.linalg.norm(target_pos - cursor_pos))
        target_difficulty = float(
            np.log2(target_distance / max(2 * target_radius, 1e-12) + 1)
        )

        return {
            "normalized": {
                name: float(self.param_mod_z[name])
                for name in self.user.param_modul.list
            },
            "actual": {
                name: float(np.asarray(self.user_param[name]).reshape(-1)[0])
                for name in self.user.param_modul.list
            },
            "task": {
                "target_radius": target_radius,
                "target_speed": float(initial["target_spd"]),
                "target_direction": float(initial["target_dir"]),
                "target_difficulty": target_difficulty,
                "target_distance": target_distance,
                "window_width": float(initial["window_width"]),
                "mouse_base_gain": float(initial["mouse_gain"]),
                "mouse_gain": float(initial["effective_mouse_gain"]),
                "cpi": float(initial["cpi"]),
                "initial_mouse_orientation_rad": float(initial["mouse_ori"]),
                "cursor_pos": cursor_pos.tolist(),
                "target_pos": target_pos.tolist(),
            },
            "outcome": {
                "success": bool(state.click_success),
                "click_distance": float(state.click_distance),
                "elapsed_time": float(state.click_moment) / 1000,
                "motor_effort": float(self.motor_effort),
            },
            # "environment_defaults": {
            #     "mouse_rotation_gain_x": float(
            #         self.pnc_env.config.get("mouse_rotation_gain_x", -3.0)
            #     ),
            #     "mouse_rotation_gain_y": float(
            #         self.pnc_env.config.get("mouse_rotation_gain_y", 0.0)
            #     ),
            # },
        }


    def _reward_function(self, state, action):
        if state.done:
            click_reward = self.user_param["rew_succ"] if state.click_success else -self.user_param["rew_fail"]
            time_cost = state.elapsed_time_step / 1000 * self.user_param["time_penalty"]
            effort_cost = state.motor_effort * self.user_param["effort_cost"]

            return click_reward - time_cost - effort_cost + self.user_param.get("rew_const", 0)
        
        elif state.truncated:
            return self.user.truncate_penalty
        
        return -state.perceived_elapsed_time * self.user_param["time_penalty"] - state.motor_effort * self.user_param["effort_cost"]


    # def _reward_function(self, state, action):
    #     if state.done:
    #         target_reward = self.user_param["target_stake"] if state.click_success else -self.user_param["target_stake"]
    #         base_reward = self.user_param["base_reward"]

    #         tct = state.click_moment / 1000 + RING_TIMER * self.user_param.get("include_ring_time", 0)
    #         time_penalty = -tct * self.user_param["time_cost_weight"]

    #         return target_reward + base_reward + time_penalty
        
    #     elif state.truncated:
    #         return self.user.truncate_penalty
        
    #     return 0



class PnCClassicClick(PnCEnvDefault):
    # def _unpack_action(self, action_norm):
    #     action_val = linear_denormalize(action_norm, *self.action_range.T)
    #     _action = {a: v for a, v in zip(self.user.action.list, action_val)}
        
    #     # Processing action values
    #     _action["th"] = int(round(_action["th"]) // self.intv.muscle) * self.intv.muscle
    #     _action["th"] = min(max(_action["th"], int(round(self.user.action.th.min))), int(round(self.user.action.th.max)))
    #     _action["kc"] = int(np.digitize(_action["kc"], self.user.action.kc.threshold)) ### Modification made

    #     return _action
    

    def _plan_click(self, state, action):
        if self.click_mp_determined and not self.click_timing_determined:
            self.click_timing_determined = True
            interpolated_plan_p, interpolated_plan_v = Motor.interpolate_plan(
                state.ideal_plan_p, state.ideal_plan_v,
                self.intv.muscle, self.intv.interp)
            future_cursor_traj_p, *_ = self.pnc_env.get_cursor_and_hand_update(
                interpolated_plan_p,  interpolated_plan_v,
                cursor_pos=state.cpos_hat_1, return_traj=True)
            self.click_timing = Click.intermittent_click_planning(
                future_cursor_traj_p,  state.tpos_hat_2, state.target_rad,
                self.intv.interp, self.click_period_time,
                self.user_param["theta_c"], self.user_param["c_mu"],
                self.user_param["nu"], self.user_param["delta"],
                offset=self.intv.bump)

    


class PnCClassicRew(PnCEnvDefault):
    def _reward_function(self, state, action):
        if state.done:
            click_reward = self.user_param["rew_succ"] if state.click_success else -self.user_param["rew_fail"]
            time_cost = state.elapsed_time_step / 1000 * self.user_param["time_penalty"]
            effort_cost = state.motor_effort * self.user_param["effort_cost"]

            return click_reward - time_cost - effort_cost
        
        elif state.truncated:
            return self.user.truncate_penalty
        
        return -state.perceived_elapsed_time * self.user_param["time_penalty"] - state.motor_effort * self.user_param["effort_cost"]


class PnCClassicSymmetric(PnCEnvDefault):
    def _reward_function(self, state, action):
        if state.done:
            click_reward = self.user_param["rew_succ"] if state.click_success \
                else -self.user_param["rew_succ"] * self.user_param["fail_scale"]
            time_cost = state.elapsed_time_step / 1000 * self.user_param["time_penalty"]
            effort_cost = state.motor_effort * self.user_param["effort_cost"]

            return click_reward - time_cost - effort_cost
        
        elif state.truncated:
            return self.user.truncate_penalty
        
        return -state.perceived_elapsed_time * self.user_param["time_penalty"] - state.motor_effort * self.user_param["effort_cost"]





######################################


class PnCRewSimple(PnCEnvDefault):
    def _reward_function(self, state, action):
        if state.done:
            tct = state.click_moment / 1000
            target_reward = self.pnc_env.target_rew if state.click_success else -self.pnc_env.target_rew
            base_reward = BASE_SCORE / TOTAL_TIME_SEC * tct

            rew_x = target_reward + base_reward
            return self.user_param["w_earn"] * rew_x if rew_x >= 0 \
                else self.user_param.get("w_loss",  self.user_param["w_earn"]) * rew_x
        
        elif state.truncated:
            return self.user.truncate_penalty
        
        return 0


    def _update_reward_cstate(self):
        self.cstate["target_rew"] = self.pnc_env.target_rew



class PnCRewSimplePerc(PnCEnvDefault):
    def _reward_function(self, state, action):
        if state.done:
            tct = state.click_moment / 1000
            target_reward = self.pnc_env.target_rew if state.click_success else -self.pnc_env.target_rew
            base_reward = BASE_SCORE / TOTAL_TIME_SEC * tct

            rew_x = target_reward + base_reward
            return self.user_param["w_earn"] * rew_x if rew_x >= 0 \
                else self.user_param.get("w_loss",  self.user_param["w_earn"]) * rew_x
        
        elif state.truncated:
            return self.user.truncate_penalty
        
        return 0


class PnCRewSimpleTime(PnCEnvDefault):
    def _reward_function(self, state, action):
        if state.done:
            tct = state.click_moment / 1000
            target_reward = self.pnc_env.target_rew if state.click_success else -self.pnc_env.target_rew
            base_reward = BASE_SCORE / TOTAL_TIME_SEC * tct
            time_penalty = self.user_param["w_time"] * state.elapsed_time_step / 1000

            rew_x = target_reward + base_reward
            return self.user_param["w_earn"] * rew_x - time_penalty if rew_x >= 0 \
                else self.user_param.get("w_loss",  self.user_param["w_earn"]) * rew_x - time_penalty
        
        elif state.truncated:
            return self.user.truncate_penalty
        
        return -state.perceived_elapsed_time * self.user_param["w_time"]

    def _update_reward_cstate(self):
        self.cstate["target_rew"] = self.pnc_env.target_rew if \
            self.timer >= self.user_param.get("theta_r", -1) else \
                self.user_param.get("init_stake_belief", self.pnc_env.target_rew)






####################### BELOW IS ~250626
    



class PnCRewPercV1(PnCEnvDefault):
    def _reward_function(self, state, action):
        if state.done:
            tct = state.click_moment / 1000
            target_reward = self.pnc_env.target_rew if state.click_success else -self.pnc_env.target_rew
            base_reward = BASE_SCORE / TOTAL_TIME_SEC * tct
            time_penalty = self.user_param["init_stake_belief"] * state.elapsed_time_step / 1000

            return self.user_param["money_weight"] * (target_reward + base_reward) - time_penalty
        
        elif state.truncated:
            return self.user.truncate_penalty
        
        return -state.perceived_elapsed_time * self.user_param["init_stake_belief"]


class PnCRewPercV2(PnCEnvDefault):
    def _reward_function(self, state, action):
        if state.done:
            tct = state.click_moment / 1000
            target_reward = self.pnc_env.target_rew if state.click_success else -self.pnc_env.target_rew
            base_reward = BASE_SCORE / TOTAL_TIME_SEC * tct
            time_penalty = self.user_param["init_stake_belief"] * tct

            rew_x = target_reward + base_reward - time_penalty
            return self.user_param["w_earn"] * rew_x if rew_x >= 0 \
                else self.user_param.get("w_loss",  self.user_param["w_earn"]) * rew_x
        
        elif state.truncated:
            return self.user.truncate_penalty
        
        return 0
    

class PnCRewPercV3(PnCEnvDefault):
    def _reward_function(self, state, action):
        if state.done:
            tct = state.click_moment / 1000
            target_reward = self.pnc_env.target_rew if state.click_success else -self.pnc_env.target_rew
            base_reward = BASE_SCORE / TOTAL_TIME_SEC * tct
            time_penalty = self.user_param["r_life"] * state.elapsed_time_step / 1000

            rew_x = target_reward + base_reward
            return self.user_param["w_earn"] * rew_x - time_penalty if rew_x >= 0 \
                else self.user_param.get("w_loss",  self.user_param["w_earn"]) * rew_x - time_penalty
        
        elif state.truncated:
            return self.user.truncate_penalty
        
        return -state.perceived_elapsed_time * self.user_param["r_life"]




class PnCRewPercWeightV1(PnCEnvDefault):
    """
    To use this class, configure the observation to include target_rew_weighted.
    Include r_future in the reward parmaeter setting.
    """
    def _reward_function(self, state, action):
        if state.done:
            tct = state.click_moment / 1000
            target_reward = self.pnc_env.target_rew if state.click_success else -self.pnc_env.target_rew
            base_reward = BASE_SCORE / TOTAL_TIME_SEC * tct
            time_penalty = self.user_param["r_future"] * state.elapsed_time_step / 1000

            rew_x = target_reward + base_reward
            return self.user_param["money_weight"] * rew_x - time_penalty if rew_x >= 0 \
                else self.user_param.get("w_loss",  self.user_param["money_weight"]) * rew_x - time_penalty
        
        elif state.truncated:
            return self.user.truncate_penalty
        
        return -state.perceived_elapsed_time * self.user_param["r_future"]
    

class PnCRewPercWeightV2(PnCEnvDefault):
    """
    To use this class, configure the observation to include target_rew_weighted.
    Include r_future in the reward parmaeter setting.
    """
    def _reward_function(self, state, action):
        if state.done:
            tct = state.click_moment / 1000
            target_reward = self.pnc_env.target_rew if state.click_success else -self.pnc_env.target_rew
            base_reward = BASE_SCORE / TOTAL_TIME_SEC * tct
            time_penalty = self.user_param["r_future"] * state.click_moment / 1000

            rew_x = target_reward + base_reward
            return self.user_param["money_weight"] * rew_x - time_penalty if rew_x >= 0 \
                else self.user_param.get("w_loss",  self.user_param["money_weight"]) * rew_x - time_penalty
        
        elif state.truncated:
            return self.user.truncate_penalty
        
        return 0
    

class PnCRewPercWeightV3(PnCEnvDefault):
    def _reward_function(self, state, action):
        if state.done:
            tct = state.click_moment / 1000
            target_reward = self.pnc_env.target_rew if state.click_success else -self.pnc_env.target_rew
            base_reward = BASE_SCORE / TOTAL_TIME_SEC * tct
            time_penalty = self.user_param["init_stake_belief"] * state.click_moment / 1000

            rew_x = target_reward + base_reward - time_penalty
            return self.user_param["money_weight"] * rew_x if rew_x >= 0 \
                else self.user_param.get("w_loss",  self.user_param["money_weight"]) * rew_x
        
        elif state.truncated:
            return self.user.truncate_penalty
        
        return 0

    def _update_reward_cstate(self):
        self.cstate["target_rew"] = self.pnc_env.target_rew if \
            self.timer >= self.user_param.get("rew_proc_t", -1) else \
                self.user_param.get("init_stake_belief", self.pnc_env.target_rew)
        
        self.cstate["target_rew_weighted"] = self.pnc_env.target_rew * self.__rew_succ_scale() if \
            self.timer >= self.user_param.get("rew_proc_t", -1) else \
                self.user_param.get("r_future", self.pnc_env.target_rew * self.__rew_succ_scale())


# class PnCStepTimePenalty(PnCEnvDefault):
#     def _reward_function(self, state, action):
#         if state.done:
#             target_reward = max(0, self.user_param["target_stake"]) if state.click_success else -max(0, self.user_param["target_stake"])
            
#             # init_target_dist = np.linalg.norm(self.initial_env_cond["cursor_pos"] - self.initial_env_cond["target_pos"])
#             # click_reward = self.user_param.get("dist_reward", 0) * (init_target_dist - state.click_distance)
#             # tct = state.click_moment / 1000 + RING_TIMER * self.user_param.get("include_ring_time", 0)

#             base_reward = max(0, self.user_param["base_reward"])
#             time_penalty = -max(0, self.user_param["time_cost_weight"]) * state.elapsed_time_step / 1000
#             # dist_penalty = cd * self.user_param.get("dist_reward", 0)

#             return target_reward + base_reward + time_penalty # + click_reward
        
#         elif state.truncated:
#             return self.user.truncate_penalty
        
#         return -max(0, self.user_param["time_cost_weight"]) * self.intv.bump / 1000
    


# class PnCRewardPerception(PnCEnvDefault):
#     def _reward_function(self, state, action):
#         if state.done:
#             click_reward = self.pnc_env.target_rew if state.click_success else -self.pnc_env.target_rew
#             click_reward = np.clip(click_reward * self.user_param["money_weight"],
#                                    -self.user_param.get("max_rew", 100),
#                                    self.user_param.get("max_rew", 100))
#             base_reward = self.user_param["base_reward"]
#             money_reward = click_reward + base_reward
#             time_penalty = -self.user_param["time_cost_weight"] * state.elapsed_time_step / 1000
#             # time_penalty = -self.user_param["time_cost_weight"] * self.cstate["elapsed_time"]
            
#             return money_reward + time_penalty
        
#         elif state.truncated:
#             return self.user.truncate_penalty
        
#         return -self.user_param["time_cost_weight"] * state.perceived_elapsed_time


# class PnCRewardPerceptionAdvanced(PnCEnvDefault):
#     def _reward_function(self, state, action):
#         if state.done:
#             tct = self.cstate["elapsed_time"]

#             base_reward = BASE_SCORE / TOTAL_TIME_SEC / tct
#             target_incentive = self.pnc_env.target_rew if state.click_success else -self.pnc_env.target_rew
#             money_reward = self.user_param["k_incentive"] * (base_reward + target_incentive)

#             base_life = self.user_param["base_life"] / tct

#             # time_penalty = -self.user_param["base_future"] * state.elapsed_time_step / 1000
#             time_penalty = -self.user_param["base_future"] * tct
            
#             return money_reward - base_life + time_penalty
        
#         elif state.truncated:
#             return self.user.truncate_penalty
        
#         return 0 # -self.user_param["base_future"] * state.perceived_elapsed_time



# class PnCTimeHyperDiscount(PnCEnvDefault):
#     def _reward_function(self, state, action):
#         if state.done:
#             target_reward = max(0, self.user_param["target_stake"]) if state.click_success else -max(0, self.user_param["target_stake"])
#             base_reward = max(0, self.user_param["base_reward"])
#             # time_penalty = -max(0, self.user_param["time_cost_weight"]) * state.elapsed_time_step / 1000

#             tct = state.click_moment / 1000 + RING_TIMER * self.user_param.get("include_ring_time", 0)
#             time_discount = 1 + max(0, self.user_param["time_cost_weight"]) * tct
#             # dist_penalty = cd * self.user_param.get("dist_reward", 0)

#             return (target_reward + base_reward) / time_discount
        
#         elif state.truncated:
#             return self.user.truncate_penalty
        
#         return 0



        
if __name__ == "__main__":
    d = PnCEnvDefault()
    # print(PnCLinearTimeCost.__name__)
    # rt = list()
    # for _ in range(10000):
    #     d.reset()
    #     # rt.append(d.initial_env_cond[FIELD.HAND_RT])
    #     rt.append(d.np_z()[2])
    # # print(np.array(rt))    
    # import matplotlib.pyplot as plt
    # plt.hist(rt, bins=100)
    # plt.show()

    pass
