"""
Point-and-Click task environment

Code written by June-Seop Yoon
"""

import numpy as np
from box import Box

from .module_mouse import Mouse
from ..config.config import ENV
from ..config.constant import CONST, TARGET_SCORE, BASE_SCORE
from ..utils.mymath import (
    cos_sin_array,
    symmetric_value
)

MOUSE_PAD_SIZE = 1.5    # Hand cannot exceed this area

class PnCGame:
    def __init__(self, config_name="default", config=None):
        """
        Distance unit: meter
        Speed unit: meter / second
        Direction angle unit: degree
        """
        config = ENV[config_name] if config is None else config
        self.mouse_gain = config.mouse_gain     # Hand coordinate (m) -> Monitor coordinate (m)
        self.mouse_mapping = config.get("mouse_mapping", "linear")
        self.mouse_rotation_gain_x = config.get("mouse_rotation_gain_x", -3.0)
        self.mouse_rotation_gain_y = config.get("mouse_rotation_gain_y", 0.0)
        self.mouse_plan_correction = config.get("mouse_plan_correction", 0.0)

        # Sample options
        self.window_width_range = (
            config.range.window_width.min, 
            config.range.window_width.max
        )
        self.target_rad_range = (
            config.range.target.radius.min,
            config.range.target.radius.max
        )
        self.target_spd_range = (
            config.range.target.speed.min,
            config.range.target.speed.max
        )
        self.target_id_range = (
            config.range.target.id.min,
            config.range.target.id.max
        )
        self.target_id_uniform_sample = config.option.id.uniform
        self.target_spd_min_sample_prob = config.option.speed.sample_min_prob
        self.target_min_dist_to_cursor = config.option.pos.min_dist_to_cursor

        # Main attributes
        self.window_ratio = ENV.window_ratio_height / ENV.window_ratio_width
        self.window_width = config.range.window_width.max
        self.window_height = self.window_width * self.window_ratio
        self.window_qt = np.array([self.window_width, self.window_height], dtype=np.float32) / 2

        self.cursor_pos = np.zeros(2)
        self.cursor_vel = np.zeros(2)
        self.hand_pos = np.zeros(2)
        self.hand_vel = np.zeros(2)
        self.mouse_ori = 0.0

        self.target_pos = np.zeros(2)
        self.target_spd = 0.0
        self.target_dir = 0.0     # 0 degree -> Left to Right. Counterclockwise rotation
        self.target_vel = self.target_spd * cos_sin_array(self.target_dir)
        self.target_rad = 0.0045
        self.target_rew = 0

        self.config = config

    
    def reset(
        self,
        window_width=None,
        cursor_pos=None,
        cursor_vel=None,
        hand_pos=None,
        hand_vel=None,
        target_pos=None,
        target_rad=None,
        target_spd=None,
        target_dir=None,
        target_rew=None,
        mouse_gain=None,
        max_target_pos_sample=100,
        return_reset_condition=True,
        **kwargs
    ):

        self.window_width = np.random.uniform(*self.window_width_range) if window_width is None else window_width
        self.window_height = self.window_width * self.window_ratio
        self.window_qt = np.array([self.window_width, self.window_height], dtype=np.float32) / 2

        self.cursor_pos = np.random.uniform(-self.window_qt, self.window_qt) if cursor_pos is None else np.array(cursor_pos)
        self.cursor_vel = np.zeros(2) if cursor_vel is None else np.array(cursor_vel)

        self.hand_pos = np.zeros(2) if hand_pos is None else np.array(hand_pos)
        self.hand_vel = np.zeros(2) if hand_vel is None else np.array(hand_vel)
        self.mouse_ori = 0.0

        if target_spd is not None: self.target_spd = target_spd
        else:
            if np.random.uniform(0, 1) < self.target_spd_min_sample_prob: self.target_spd = self.target_spd_range[0]
            else: self.target_spd = np.random.uniform(*self.target_spd_range)
        self.target_dir = np.random.uniform(0, 360) if target_dir is None else target_dir
        self._update_target_vel()


        # Target task difficulty
        if target_rad is None and target_pos is None:
            if self.target_id_uniform_sample:
                while True:
                    task_id = np.random.uniform(*self.target_id_range)
                    target_pos = np.random.uniform(-self.window_qt, self.window_qt)
                    target_dist = np.linalg.norm(target_pos - self.cursor_pos)
                    target_rad = target_dist / (2**task_id - 1) / 2
                    if np.all(target_pos >= -self.window_qt + target_rad) and \
                        np.all(target_pos <= self.window_qt - target_rad) and \
                            target_dist >= target_rad:
                        break
                self.target_pos = target_pos
                self.target_rad = target_rad
            else:
                self.target_rad = np.random.uniform(*self.target_rad_range)
                for _ in range(max_target_pos_sample):
                    target_pos = np.random.uniform(-self.window_qt + self.target_rad, self.window_qt - self.target_rad)
                    if np.linalg.norm(target_pos - self.cursor_pos) >= self.target_min_dist_to_cursor: break
                self.target_pos = target_pos
        else:
            self.target_rad = np.random.uniform(*self.target_rad_range) if target_rad is None else target_rad
            if target_pos is not None:
                self.target_pos = np.array(target_pos)
            else:
                for _ in range(max_target_pos_sample):
                    target_pos = np.random.uniform(-self.window_qt + self.target_rad, self.window_qt - self.target_rad)
                    if np.linalg.norm(target_pos - self.cursor_pos) >= self.target_min_dist_to_cursor: break
                self.target_pos = target_pos

        self.target_rew = np.random.choice(TARGET_SCORE) if target_rew is None else target_rew

        if mouse_gain is not None: self.mouse_gain = mouse_gain

        if return_reset_condition:
            return Box(dict(
                window_width = self.window_width,
                window_qt = self.window_qt,
                cursor_pos = self.cursor_pos,
                cursor_vel = self.cursor_vel,
                hand_pos = self.hand_pos,
                hand_vel = self.hand_vel,
                target_pos = self.target_pos,
                target_spd = self.target_spd,
                target_dir = self.target_dir,
                target_rad = self.target_rad,
                target_rew = self.target_rew,
                mouse_gain = self.mouse_gain,
            ))


    def update_target_move(self, delta_time, unit='s'):
        self.target_pos, _, self.target_dir = self.get_target_update(delta_time, unit=unit)
        self._update_target_vel()
    

    def get_target_update(self, delta_time, unit='s', target_pos=None, target_vel=None):
        dt = delta_time if unit == 's' else delta_time / 1000

        virt_target_pos = np.array(self.target_pos) if target_pos is None else np.array(target_pos)
        virt_target_vel = np.array(self.target_vel) if target_vel is None else np.array(target_vel)
        virt_target_spd, virt_target_dir = self._parse_spd_dir(virt_target_vel)

        future_target_pos = virt_target_pos + dt * virt_target_vel
        virt_target_dir = self._bounce_angle(future_target_pos, virt_target_dir, self.window_qt, self.target_rad)
        future_target_pos = np.array([
            self._bounce_coordinate(future_target_pos[CONST.X], self.window_qt[CONST.X], self.target_rad),
            self._bounce_coordinate(future_target_pos[CONST.Y], self.window_qt[CONST.Y], self.target_rad)
        ])
        # future_target_pos, virt_target_dir = self._compute_target_bounces(future_target_pos, virt_target_dir)
        
        return future_target_pos, virt_target_spd, virt_target_dir


    def _parse_spd_dir(self, vel):
        spd = np.linalg.norm(vel)
        dir = np.degrees(np.arctan2(vel[1], vel[0])) % 360
        return spd, dir
    

    def update_cursor_and_hand_move(self, htraj_p, htraj_v):
        (
            self.cursor_pos,
            self.cursor_vel,
            self.hand_pos,
            self.hand_vel,
            self.mouse_ori
        ) = self.get_cursor_and_hand_update(htraj_p, htraj_v, return_mouse_ori=True)



    def get_cursor_and_hand_update(
        self,
        htraj_p,
        htraj_v,
        cursor_pos=None,
        return_traj=False,
        return_mouse_ori=False,
    ):
        # We are assuming the infinite space for mouse pad
        # To limit the range of hand area,
        # modify the logics below.
        virt_hand_pos = self.hand_pos + htraj_p[-1] - htraj_p[0]
        virt_hand_vel = htraj_v[-1]
        virt_mouse_ori = self.mouse_ori

        hand_deltas = np.diff(htraj_p, axis=0)
        if self.mouse_mapping == "rotational":
            cursor_deltas, next_mouse_orientations = (
                Mouse.get_cursor_displacements_from_desk_rotation(
                    hand_deltas,
                    self.mouse_gain,
                    self.mouse_ori,
                    self.mouse_rotation_gain_x,
                    self.mouse_rotation_gain_y,
                )
            )
            cursor_velocities = self.mouse_gain * Mouse.rot_mat(
                htraj_v[1:], -next_mouse_orientations
            )
        elif self.mouse_mapping == "linear":
            cursor_deltas = self.mouse_gain * hand_deltas
            cursor_velocities = self.mouse_gain * htraj_v[1:]
            next_mouse_orientations = np.full(len(hand_deltas), self.mouse_ori)
        else:
            raise ValueError(f"Unknown mouse_mapping: {self.mouse_mapping}")

        # Cursor update
        if not return_traj:
            virt_cursor_pos = np.array(self.cursor_pos) if cursor_pos is None else np.array(cursor_pos)
            virt_cursor_vel = np.array(self.cursor_vel)

            for cursor_delta, cursor_vel, virt_mouse_ori in zip(
                cursor_deltas, cursor_velocities, next_mouse_orientations
            ):
                virt_cursor_pos = virt_cursor_pos + cursor_delta
                virt_cursor_pos, virt_cursor_vel = self._compute_cursor_blocked(virt_cursor_pos, cursor_vel)
        else:
            virt_cursor_pos = [np.array(self.cursor_pos)] if cursor_pos is None else [np.array(cursor_pos)]
            virt_cursor_vel = [np.array(self.cursor_vel)]
            virt_mouse_ori = [virt_mouse_ori]

            for cursor_delta, cursor_vel, next_mouse_ori in zip(
                cursor_deltas, cursor_velocities, next_mouse_orientations
            ):
                virt_cursor_pos_new = virt_cursor_pos[-1] + cursor_delta
                virt_cursor_pos_new, virt_cursor_vel_new = self._compute_cursor_blocked(virt_cursor_pos_new, cursor_vel)
                virt_cursor_pos.append(virt_cursor_pos_new)
                virt_cursor_vel.append(virt_cursor_vel_new)
                virt_mouse_ori.append(next_mouse_ori)
            
            virt_cursor_pos = np.array(virt_cursor_pos)
            virt_cursor_vel = np.array(virt_cursor_vel)
            virt_mouse_ori = np.array(virt_mouse_ori)

        if return_mouse_ori:
            return virt_cursor_pos, virt_cursor_vel, virt_hand_pos, virt_hand_vel, virt_mouse_ori
        return virt_cursor_pos, virt_cursor_vel, virt_hand_pos, virt_hand_vel


    def get_mouse_orientation_update(self, htraj_p):
        if self.mouse_mapping == "linear":
            return self.mouse_ori
        if self.mouse_mapping != "rotational":
            raise ValueError(f"Unknown mouse_mapping: {self.mouse_mapping}")

        hand_deltas = np.diff(htraj_p, axis=0)
        rotation_deltas = np.deg2rad(
            (
                self.mouse_rotation_gain_x * hand_deltas[:, 0]
                + self.mouse_rotation_gain_y * hand_deltas[:, 1]
            )
            * 100
        )
        return self.mouse_ori + rotation_deltas.sum()


    def _cursor_delta_for_hand_plan(self, htraj_p, mouse_ori):
        hand_deltas = np.diff(htraj_p, axis=0)
        if self.mouse_mapping == "linear":
            return self.mouse_gain * hand_deltas.sum(axis=0)
        if self.mouse_mapping != "rotational":
            raise ValueError(f"Unknown mouse_mapping: {self.mouse_mapping}")

        cursor_deltas, _ = Mouse.get_cursor_displacements_from_desk_rotation(
            hand_deltas,
            self.mouse_gain,
            mouse_ori,
            self.mouse_rotation_gain_x,
            self.mouse_rotation_gain_y,
        )
        return cursor_deltas.sum(axis=-2)


    def _hand_plan_endpoint_correction(self, ctraj_p, hand_pos, mouse_ori):
        intended_delta = ctraj_p[-1] - ctraj_p[0]
        if np.linalg.norm(intended_delta) == 0:
            return 0.0

        hand_segment_deltas = np.diff(ctraj_p, axis=0) / self.mouse_gain

        best_angle = 0.0
        search_width = np.pi / 4
        for _ in range(4):
            candidates = np.linspace(best_angle - search_width, best_angle + search_width, 15)
            rotated_hand_deltas = Mouse.rot_mat(
                hand_segment_deltas[np.newaxis, ...],
                mouse_ori + candidates[:, np.newaxis],
            )
            cursor_deltas, _ = Mouse.get_cursor_displacements_from_desk_rotation(
                rotated_hand_deltas,
                self.mouse_gain,
                mouse_ori,
                self.mouse_rotation_gain_x,
                self.mouse_rotation_gain_y,
            )
            actual_deltas = cursor_deltas.sum(axis=-2)
            errors = np.linalg.norm(actual_deltas - intended_delta, axis=-1)
            best_angle = candidates[np.argmin(errors)]
            search_width /= 4
        return best_angle


    def _cursor_delta_from_hand_segment(self, hand_start, hand_end, mouse_ori):
        if self.mouse_mapping == "rotational":
            return Mouse.get_cursor_displacement_from_desk_rotation(
                hand_start,
                hand_end,
                self.mouse_gain,
                mouse_ori,
                self.mouse_rotation_gain_x,
                self.mouse_rotation_gain_y,
            )
        if self.mouse_mapping == "linear":
            return self.mouse_gain * (hand_end - hand_start), mouse_ori
        raise ValueError(f"Unknown mouse_mapping: {self.mouse_mapping}")
    

    def _cursor_velocity_from_hand_velocity(self, hand_vel, mouse_ori):
        if self.mouse_mapping == "rotational":
            return self.mouse_gain * Mouse.rot_mat(hand_vel, -mouse_ori)
        if self.mouse_mapping == "linear":
            return self.mouse_gain * hand_vel
        raise ValueError(f"Unknown mouse_mapping: {self.mouse_mapping}")
    

    def _hand_delta_from_cursor_delta(self, cursor_delta):
        if self.mouse_mapping == "rotational":
            return Mouse.rot_mat(cursor_delta / self.mouse_gain, self.mouse_ori)
        if self.mouse_mapping == "linear":
            return cursor_delta / self.mouse_gain
        raise ValueError(f"Unknown mouse_mapping: {self.mouse_mapping}")
    

    def _hand_velocity_from_cursor_velocity(self, cursor_vel):
        if self.mouse_mapping == "rotational":
            return Mouse.rot_mat(cursor_vel / self.mouse_gain, self.mouse_ori)
        if self.mouse_mapping == "linear":
            return cursor_vel / self.mouse_gain
        raise ValueError(f"Unknown mouse_mapping: {self.mouse_mapping}")
    

    def convert_cursor_plan_to_hand_plan(
        self,
        ctraj_p,
        ctraj_v,
        hand_pos=None,
        lead_in_htraj_p=None,
        interval=None,
    ):
        if hand_pos is None:
            hand_pos = self.hand_pos

        if self.mouse_mapping == "rotational":
            theta = self.get_mouse_orientation_update(lead_in_htraj_p) \
                if lead_in_htraj_p is not None else self.mouse_ori
            if self.mouse_plan_correction > 0:
                correction = self._hand_plan_endpoint_correction(ctraj_p, hand_pos, theta)
                theta += np.clip(self.mouse_plan_correction, 0.0, 1.0) * correction
            htraj_p = hand_pos + Mouse.rot_mat(
                ctraj_p - ctraj_p[0], theta
            ) / self.mouse_gain
            htraj_v = Mouse.rot_mat(ctraj_v, theta) / self.mouse_gain
            return htraj_p, htraj_v

        htraj_p = hand_pos + (ctraj_p - ctraj_p[0]) / self.mouse_gain
        htraj_v = ctraj_v / self.mouse_gain
        return htraj_p, htraj_v


    def _compute_cursor_blocked(self, cursor_pos, cursor_vel):
        cursor_range_flag = self._get_range_flag(cursor_pos, -self.window_qt, self.window_qt)
        virt_cursor_pos = np.clip(cursor_pos, -self.window_qt, self.window_qt)
        # Reduce velocity on case of wall-collision or wall-stuck
        virt_cursor_vel = cursor_vel.copy()
        virt_cursor_vel[cursor_range_flag != 0] = 0
        return virt_cursor_pos, virt_cursor_vel


    def replay_and_save(
        self, 
        htraj_p, htraj_v, interval, unit='s', 
        keys=["cursor_pos", "target_pos", "hand_pos", "cursor_vel", "target_vel", "hand_vel"]
    ):
        state = self.get_current_state()
        log = {key: [state[key]] for key in keys}

        for i in range(1, len(htraj_p)):
            self.update_target_move(interval, unit=unit)
            self.update_cursor_and_hand_move(htraj_p[i-1:i+1], htraj_v[i-1:i+1])
            state = self.get_current_state()
            for key in keys:
                log[key].append(state[key])
        
        for key in keys:
            log[key] = np.array(log[key])
        
        return log

    
    def get_required_hand_movement(self, cursor_pos=None, target_pos=None, target_vel=None):
        if cursor_pos is None: cursor_pos = self.cursor_pos
        if target_pos is None: target_pos = self.target_pos
        if target_vel is None: target_vel = self.target_vel

        return (
            self._hand_delta_from_cursor_delta(target_pos - cursor_pos),
            self._hand_velocity_from_cursor_velocity(target_vel)
        )

    
    def get_current_state(self):
        return dict(
            cursor_pos = self.cursor_pos,
            cursor_vel = self.cursor_vel,
            hand_pos = self.hand_pos,
            hand_vel = self.hand_vel,
            target_pos = self.target_pos,
            target_pos_rel = self.target_pos - self.cursor_pos,
            target_vel = self.target_vel,
            target_spd = self.target_spd,
            target_dir = self.target_dir,
            target_rad = self.target_rad,
            target_rew = self.target_rew,
            window_qt = self.window_qt,
            mouse_gain = self.mouse_gain,
            mouse_ori = np.array([np.cos(self.mouse_ori), np.sin(self.mouse_ori)]),
        )
    
    def get_cursor_target_distance(self):
        return np.linalg.norm(self.cursor_pos - self.target_pos)

    
    def sample_task_conditions(self, n, fix_cond_list=list(), **cond):
        if not len(fix_cond_list):
            return [self.reset(return_reset_condition=True, **cond) for _ in range(n)]
        
        cond_dict = cond.copy()
        one_reset_for_fixing_condition = self.reset(return_reset_condition=True, **cond)
        for cond_key in fix_cond_list:
            cond_dict[cond_key] = one_reset_for_fixing_condition[cond_key]
        return [self.reset(return_reset_condition=True, **cond_dict) for _ in range(n)]



    def _check_target_bounce(self, future_pos):
        return self._get_range_flag(future_pos, -self.window_qt + self.target_rad, self.window_qt - self.target_rad)

    def _update_target_vel(self):
        self.target_vel = self.target_spd * cos_sin_array(self.target_dir)

    def _get_range_flag(self, x, a, b):
        # -1 := x < a, 0 := a < x < b, 1 := b < x
        return np.array(x > b, dtype=int) - np.array(x < a, dtype=int)

    
    def _bounce_coordinate(self, point_x, wall_x_half, target_rad):
        w = wall_x_half - target_rad
        # point_x should range in [-wall_x, wall_x]
        new_x = (point_x + 2 * w) % (4 * w) - 2 * w
        if -w <= new_x <= w:
            return new_x
        if w < new_x:
            return 2 * w - new_x
        if -w > new_x:
            return -2 * w - new_x
    
    def _bounce_angle(self, point, dir_angle, window_qt, target_rad):
        a = dir_angle
        w = window_qt - target_rad
        pt = ((np.abs(point) + w) // (2 * w) % 2).astype(int)
        if pt[CONST.X] == 1:
            a = symmetric_value(a, 90) % 360
        if pt[CONST.Y] == 1:
            a = symmetric_value(a, 0) % 360
        return a





if __name__ == "__main__":
    import matplotlib.pyplot as plt

    g = PnCGame(config_name="static_target_window")

    # task_list = g.sample_task_conditions(10, fix_cond_list=["target_rew"])
    # for t in task_list:
    #     print(t["target_rew"])

    tid = list()
    for _ in range(10000):
        c = g.reset()
        cp = c.cursor_pos
        tp = c.target_pos
        tr = c.target_rad

        # tid.append(np.log2(np.linalg.norm(cp - tp) / 2 / tr + 1))
        tid.append(tr)

    tid = np.array(tid)

    plt.hist(tid, bins=20)
    plt.show()
