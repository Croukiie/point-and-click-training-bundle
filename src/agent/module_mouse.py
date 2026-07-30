"""
An implementation of "Point-and-click model"
from Seungwon et al., "A simulation model of intermittently controlled point-and-click behaviour." CHI 2021.

Original code: https://github.com/dodoseung/point-and-click-simulation-model
"""

import numpy as np
from scipy import interpolate
import os.path
from pathlib import Path

from ..utils.mouse_sensitivity import INCH_TO_METERS


rotation_map_rad = np.load(os.path.join(Path(__file__).parent.parent, "materials/rotation_map_rad.npy"))

class Mouse:
    interpolater = interpolate.RectBivariateSpline(np.arange(-1.5, 1.51, 0.1), 
                                                   np.arange(-1.5, 1.51, 0.1), 
                                                   rotation_map_rad[::-1,:])

    def get_hand_orientation(hand_loc, forearm_length=0.257):
        # hand location in physical surface (when neutral position x,y=0)

        # see surface_definition.png for more details on the definition of coordinate system
        hand_x = hand_loc[0]
        hand_y = hand_loc[1]

        normalized_hand_x = hand_x / forearm_length
        normalized_hand_y = hand_y / forearm_length

        # get hand orientation from rotation_map_rad.csv (matrix)
        # the matrix divides the surface into 31 by 31 grid, and for each grid point,
        # the matrix contains the expected hand orientation in rad (= the same as mouse rotation)
        # intermediate location should be interpolated from neighborhoods

        hand_orientation = Mouse.interpolater(normalized_hand_x, normalized_hand_y)

        return hand_orientation[0][0]

    @staticmethod
    def sample_realistic_orientation(
        normalized_hand_min=(-0.3, -0.3),
        normalized_hand_max=(0.3, 0.3),
        forearm_length=0.257,
    ):
        """Sample a plausible neutral mouse angle from the anatomical map.

        The rotation map used by the original point-and-click model describes
        the expected hand/mouse orientation over the reachable desk surface.
        Sampling its central region models small between-trial changes in grip
        and forearm placement without introducing implausible 360-degree
        rotations.
        """
        normalized_hand_loc = np.random.uniform(
            np.asarray(normalized_hand_min, dtype=float),
            np.asarray(normalized_hand_max, dtype=float),
        )
        return Mouse.get_hand_orientation(
            normalized_hand_loc * float(forearm_length),
            forearm_length=float(forearm_length),
        )


    def get_cursor_displacement(hand_start_loc, hand_end_loc, gain, user_forearm_length=0.257):

        # hand_start_loc, hand_end_loc = x,y location of hand on the mouse pad (a predefined surface)
        # hand_start_angle, hand_end_angle = upright angle as zero rad, clock-wise positive
        # gain = value of gain function (now simply assumed to be constant)

        current_hand_orientation = Mouse.get_hand_orientation(hand_end_loc, user_forearm_length)
        prev_hand_orientation = Mouse.get_hand_orientation(hand_start_loc, user_forearm_length)

        hand_displacement = np.linalg.norm(
            np.array(hand_end_loc) - np.array(hand_start_loc)
        )
        # net_hand_rotation = prev_hand_orientation - current_hand_orientation
        net_hand_rotation = current_hand_orientation - prev_hand_orientation
        if net_hand_rotation == 0:
            net_hand_rotation = np.finfo(float).tiny

        # get cursor displacement
        cursor_displacement = (
            gain * hand_displacement * 2 * np.sin(net_hand_rotation / 2)
        ) / net_hand_rotation

        # normalized hand velocity vector
        hand_displacement = (
            np.finfo(float).tiny if hand_displacement == 0 else hand_displacement
        )
        hand_dx = (hand_end_loc[0] - hand_start_loc[0]) / hand_displacement
        hand_dy = (hand_end_loc[1] - hand_start_loc[1]) / hand_displacement

        # resulting cursor dx and dy after passing through the mouse disturbance effect
        # The sensor-coordinate axes rotate with the mouse; the pointer displacement
        # induced by a desk-frame hand movement is rotated in the opposite direction.
        sensor_rotation = -(net_hand_rotation / 2 + prev_hand_orientation)
        cursor_dx = (
            hand_dx * np.cos(sensor_rotation)
            - hand_dy * np.sin(sensor_rotation)
        ) * cursor_displacement
        cursor_dy = (
            hand_dx * np.sin(sensor_rotation)
            + hand_dy * np.cos(sensor_rotation)
        ) * cursor_displacement

        return np.array([cursor_dx, cursor_dy])


    def get_cursor_displacement_from_desk_rotation(
        hand_start_loc,
        hand_end_loc,
        cursor_gain,
        mouse_orientation,
        rotation_gain_x_deg_per_cm=-3.0,
        rotation_gain_y_deg_per_cm=0.0,
        mouse_cpi=None,
        count_residual=None,
        return_count_residual=False,
    ):
        hand_delta = np.array(hand_end_loc) - np.array(hand_start_loc)
        rotation_delta = np.deg2rad(
            (rotation_gain_x_deg_per_cm * hand_delta[0]
             + rotation_gain_y_deg_per_cm * hand_delta[1]) * 100
        )
        next_mouse_orientation = mouse_orientation + rotation_delta
        next_residual = (
            np.zeros(2, dtype=float)
            if count_residual is None
            else np.asarray(count_residual, dtype=float)
        )

        if np.allclose(hand_delta, 0):
            result = (np.zeros(2), next_mouse_orientation)
            return (*result, next_residual) if return_count_residual else result

        if np.isclose(rotation_delta, 0):
            scale = 1.0
        else:
            scale = 2 * np.sin(rotation_delta / 2) / rotation_delta

        sensor_rotation = -(mouse_orientation + rotation_delta / 2)
        sensor_displacement = scale * Mouse.rot_mat(hand_delta, sensor_rotation)
        if mouse_cpi is not None:
            raw_count = (
                sensor_displacement * float(mouse_cpi) / INCH_TO_METERS
                + next_residual
            )
            quantized_count = np.rint(raw_count)
            next_residual = raw_count - quantized_count
            sensor_displacement = (
                quantized_count * INCH_TO_METERS / float(mouse_cpi)
            )

        cursor_delta = cursor_gain * sensor_displacement
        result = (cursor_delta, next_mouse_orientation)
        return (*result, next_residual) if return_count_residual else result

    @staticmethod
    def get_cursor_displacements_from_desk_rotation(
        hand_deltas,
        cursor_gain,
        mouse_orientation,
        rotation_gain_x_deg_per_cm=-3.0,
        rotation_gain_y_deg_per_cm=0.0,
        mouse_cpi=None,
        count_residual=None,
        return_count_residual=False,
    ):
        """Vectorized rotational mapping for one or more hand trajectories.

        ``hand_deltas`` has shape ``(..., n_segments, 2)``. Leading dimensions
        are treated as independent trajectories and are evaluated together.
        The returned orientations are the orientations *after* each segment.
        When ``mouse_cpi`` is set, physical sensor counts are rounded to
        integers while the sub-count residual is carried between segments.
        """
        hand_deltas = np.asarray(hand_deltas)
        initial_residual = (
            np.zeros(hand_deltas.shape[:-2] + (2,), dtype=float)
            if count_residual is None
            else np.asarray(count_residual, dtype=float)
        )
        if hand_deltas.shape[-2] == 0:
            empty_orientation = np.empty(
                hand_deltas.shape[:-2] + (0,), dtype=float
            )
            result = (hand_deltas.copy(), empty_orientation)
            return (*result, initial_residual) if return_count_residual else result

        rotation_deltas = np.deg2rad(
            (
                rotation_gain_x_deg_per_cm * hand_deltas[..., 0]
                + rotation_gain_y_deg_per_cm * hand_deltas[..., 1]
            )
            * 100
        )

        cumulative_rotation = np.cumsum(rotation_deltas, axis=-1)
        initial_orientation = np.asarray(mouse_orientation)
        next_orientations = initial_orientation[..., np.newaxis] + cumulative_rotation
        previous_orientations = next_orientations - rotation_deltas

        scales = np.ones_like(rotation_deltas, dtype=np.result_type(rotation_deltas, float))
        nonzero_rotation = ~np.isclose(rotation_deltas, 0)
        np.divide(
            2 * np.sin(rotation_deltas / 2),
            rotation_deltas,
            out=scales,
            where=nonzero_rotation,
        )

        sensor_rotations = -(previous_orientations + rotation_deltas / 2)
        sensor_displacements = (
            scales[..., np.newaxis]
            * Mouse.rot_mat(hand_deltas, sensor_rotations)
        )
        # Match the scalar implementation's ``np.allclose(hand_delta, 0)``
        # branch while retaining the orientation update for tiny segments.
        near_zero_displacement = np.all(np.isclose(hand_deltas, 0), axis=-1)
        sensor_displacements = np.where(
            near_zero_displacement[..., np.newaxis],
            0.0,
            sensor_displacements,
        )

        next_residual = initial_residual
        if mouse_cpi is not None:
            raw_counts = (
                sensor_displacements * float(mouse_cpi) / INCH_TO_METERS
            )
            cumulative_raw = (
                np.cumsum(raw_counts, axis=-2)
                + np.expand_dims(initial_residual, axis=-2)
            )
            cumulative_quantized = np.rint(cumulative_raw)
            previous_quantized = np.concatenate(
                (
                    np.zeros_like(cumulative_quantized[..., :1, :]),
                    cumulative_quantized[..., :-1, :],
                ),
                axis=-2,
            )
            quantized_counts = cumulative_quantized - previous_quantized
            next_residual = (
                cumulative_raw[..., -1, :]
                - cumulative_quantized[..., -1, :]
            )
            sensor_displacements = (
                quantized_counts * INCH_TO_METERS / float(mouse_cpi)
            )

        cursor_deltas = cursor_gain * sensor_displacements
        result = (cursor_deltas, next_orientations)
        return (*result, next_residual) if return_count_residual else result

    @staticmethod
    def rot_mat(p, rad):
        """Rotate one vector or an array of vectors around the last axis."""
        p = np.asarray(p)
        c, s = np.cos(rad), np.sin(rad)
        return np.stack(
            (c * p[..., 0] - s * p[..., 1], s * p[..., 0] + c * p[..., 1]),
            axis=-1,
        )
    

    def mouse_noise(vel, _hpos, mouse_gain, interval):
        # Cursor -> linear mapping -> Hand -> add rotational noise in this method
        dp = (vel[1:] + vel[:-1]) / 2 * interval / 1000
        hpos = _hpos.copy()
        # dx = (vel_x[1:] + vel_x[:-1]) / 2 * interval / 1000
        # dy = (vel_y[1:] + vel_y[:-1]) / 2 * interval / 1000
        hand_ori_prev = Mouse.get_hand_orientation(hpos)
        for idx in range(len(dp)):
            if np.linalg.norm(dp[idx]) > 0:
                hand_pos_prev = np.array(hpos)
                hand_dp = Mouse.rot_mat(dp[idx], -hand_ori_prev)
                hpos += hand_dp
                mouse_dp = Mouse.get_cursor_displacement(
                    hand_pos_prev,
                    hpos,
                    mouse_gain,
                )
                dp[idx] = mouse_dp

        return dp, hpos
    

if __name__ == "__main__":
    from ..agent.module_motor import Motor

    p, v = Motor.plan_hand_movement(np.zeros(2), np.zeros(2), np.array([-0.1, -0.12]), np.zeros(2), 1000, 1000, 50)
    dp, pn = Mouse.mouse_noise(v, p[0], 1, 50)

    import matplotlib.pyplot as plt

    plt.scatter(*p.T, s=1, color='k')
    
    pp = p[0] + np.cumsum(dp, axis=0)

    plt.scatter(*pp.T, s=1, color='r')
    plt.scatter(*pn, s=10, color='r')
    plt.gca().set_aspect('equal')
    plt.show()
