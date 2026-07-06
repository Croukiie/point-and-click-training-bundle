import numpy as np
from ..utils.otg import otg_2d

class Motor:
    def plan_hand_movement(
        p0, v0, p1, v1, 
        plan_duration:int, 
        execute_duration:int, 
        interval:int,
        maximum_hand_speed=10000
    ):
        """
        p0, v0 := initial state
        p1, v1 := destination state
        position unit: meter
        speed unit : meter/second
        duration and intervals are millisecond time (integer)
        """
        p, v = otg_2d(p0, v0, p1, v1, interval, plan_duration, execute_duration)
        s = np.linalg.norm(v, axis=1)

        # Check speed violation
        if np.any(s >= maximum_hand_speed):
            s_ratio = np.maximum(1.0, s / maximum_hand_speed)[:, np.newaxis]
            v_limited = v / s_ratio
            p_limited = p0 + np.cumsum(
                (v_limited[1:] + v_limited[:-1]) / 2 * interval / 1000,
                axis=0
            )
            p_limited = np.insert(p_limited, 0, p0, axis=0)

            return p_limited, v_limited
        return p, v


    def add_motor_noise(p0, v, interval, n_v, n_p=None, ppd_ratio=0.192):
        """
        Add motor noise to hand motor plan
        n_v := motor noise in parallel direction
        n_p := motor noise in perpendicular direction
        """
        if n_p is None:
            n_p = n_v * ppd_ratio
        
        v_noisy = np.copy(v)
        moving_v = v[1:]
        speeds = np.linalg.norm(moving_v, axis=1)
        directions = np.divide(
            moving_v,
            speeds[:, np.newaxis],
            out=np.zeros_like(moving_v),
            where=speeds[:, np.newaxis] != 0,
        )
        perpendiculars = np.stack((-directions[:, 1], directions[:, 0]), axis=1)
        noise = (
            np.random.normal(0, 1, size=(len(moving_v), 2))
            * speeds[:, np.newaxis]
            * np.array([n_v, n_p])
        )
        v_noisy[1:] += (
            noise[:, :1] * directions
            + noise[:, 1:] * perpendiculars
        )
        
        p_noisy = p0 + np.cumsum(
            (v_noisy[1:] + v_noisy[:-1]) / 2 * interval / 1000, 
            axis=0
        )
        p_noisy = np.insert(p_noisy, 0, p0, axis=0)

        return p_noisy, v_noisy


    def accel_sum(v):
        if v.shape[0] < 2: return 0
        return np.sum(np.linalg.norm(np.diff(v, axis=0), axis=1))


    def interp_segment(p0, v0, p1, v1, full_interval:int, interp_interval:int):
        return otg_2d(
            p0, v0, p1, v1,
            interp_interval,
            full_interval,
            full_interval
        )


    def interpolate_plan(traj_p, traj_v, original_interval:int, interp_interval:int):
        if traj_p.shape[0] < 2:
            return traj_p.copy(), traj_v.copy()

        # Evaluate every segment in one OTG call. Result shape is
        # (samples_per_segment, segment_count, 2).
        segment_p, segment_v = otg_2d(
            traj_p[:-1],
            traj_v[:-1],
            traj_p[1:],
            traj_v[1:],
            interp_interval,
            original_interval,
            original_interval,
        )
        interp_p = segment_p[1:].transpose(1, 0, 2).reshape(-1, 2)
        interp_v = segment_v[1:].transpose(1, 0, 2).reshape(-1, 2)
        return (
            np.concatenate((traj_p[:1], interp_p), axis=0),
            np.concatenate((traj_v[:1], interp_v), axis=0),
        )
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    p0 = np.zeros(2)
    v0 = np.zeros(2)
    pn = np.array([0.1, 0.1])
    vn = np.array([0.1, 0.1])

    p1, v1 = Motor.plan_hand_movement(p0, v0, pn, vn, 300, 300, 50)
    p2, v2 = Motor.plan_hand_movement(p0, v0, pn, vn, 300, 300, 25)
    p3, v3 = Motor.plan_hand_movement(p0, v0, pn, vn, 300, 300, 5)
    p4, v4 = Motor.plan_hand_movement(p0, v0, pn, vn, 300, 300, 1)
    p11, v11 = Motor.add_motor_noise(p1[0], v1, 50, 0.2)

    plt.scatter(np.arange(0, 301, 50), v1[:,0], color='k', s=10)
    plt.scatter(np.arange(0, 301, 50), v11[:,0], color='r', s=10)
    # plt.scatter(np.arange(0, 301, 25), v2[:,0], color='r', s=5)
    # plt.scatter(np.arange(0, 301, 5), v3[:,0], color='b', s=3)
    plt.scatter(np.arange(0, 301, 1), v4[:,0], color='g', s=1)

    # length = 500
    # interval = 50
    # small_intv = 5

    # p, v = Motor.plan_hand_movement(p0, v0, p1, v1, length, length, interval)
    # p2, v2 = Motor.plan_hand_movement(p0, v0, p1, v1, length, length, small_intv)
    # pl, vl = Motor.interpolate_plan(p, v, interval, small_intv)
    # # pp, vv = Motor.add_motor_noise(p[0], v, interval, 0.4)
    # # pp2, vv2 = Motor.interpolate_plan(pp, vv, interval, 1)

    # plt.scatter(np.arange(0, length+1, interval), v[:,0], marker='.', s=15, color='k', zorder=15)
    # plt.scatter(np.arange(0, length+1, small_intv), v2[:,0], marker='.', s=5, color='red', zorder=50)
    # plt.scatter(np.arange(0, length+1, small_intv), vl[:,0], marker='.', s=5, color='gray', zorder=51)
    # plt.scatter(np.arange(0, length+1, interval), np.linalg.norm(vv, axis=1), marker='s', s=5, color='blue')
    # plt.scatter(np.arange(0, length+1), np.linalg.norm(vv2, axis=1), marker='s', s=5, color='red')

    plt.show()
