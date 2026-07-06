"""
Visual Perception Module

1) Perceiving target position and velocity on monitor

Reference:
1) "Noise characteristics and prior expectations in human visual speed perception"
https://www.nature.com/articles/nn1669
2) "Estimation of cortical magnification from positional error
in normally sighted and amblyopic subjects"
https://jov.arvojournals.org/article.aspx?articleid=2213265

Code modified by June-Seop Yoon
"""

import numpy as np

from ..config.constant import CONST
from ..utils.mymath import angle_between, cos_sin_array


class Perceive:
    def position_perception(pos, noise, window_qt, gaze_dist=0.05, head=CONST.DEFAULT_HEAD):
        while True:
            pos_noise = pos + cos_sin_array(np.random.uniform(0, 360)) * np.random.normal(0, gaze_dist * noise)
            if np.all(pos_noise >= -window_qt) and np.all(pos_noise <= window_qt):
                break
        return pos_noise


    def speed_perception(vel, pos, sigma_v, head=CONST.DEFAULT_HEAD, s0=0.3):
        """
        pos, vel := 2D state on the monitor
        """
        p0 = np.array([*pos, 0])
        p1 = np.array([*(pos + vel), 0])
        spd = np.linalg.norm(vel)
        if spd <= 0: return vel

        # Angle between objective movement and head to objective
        k = np.sin(angle_between(p1 - p0, head - p0, return_in_degree=False))
        d = np.linalg.norm(head - p0)

        # Compute d(a2)/dt and convert to monitor velocity
        vis_spd = spd * k / d
        vis_spd_hat = _sample_valid_s(vis_spd, sigma_v, s0=s0)
        spd_hat = vis_spd_hat * d / k

        return (spd_hat / spd) * vel

    def timing_perception(t, c_sigma):
        while True:
            clock_noise = np.random.normal(1, c_sigma)
            if 0.01 < clock_noise < 2: break
        return t * clock_noise



def _sample_valid_s(s, sigma_v, s0=0.3):
    while True:
        s_final = s0 * (np.random.lognormal(np.log(1 + s / s0), sigma_v) - 1)
        if s_final >= 0: return s_final


if __name__ == "__main__":
    pos = np.zeros(2)
    vel = np.array([0.2, 0])

    pos_hat = np.array([
        Perceive.position_perception(pos, 0.4, np.array([0.3, 0.3 * 9 / 16])) for _ in range(10000)
    ])

    import matplotlib.pyplot as plt

    plt.hist(pos_hat[:,0], bins=100, histtype='step')
    plt.hist(pos_hat[:,1], bins=100, histtype='step')
    plt.show()