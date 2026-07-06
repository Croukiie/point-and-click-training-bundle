"""
Shoot Action Module
Intermittent Click Planing (ICP) model

Reference:
1) "An Intermittent Click Planning Model", Eubnji Park and Byungjoo Lee
https://dl.acm.org/doi/abs/10.1145/3313831.3376725

Code modified by June-Seop Yoon
Original code was written by Seungwon Do (dodoseung)
"""

import numpy as np
from ..utils.mymath import distance_point2line

class Click:
    def intermittent_click_planning(ctraj, tpos, trad, time_interval, 
                                    p, c_sigma, c_mu, nu, delta, 
                                    t_min=1, t_max=np.inf, offset=100, allow_sample=1000):
        # Assumption: end of ctraj == tpos
        dist_traj = np.linalg.norm(ctraj - tpos, axis=1)
        
        # c_mu 0 := touch, 0.5 := center of the target
        dist_flag = (dist_traj <= trad * (1 - 2 * c_mu)).astype(int)
        if np.any(dist_flag):   # 1 exist in dist_flag
            click_moment_true = np.argmax(dist_flag) * time_interval # np.argmax returns first index of 1 (touch)
        else:
            click_moment_true = dist_flag.shape[0] * time_interval  # c_mu = 0.5, end of the traj = tpos
        click_moment_true += offset

        # compute t_cue := from start to target touch
        dist_flag = (dist_traj <= trad).astype(int)
        touch_moment = np.argmax(dist_flag) * time_interval if np.any(dist_flag) else dist_flag.shape[0] * time_interval
        touch_moment += offset
        touch_moment /= 1000    # time_interval is millisec unit. Convert from ms to sec.

        sig_t = c_sigma * (p / 1000)
        sig_v = delta + 1/(np.exp(np.clip(nu * touch_moment, 1e-6, 10)) - 1)
        sig = np.sqrt(sig_t**2 * sig_v**2 / (sig_t**2 + sig_v**2))

        for _ in range(allow_sample):
            click_timing_noise = round(np.random.normal(0, sig) * 1000)
            if t_min <= click_moment_true + click_timing_noise <= t_max:
                return click_moment_true + click_timing_noise
        
        raise RuntimeError(f"Possibly click timing sampling fallen into infinite loop!: {click_moment_true}, {touch_moment}, {sig_t:.2f}, {sig_v:.2f}, {sig:.2f}")


    # def intermittent_click_planning(ctraj, tpos, trad, interval, theta_c, c_mu, click_offset=0):
    #     # Assumption: end of ctraj == tpos
    #     dist_flag = (np.linalg.norm(ctraj - tpos, axis=1) <= trad * np.clip(1 - 2 * c_mu, 0, 1)).astype(int)
    #     # end_time = len(dist_flag) * interval

    #     # c_mu 0 := touch, 0.5 := center of the target
    #     if np.any(dist_flag):   # 1 exist in dist_flag
    #         click_moment_true = np.argmax(dist_flag) * interval # np.argmax returns first index of 1 (touch)
    #         # t_penetrate = end_time - t_touch
    #     else:
    #         click_moment_true = len(dist_flag) * interval
    #         # t_penetrate = 0

    #     # click_moment_true = t_touch + t_penetrate * (c_mu * 2)
    #     while True:
    #         clock_noise = np.random.normal(1, theta_c)
    #         if 0.01 <= clock_noise <= 2: break
    #     return round(clock_noise * (click_moment_true + click_offset))


    def sample_shoot_timing(cpos, tpos, cvel, tvel, trad, params, t_min=1, t_max=np.inf):
        vel = cvel - tvel
        dist = np.linalg.norm(cpos - tpos)
        spd = np.linalg.norm(vel)

        if spd == 0:
            return Click.ICP_model(0, 1e-6, params)
        else:
            dist_tar2ctraj = distance_point2line(tpos, cpos, vel)
            if dist_tar2ctraj <= trad:
                tc = 1e-6
                value = np.sqrt(trad**2 - dist_tar2ctraj**2)
                bot = np.sqrt(dist**2 - dist_tar2ctraj**2)
                # Compute the relative position of crosshair by perpendicular line
                if (-tvel).dot(cpos-tpos) > 0:
                    inter_len = value - bot
                else:
                    inter_len = value + bot
            else:
                if dist_tar2ctraj > trad: 
                    inter_len = 0
                else: 
                    inter_len = 2*np.sqrt(trad**2 - dist_tar2ctraj**2)
                
                tc = (np.sqrt(dist**2 - dist_tar2ctraj**2) - inter_len/2) / spd
            
            Wt = inter_len / spd
            return Click.ICP_model(Wt, tc, params, _min=t_min, _max=t_max)
        

    def ICP_model(Wt, tc, params, _min=1, _max=np.inf):
        m = params["icp_cmu"] * Wt
        new_tc = tc + m
        # sig_t = params["theta_c"] * new_tc
        sig_v = params["theta_c"] * (1/(np.exp(np.clip(new_tc * params["icp_nu"], 1e-5, 10)) - 1) + params["icp_delta"])
        # sigma = np.sqrt(sig_t**2 * sig_v**2 / (sig_t**2 + sig_v**2))

        while True:
            click_timing = round(np.random.normal(m, sig_v) * 1000)
            if _min < click_timing < _max:
                return click_timing