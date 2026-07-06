"""
Optimal Trajectory Generation

Reference:
Bye, Robin Trulssen. 
The BUMP model of response planning. Diss. Ph. D. Dissertation. 
Sydney, Australia: The University of New South Wales, 2009.

p.49 ~ 55
"""

import numpy as np
import os
from pathlib import Path

from tqdm import tqdm
from collections import defaultdict

from ..utils.myutils import pickle_load, pickle_save


def prepare_otg_calculation(time_unit, time_interval, sibling_folder='materials'):
    steps = time_interval // time_unit + 2
    mat_path = os.path.join(Path(__file__).parent.parent, f"{sibling_folder}/grams_{time_unit}_{time_interval}.pkl")

    if os.path.exists(mat_path):
        Gs, GTs, Grams, Gram_invs = pickle_load(mat_path)
    else:
        # Preparations for OTG
        tu = time_unit / 1000
        G = np.array([[1, tu], [0, 1]])
        H = np.array([[tu**2 / 2], [tu]])

        def get_Gs():
            Gs, GTs = [np.identity(2)], [np.identity(2)]
            for _ in range(1, steps):
                Gs.append(Gs[-1] @ G)
                GTs.append(GTs[-1] @ G.T)
            return Gs, GTs

        Gs, GTs = get_Gs()

        def get_gram(n, k):
            gram = np.zeros((2, 2))
            for j in range(0, k):
                gram += Gs[j] @ H @ H.T @ GTs[n - k + j]
            return gram

        def get_grammians():
            Grams, Gram_invs = defaultdict(dict), dict()
            for n in tqdm(range(steps)):
                for k in range(n + 1):
                    Grams[n][k] = get_gram(n, k)
                if n > 1:
                    Gram_invs[n] = np.linalg.inv(get_gram(n, n))
            return Grams, Gram_invs

        Grams, Gram_invs = get_grammians()

        pickle_save(mat_path, (Gs, GTs, Grams, Gram_invs))
    
    # Pack the triangular Gramian table once. Training calls OTG millions of
    # times, so indexing dense arrays is substantially cheaper than rebuilding
    # Python lists for every trajectory.
    Gs = np.asarray(Gs)
    GTs = np.asarray(GTs)
    Grams = {
        n: np.stack(
            [Grams[n][k] for k in range(n + 1)]
            if isinstance(Grams[n], dict)
            else Grams[n]
        )
        for n in range(len(Gs))
    }
    return Gs, GTs, Grams, Gram_invs

GRAMS = {
    1: prepare_otg_calculation(1, 500),
    5: prepare_otg_calculation(5, 500),
    10: prepare_otg_calculation(10, 500),
    25: prepare_otg_calculation(25, 500),
    50: prepare_otg_calculation(50, 2500),
}

# 1-dimensional OTG
def _otg_nd(p0, v0, p1, v1, time_unit:int, plan_duration:int, execute_duration:int):
    """Vectorized OTG over arbitrary trailing position dimensions."""
    Gs, _, Grams, Gram_invs = GRAMS[time_unit]
    pu = plan_duration // time_unit
    eu = execute_duration // time_unit

    x0 = np.stack((p0, v0), axis=0)
    x1 = np.stack((p1, v1), axis=0)

    g_n = Gs[pu]
    residual = x1 - np.tensordot(g_n, x0, axes=(1, 0))
    span = np.tensordot(Gram_invs[pu], residual, axes=(1, 0))

    g_states = np.tensordot(Gs[:eu + 1], x0, axes=(2, 0))
    controlled_states = np.tensordot(Grams[pu][:eu + 1], span, axes=(2, 0))
    return g_states + controlled_states


def otg_1d(p0, v0, p1, v1, time_unit:int, plan_duration:int, execute_duration:int):
    '''
    Generate min squared accel. trajectory.
    Initial pos/vel -> dest pos/vel
    
    Note that if plan duration is not completely divisble by time unit,
    return value could be slightly different from analytic computation.
    '''

    xs = _otg_nd(p0, v0, p1, v1, time_unit, plan_duration, execute_duration)
    return xs[:, 0], xs[:, 1]     # Interpolated position and velocity


def otg_2d(p0, v0, p1, v1, time_unit:int, plan_duration:int, execute_duration:int):
    """Two-dimensional OTG, optionally batched on leading dimensions."""
    xs = _otg_nd(p0, v0, p1, v1, time_unit, plan_duration, execute_duration)
    return xs[:, 0], xs[:, 1]


def add_motor_noise(p0, v, n_v, input_interval=1, noise_interval=10, ppd_ratio=0.192):
    noise_step = noise_interval // input_interval
    N = v.shape[0]
    
    v_nodes = v[::noise_step].copy()
    num_nodes = v_nodes.shape[0]
    
    nc = np.array([n_v, n_v * ppd_ratio])
    for i in range(1, num_nodes):
        curr_v = v_nodes[i]
        norm_v = np.linalg.norm(curr_v)
        if norm_v == 0:
            v_dir = np.zeros(2)
        else:
            v_dir = curr_v / norm_v
        v_per = np.array([-v_dir[1], v_dir[0]])
        noise = nc * norm_v * np.random.normal(0, 1, 2)
        v_nodes[i] += noise[0] * v_dir + noise[1] * v_per

    p_nodes = [np.array(p0)]
    for i in range(1, num_nodes):
        dt = noise_interval / 1000.0  # convert to seconds
        dp = (v_nodes[i-1] + v_nodes[i]) / 2 * dt
        p_nodes.append(p_nodes[-1] + dp)
    p_nodes = np.array(p_nodes)
    
    pos_segments = []
    vel_segments = []
    
    # Process full segments
    for seg in range(num_nodes - 1):
        pos_seg, vel_seg = otg_2d(
            p_nodes[seg], v_nodes[seg],
            p_nodes[seg+1], v_nodes[seg+1],
            input_interval, noise_interval, noise_interval
        )
        if seg < num_nodes - 2:
            pos_segments.append(pos_seg[:-1])
            vel_segments.append(vel_seg[:-1])
        else:
            pos_segments.append(pos_seg)
            vel_segments.append(vel_seg)
    
    # Concatenate all segments
    p_final = np.concatenate(pos_segments, axis=0)
    v_final = np.concatenate(vel_segments, axis=0)
    
    # If the total length differs from the input length N, adjust by trimming or padding.
    if p_final.shape[0] > N:
        p_final = p_final[:N]
        v_final = v_final[:N]
    elif p_final.shape[0] < N:
        pad_count = N - p_final.shape[0]
        # Repeat the last sample for padding.
        p_final = np.vstack([p_final, np.tile(p_final[-1], (pad_count, 1))])
        v_final = np.vstack([v_final, np.tile(v_final[-1], (pad_count, 1))])
    
    return p_final, v_final



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    e = list()
    for interval in [1, 5, 10, 25, 50]:
        p, v = otg_2d(np.zeros(2), np.zeros(2), np.ones(2), np.zeros(2), interval, 100, 100)
        effort = np.sum(np.linalg.norm(np.diff(v, axis=0), axis=1))
        e.append(effort)
    
    plt.plot([1, 5, 10, 25, 50], e)
    plt.xlabel("interval")
    plt.ylabel("Accel sum")
    plt.show()
