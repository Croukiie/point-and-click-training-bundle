import numpy as np
import cv2
from box import Box


PARAM_SYMBOL = dict(
    # theta_m=r"$\theta_m$",
    theta_m=r"$n_v$",
    theta_n=r"$n_p$",
    theta_s=r"$\theta_s$", 
    # theta_c=r"$\theta_c$",
    theta_c=r"$c_\sigma$",
    theta_p=r"$\theta_p$",
    c_mu=r"$c_\mu$",
    theta_r=r"$\theta_r$",

    nu=r"$\nu$",
    delta=r"$\delta$",

    rew_succ=r"$r_{succ}$",
    rew_fail=r"$r_{fail}$",
    decay_succ=r"$\lambda_{succ}$",
    decay_fail=r"$\lambda_{fail}$",

    time_penalty=r"$w_{time}$",
    effort_cost=r"$w_{effort}$",
    motor_effort=r"$r_{effort}$",

    rew_time=r"$r_{time}$",
    rew_decay=r"$\lambda_{succ}$",
    penalty_dist=r"$w_{dist}$",
    rew_dist=r"$d_{motive}$",

    rew_util_a=r"$a_{rew}$",
    time_util_a=r"$a_{time}$",

    prospect_a=r"$\alpha$",
    w_time_to_rew=r"$w_{time}$",
    prospect_b=r"$\beta$",
    prospect_l=r"$\lambda$",

    prospect_a_money=r"$\alpha_{stake}$",
    prospect_a_time=r"$\alpha_{time}$",
    w_time_util=r"$w_{time}$",

    stake_scaler=r"$k_{stake}$",
    prospect_a_stake=r"$\alpha_{stake}$",
    time_scaler=r"$k_{time}$",
    # prospect_a_time=r"$\alpha_{time}$",

    target_stake=r"$r_{stake}$",
    base_reward=r"$r_{base}$",
    time_cost_weight=r"$w_{time}$",
    dist_penalty=r"$w_{dist}$",
    dist_reward=r"$r_{dist}$",

    money_weight = r"$w_{stake}$",
    init_stake_belief = r"$r_{stake0}$",
    rew_proc_t = r"$\tau$",

    k_incentive = r"$k_{stake}$",
    base_life = r"$b_{life}$",
    base_future = r"$b_{future}$",

    w_earn = r"$w_{earn}$",
    w_loss = r"$w_{loss}$",
    w_time = r"$w_{time}$",
    r_life = r"$r_{life}$",
    r_future = r"$r_{future}$",

    fail_scale = r"$\lambda_{fail}$",

    mouse_rotation_gain_x=r"$k_{rot,x}$",
    mouse_rotation_gain_y=r"$k_{rot,y}$",
    mouse_plan_correction=r"$k_{corr}$",
    cpi=r"$CPI$",

    rew_succ_0=r"$r_{succ}^{0}$",
    rew_fail_0=r"$r_{fail}^{0}$",
)

PARAM_NAME = dict(
    theta_m = "motor noise",
    theta_c = "clock noise",
    c_mu = "implicit aim point",
    rew_succ = "success reward",
    time_penalty = "time penalty",
    motor_effort = "motor effort",
    rew_time = "time penalty",
    cpi = "mouse CPI",
)

CONST = Box(dict(
    DEFAULT_HEAD = np.array([0, 0, 0.63]),
    X = 0,
    Y = 1,
))

TARGET_SCORE = np.array([0, 0.1, 1, 10])      # Target reward
BASE_SCORE = 461                    # Base salary
TOTAL_TIME_SEC = 30 * 60            # Experiment time - 30 minutes
RING_TIMER = 0.8                    # Waiting for next trial - 0.8 sec
    
    
FOLDER = Box(dict(
    DATA_RL_MODEL = "pnc_model",
    MODEL_CHECKPT = "checkpoint",
    MODEL_BEST = "best",
    RL_TENSORBOARD = "tb_log",
    VISUALIZATION = "visualization",
    VIDEO = "video",
    GIF = "gif",

    AMORTIZER = "amortizer",
    AMORTIZER2 = "amortizer_cond",
    AMORTIZER3 = "amortizer_lv2cond"
    
))
   


FIELD = Box(dict(
    # Behavior summary
    TCT = "trial_completion_time",
    MVT = "movement_time",
    HIT = "target_hit",
    CLICK_DIST = "click_distance",
    CLICK_DIST_NORM = "click_distance_normalized",
    CLICK_ENDPT_X = "click_endpoint_x",
    CLICK_ENDPT_Y = "click_endpoint_y",
    TARGET_ENDPT_X = "target_endpoint_x",
    TARGET_ENDPT_Y = "target_endpoint_y",
    TARGET_DIST = "target_distance",
    TARGET_DIFF_ID = "target_index_of_difficulty",
    CLICK_REL_ENDPT_X = "click_rel_endpoint_x",
    CLICK_REL_ENDPT_Y = "click_rel_endpoint_y",
    TRUNC = "trauncated",
    HAND_RT = "hand_reaction_time",
    CLICK_PRD = "click_period",
    TRAV_LEN = "traverse_length",
    MAX_CURSOR_SPD = "max_cursor_speed",
    HAND_TRAV_LEN = "raw_path_length_m",
    MAX_HAND_SPD = "raw_max_speed_m_s",
    MOTOR_EFFORT = "motor_effort",
    CURSOR_TOTAL_ABS_TURN = "cursor_total_abs_turn_rad",
    CURSOR_NET_TURN = "cursor_net_turn_rad",
    CURSOR_MEAN_ABS_TURN = "cursor_mean_abs_turn_rad",
    CURSOR_P90_ABS_TURN = "cursor_p90_abs_turn_rad",
    CURSOR_TURN_ENERGY = "cursor_turn_energy_rad2",
    CURSOR_MAX_DEV_NORM = "cursor_max_deviation_normalized",
    CURSOR_PATH_EXCESS = "cursor_path_excess_ratio",
    CURSOR_EARLY_ABS_TURN = "cursor_early_mean_abs_turn_rad",
    CURSOR_MIDDLE_ABS_TURN = "cursor_middle_mean_abs_turn_rad",
    CURSOR_LATE_ABS_TURN = "cursor_late_mean_abs_turn_rad",

    # Initial task conditions
    WINDOW_WIDTH = "window_width",
    WINDOW_HEIGHT = "window_height",
    CURSOR_POS_X0 = "cursor_pos_x0",
    CURSOR_POS_Y0 = "cursor_pos_y0",
    TARGET_POS_X0 = "target_pos_x0",
    TARGET_POS_Y0 = "target_pos_y0",
    TARGET_VEL_X0 = "target_vel_x0",
    TARGET_VEL_Y0 = "target_vel_y0",
    TARGET_POS_X0_REL = "target_rel_pos_x0",
    TARGET_POS_Y0_REL = "target_rel_pos_y0",
    TARGET_REW = "target_rew",
    TARGET_RAD = "target_radius",
    HAND_POS_X0 = "hand_pos_x0",
    HAND_POS_Y0 = "hand_pos_y0",
    HAND_VEL_X0 = "hand_vel_x0",
    HAND_VEL_Y0 = "hand_vel_y0",
    MOUSE_ORI0 = "mouse_orientation_0",
    MOUSE_GAIN = "mouse_gain_value",


    # Trajectory or general field name
    TIMESTAMP = "timestamp",
    TIMEDIFF = "timediff",
    TIMERTFLAG = "timereactionflag",
    TARGET_POS_X = "target_pos_x",
    TARGET_POS_Y = "target_pos_y",
    TARGET_POS_X_REL = "target_rel_pos_x",
    TARGET_POS_Y_REL = "target_rel_pos_y",
    CURSOR_POS_X = "cursor_pos_x",
    CURSOR_POS_Y = "cursor_pos_y",
    HAND_POS_X = "hand_pos_x",
    HAND_POS_Y = "hand_pos_y",

    # SIMULATION-ONLY
    ACTION_KC = "action_kc",
))

NAME_ABBR = Box(dict(
    TCT = "TCT",
    MVT = "MVT",
    HIT = "ACC",
    CLICK_DIST = "CDIST",
    MAX_CURSOR_SPD = "Max C. SPD",
    ACTION_KC = 'Action Kc'
))

NAME_FULL = Box(dict(
    TCT = "Trial completion time",
    MVT = "Movement time",
    HIT = "Accuracy",
    CLICK_DIST = "Click distance",
    MAX_CURSOR_SPD = "Max cursor speed",
    ACTION_KC = 'Action Kc'
))
    


# R, G, B
COLOR = Box(dict(
    RED = (255, 0, 0),
    GREEN = (0, 255, 0),
    BLUE = (0, 0, 255),
    BLACK = (0, 0, 0),
    WHITE = (255, 255, 255),
    GRAY = (40, 40, 40),
    DARKGRAY = (20, 20, 20),
    SKYBLUE = (0, 176, 240),
    VIOLET = (232, 24, 163),
))
    

VIDEO = Box(dict(
    CODEC = cv2.VideoWriter_fourcc(*'X264'),
    FONT = cv2.FONT_HERSHEY_SIMPLEX,
))
    
