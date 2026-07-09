"""
Soft Actor Critic
for Point-and-Click agent training

Code written by June-Seop Yoon
"""
import os, psutil
from pathlib import Path

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from ..utils.myutils import get_compact_timestamp_str, save_dict_to_yaml
# from ..agent.agent_manager import PNC_ENV
from ..agent import pnc_agent
from ..config import config as config_pnc
from ..config import constant as constant_pnc
from .sac_policy import ModulatedSACPolicy
from .sac_logger import TensorboardStdCallback
from .parameter_eval_callback import ParameterTrackingEvalCallback

try:
    from ..agent_ans import ans_agent
    from ..config_ans import config as config_ans
    from ..config_ans import constant as constant_ans
except ModuleNotFoundError:
    ans_agent = None
    config_ans = None
    constant_ans = None

# tensorboard --logdir={tb log directory}
DIR_TO_DATA = Path(__file__).parent.parent.parent

# Normal RL with fixed parameters
class SACTrainer:
    def __init__(
        self,
        model_name_prefix=None,
        time_tag=True,
        num_cpu=16,
        train_config="default",
        env_class="default",
        env_setting=dict(
            agent_name="default",
            env_name="default",
            interval_name="default",
        )
    ):
        # Set model name
        if model_name_prefix is None:
            self.model_name = get_compact_timestamp_str()
        elif time_tag:
            self.model_name = f"{model_name_prefix}_{get_compact_timestamp_str()}"
        else:
            self.model_name = model_name_prefix
        
        if hasattr(pnc_agent, env_class):
            RLC = config_pnc.RLC
            FOLDER = constant_pnc.FOLDER
            if "env_name" in env_setting:
                env_setting["pnc_env_name"] = env_setting.pop("env_name")
            def make_class():
                class custom_env_class(getattr(pnc_agent, env_class)):
                    def __init__(self):
                        super().__init__(**env_setting)
                return custom_env_class

        elif ans_agent is not None and hasattr(ans_agent, env_class):
            RLC = config_ans.RLC
            FOLDER = constant_ans.FOLDER
            if "env_name" in env_setting:
                env_setting["game_env_name"] = env_setting.pop("env_name")
            def make_class():
                class custom_env_class(getattr(ans_agent, env_class)):
                    def __init__(self):
                        super().__init__(**env_setting)
                return custom_env_class
            
        else:
            raise RuntimeError(f"{env_class} not found on any env implementations.")
        
        # Configuration
        self.config = RLC[train_config]

        # Environments
        self.num_cpu = min(num_cpu, psutil.cpu_count(logical=False))
        env_fns = [make_class() for _ in range(self.num_cpu)]
        self.env = DummyVecEnv(env_fns) if self.num_cpu == 1 else SubprocVecEnv(env_fns, start_method="spawn")
        # Evaluation must not reuse the training environments: doing so alters
        # their state and makes per-episode parameter measurements ambiguous.
        def make_eval_env():
            return Monitor(make_class()())
        self.eval_env = (
            DummyVecEnv([make_eval_env])
            if self.num_cpu == 1
            else SubprocVecEnv([make_eval_env], start_method="spawn")
        )
        temp_env = make_class()()
        env_info = temp_env.env_setting_info
        env_info["env_class"] = env_class
        save_dict_to_yaml(
            temp_env.env_setting_info, 
            os.path.join(DIR_TO_DATA, f"data/{FOLDER.DATA_RL_MODEL}/{self.model_name}/env_info.yaml")
        )
        self.param_modul_length = len(temp_env.param_mod_z)
        
        self.cb_list = CallbackList([])

        # Some pre executions for directory
        os.makedirs(os.path.join(DIR_TO_DATA, f"data/{FOLDER.DATA_RL_MODEL}/{self.model_name}/{FOLDER.RL_TENSORBOARD}"), exist_ok=True)
        self.folder = FOLDER
        self.set_callbacks()
        self.set_model()

    
    def _select_device(self):
        device = self.config.model.get("device", "auto")
        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"


    def set_callbacks(self):
        # Callbacks
        checkpt_cb = CheckpointCallback(
            save_freq=max(1, self.config.callback.save_freq // self.num_cpu),
            save_path=os.path.join(DIR_TO_DATA, f"data/{self.folder.DATA_RL_MODEL}/{self.model_name}/{self.folder.MODEL_CHECKPT}"),
            save_replay_buffer=True
        )
        run_path = os.path.join(
            DIR_TO_DATA, f"data/{self.folder.DATA_RL_MODEL}/{self.model_name}"
        )
        eval_cb = ParameterTrackingEvalCallback(
            eval_env=self.eval_env,
            eval_freq=max(1, self.config.callback.eval_freq // self.num_cpu),
            best_model_save_path=os.path.join(run_path, self.folder.MODEL_BEST),
            log_path=os.path.join(run_path, "evaluation"),
            parameter_log_path=os.path.join(run_path, "evaluation", "parameters"),
            callback_after_eval=None, 
            # EvalCallback already uses a separate single environment. Keep the
            # requested sample size independent of the training worker count.
            n_eval_episodes=max(1, self.config.callback.eval_ep),
            verbose=0
        )
        std_cb = TensorboardStdCallback()
        self.cb_list = CallbackList([checkpt_cb, eval_cb, std_cb])
    

    def set_model(self):
        self.device = self._select_device()
        gradient_steps = self.config.model.get("gradient_steps", 1)
        if gradient_steps == "match_envs":
            gradient_steps = self.num_cpu
        gradient_steps = int(gradient_steps)
        self.model = SAC(
            ModulatedSACPolicy,
            self.env,
            tensorboard_log=os.path.join(DIR_TO_DATA, f"data/{self.folder.DATA_RL_MODEL}/{self.model_name}/{self.folder.RL_TENSORBOARD}"),
            verbose=0,
            batch_size=self.config.model.batch,
            ent_coef=self.config.model.entropy,
            learning_rate=self.config.model.lr,
            gamma=self.config.model.gamma,
            target_entropy=self.config.model.target_entropy,
            device=self.device,
            train_freq=(1, "step"),
            gradient_steps=gradient_steps,
            policy_kwargs={
                "net_arch": self.config.model.mlp.arch,
                "sim_param_dim": self.param_modul_length,
                "concat_layers": self.config.model.mlp.concat,       # 0 - Observation, 1~: Hidden layer
                "embed_net_arch": self.config.model.mlp.embed,     # Generate concat values through net
            }
        )


    def update_config(self, keys, value):
        key = keys[0]
        if len(keys) == 1:
            self.config[key] = value
            return
        self.update_config(keys[1:], value)

    
    def update_trainer(self):
        # Run this function after configuration complete
        self.set_callbacks()
        self.set_model()


    def load_checkpoint(self, model_path, replay_buffer_path=None):
        self.model = SAC.load(
            model_path,
            env=self.env,
            device=self.device,
            tensorboard_log=os.path.join(DIR_TO_DATA, f"data/{self.folder.DATA_RL_MODEL}/{self.model_name}/{self.folder.RL_TENSORBOARD}"),
        )
        if replay_buffer_path is not None:
            self.model.load_replay_buffer(replay_buffer_path)


    def run(self, reset_num_timesteps=True):
        # Save configuration and start training
        run_path = os.path.join(
            DIR_TO_DATA, f"data/{self.folder.DATA_RL_MODEL}/{self.model_name}"
        )
        save_dict_to_yaml(
            self.config, 
            os.path.join(run_path, "train_config.yaml")
        )
        self.model.learn(
            self.config.train.step,
            log_interval=self.config.train.log_freq,
            tb_log_name=self.model_name,
            callback=self.cb_list,
            progress_bar=True,
            reset_num_timesteps=reset_num_timesteps,
        )
        last_model_path = os.path.join(run_path, "last", "last_model.zip")
        os.makedirs(os.path.dirname(last_model_path), exist_ok=True)
        self.model.save(last_model_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Training option')

    parser.add_argument('--num_cpu', type=int, default=16)
    parser.add_argument('--env_class_type', type=str, default="PnCEnvDefault")
    parser.add_argument('--agent_type', type=str, default="default")
    parser.add_argument('--pnc_type', type=str, default="default")
    parser.add_argument('--rl_setting', type=str, default="simple")
    parser.add_argument('--model_name_prefix', type=str, default=None)
    parser.add_argument('--no_time_tag', action='store_true')
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--mlp_arch', type=str, default=None)
    parser.add_argument('--batch', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument(
        '--gradient_steps',
        type=str,
        default=None,
        help="SAC updates per rollout: an integer or 'match_envs'",
    )
    parser.add_argument('--save_freq', type=int, default=None)
    parser.add_argument('--eval_freq', type=int, default=None)
    parser.add_argument('--eval_ep', type=int, default=None)
    parser.add_argument('--load_model', type=str, default=None)
    parser.add_argument('--load_replay_buffer', type=str, default=None)
    parser.add_argument('--reset_num_timesteps', action='store_true')

    args = parser.parse_args()

    trainer = SACTrainer(
        model_name_prefix=args.model_name_prefix,
        time_tag=not args.no_time_tag,
        num_cpu=args.num_cpu,
        train_config=args.rl_setting,
        env_class=args.env_class_type,
        env_setting=dict(
            agent_name=args.agent_type,
            pnc_env_name=args.pnc_type,
        )
    )
    model_changed = False
    callback_changed = False

    if args.steps is not None:
        trainer.config.train.step = args.steps
    if args.device is not None:
        trainer.config.model.device = args.device
        model_changed = True
    if args.mlp_arch is not None:
        trainer.config.model.mlp.arch = [int(v.strip()) for v in args.mlp_arch.split(",") if v.strip()]
        trainer.config.model.mlp.concat = list(
            range(len(trainer.config.model.mlp.arch) + 1)
        )
        model_changed = True
    if args.batch is not None:
        trainer.config.model.batch = args.batch
        model_changed = True
    if args.lr is not None:
        trainer.config.model.lr = args.lr
        model_changed = True
    if args.gradient_steps is not None:
        trainer.config.model.gradient_steps = (
            "match_envs"
            if args.gradient_steps == "match_envs"
            else int(args.gradient_steps)
        )
        model_changed = True
    if args.save_freq is not None:
        trainer.config.callback.save_freq = args.save_freq
        callback_changed = True
    if args.eval_freq is not None:
        trainer.config.callback.eval_freq = args.eval_freq
        callback_changed = True
    if args.eval_ep is not None:
        trainer.config.callback.eval_ep = args.eval_ep
        callback_changed = True

    if model_changed or callback_changed:
        trainer.update_trainer()
    if args.load_model is not None:
        trainer.load_checkpoint(args.load_model, replay_buffer_path=args.load_replay_buffer)
    trainer.run(reset_num_timesteps=args.reset_num_timesteps or args.load_model is None)
    
