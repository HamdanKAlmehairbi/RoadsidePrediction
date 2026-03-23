import seal.trainer.defaults as defaults
import os
import pickle
import ray

from abc import ABC, abstractmethod
from collections import defaultdict
from pandas import DataFrame
from seal.logging import *
from ray.rllib.algorithms.ppo import PPO, PPOConfig, PPOTorchPolicy
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.policy import Policy
from time import ctime
from typing import Any, Callable, Dict, List, Tuple

from seal.trainer.counter import Counter
from seal.trainer.defaults import *
from seal.trainer.util import *
from seal.sumo.abstract_env import AbstractSumoEnv

RAY_TRAINER_SEED = 54321


class BaseTrainer(ABC):

    communication_callback_cls: DefaultCallbacks
    counter: Counter
    idx: int
    num_gpus: int
    env: AbstractSumoEnv
    learning_rate: float
    log_level: str
    gamma: float
    num_gpus: int
    num_workers: int
    out_checkpoint_dir: str
    out_data_dir: str
    out_weights_dir: str
    policy: str
    policy_mapping_fn: Callable
    policy_type: type
    algorithm_cls: type

    def __init__(
            self,
            checkpoint_freq: int = 5,
            env: AbstractSumoEnv = None,
            gamma: float = 0.95,
            learning_rate: float = 0.001,
            log_level: str = "ERROR",
            model_name: str = None,
            num_gpus: int = 0,
            num_workers: int = 0,
            root_dir: List[str] = ["out", "SMARTCOMP"],
            sub_dir: str = None,
            policy: str = "ppo",
            out_prefix: str = None,
            trainer_kwargs: dict = None,
            **kwargs
    ) -> None:
        assert 0 <= gamma <= 1
        self.communication_callback_cls = None
        self.checkpoint_freq = checkpoint_freq
        self.counter = Counter()
        self.env = env
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.log_level = log_level
        self.model_name = model_name
        self.num_gpus = num_gpus
        self.num_workers = num_workers

        self.out_checkpoint_dir = os.path.join(*(root_dir + ["checkpoints"]))
        self.out_data_dir = os.path.join(*(root_dir + ["data"]))
        self.out_weights_dir = os.path.join(*(root_dir + ["weights"]))
        if sub_dir is not None:
            self.out_checkpoint_dir = os.path.join(
                self.out_checkpoint_dir, sub_dir)
            self.out_data_dir = os.path.join(self.out_data_dir, sub_dir)
            self.out_weights_dir = os.path.join(self.out_weights_dir, sub_dir)

        self.gui = kwargs.get("gui", defaults.GUI)
        self.net_file = kwargs.get("net_file", defaults.NET_FILE)
        self.ranked = kwargs.get("ranked", defaults.RANKED)
        self.rand_routes_on_reset = kwargs.get("rand_routes_on_reset",
                                               defaults.RAND_ROUTES_ON_RESET)
        self.rand_routes_config = kwargs.get("rand_routes_config",
                                             defaults.RAND_ROUTES_CONFIG)
        self.alpha = kwargs.get("alpha", 1.0)
        self.time_of_day = kwargs.get("time_of_day", False)
        self.use_time_encoding = kwargs.get("use_time_encoding", False)

        self.out_prefix = out_prefix
        self.net_dir = self.net_file.split(os.sep)[-1].split(".")[0]
        self.out_checkpoint_dir = os.path.join(
            self.out_checkpoint_dir, self.net_dir)
        self.out_data_dir = os.path.join(self.out_data_dir, self.net_dir)
        self.out_weights_dir = os.path.join(self.out_weights_dir, self.net_dir)

        if not os.path.isdir(self.out_checkpoint_dir):
            os.makedirs(os.path.join(self.out_checkpoint_dir))
        if not os.path.isdir(self.out_data_dir):
            os.makedirs(os.path.join(self.out_data_dir))
        if not os.path.isdir(self.out_weights_dir):
            os.makedirs(os.path.join(self.out_weights_dir))

        self.policy = policy
        self.__load_policy_type()

        self.trainer_name = None
        self.idx = None
        self.policy_config = None
        self.policy_mapping_fn = None
        self.trainer_kwargs = trainer_kwargs

    # ------------------------------------------------------------------------- #

    def load(self, checkpoint: str) -> None:
        if type(self) is BaseTrainer:
            raise NotImplementedError("Cannot load policy using abstract `BaseTrainer` "
                                      "class.")
        self.on_setup()
        self.ray_trainer.restore(str(checkpoint))

    # ------------------------------------------------------------------------- #

    def train(self, num_rounds: int, save_on_end: bool = True, **kwargs) -> DataFrame:
        if kwargs.get("checkpoint", None) is not None:
            self.load(kwargs["checkpoint"])
        else:
            self.policies = self.on_policy_setup()
            if GLOBAL_POLICY_VAR in self.policies:
                raise ValueError(f"Sub-classes of `BaseTrainer` cannot have "
                                 f"policies with key '{GLOBAL_POLICY_VAR}'.")
            else:
                temp = next(iter(self.policies.values()))
                self.policies[GLOBAL_POLICY_VAR] = temp
            self.on_setup()
        for r in range(num_rounds):
            self._round = r
            self._result = self.ray_trainer.train()
            self.on_data_recording_step()
            self.on_logging_step()
            if r % self.checkpoint_freq == 0:
                self.ray_trainer.save(self.model_path)
            self.save_test_policy()
        # Set the global test policy that will be used for evaluation.
        weights = self.save_test_policy()
        self.ray_trainer.get_policy(GLOBAL_POLICY_VAR).set_weights(weights)
        # Get the data from the training process and output it for visualization to see
        # how training performed over time.
        dataframe = self.on_tear_down()
        if save_on_end:
            path = os.path.join(self.out_data_dir, self.get_filename())
            try:
                dataframe.to_csv(f"{path}.csv")
                dataframe.to_excel(f"{path}.xlsx")
                dataframe.to_json(f"{path}.json")
            except FileNotFoundError:
                new_dir = os.path.join(path.split(os.sep[:-1]))
                os.makedirs(new_dir)
                dataframe.to_csv(f"{path}.csv")
                dataframe.to_excel(f"{path}.xlsx")
                dataframe.to_json(f"{path}.json")
        return dataframe

    # ------------------------------------------------------------------------- #

    def __load_policy_type(self) -> None:
        if self.policy == "ppo":
            self.algorithm_cls = PPO
            self.policy_type = PPOTorchPolicy
        else:
            raise NotImplementedError(f"Do not support policies for `{self.policy}`. "
                                      f"Only 'ppo' is supported.")

    # ------------------------------------------------------------------------- #

    def init_config(self) -> PPOConfig:
        config = (
            PPOConfig()
            .api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False,
            )
            .environment(
                env=self.env,
                env_config=self.env_config_fn(),
            )
            .framework("torch")
            .debugging(log_level=self.log_level, seed=RAY_TRAINER_SEED)
            .training(lr=self.learning_rate, gamma=self.gamma)
            .multi_agent(
                policies=self.policies,
                policy_mapping_fn=self.policy_mapping_fn,
            )
            .resources(num_gpus=self.num_gpus)
            .env_runners(num_env_runners=self.num_workers)
            .callbacks(self.communication_callback_cls)
        )
        if self.trainer_kwargs is not None:
            config = config.update_from_dict(self.trainer_kwargs)
        return config

    def env_config_fn(self) -> Dict[str, Any]:
        return {
            "gui": self.gui,
            "net-file": self.net_file,
            "rand_routes_on_reset": self.rand_routes_on_reset,
            "ranked": self.ranked,
            "use_dynamic_seed": True,
            "alpha": self.alpha,
            "time_of_day": self.time_of_day,
            "use_time_encoding": self.use_time_encoding,
        }

    def save_test_policy(self) -> Weights:
        # Get the global test policy weights and then save them to a PICKLE file object.
        # This will then be used to reload the test policy's weights for evaluation
        # in both the synthetic simulations and real-world implementation.
        weights = self.on_make_final_policy()
        ranked_str = "ranked" if self.ranked else "unranked"
        if self.out_prefix is not None:
            ranked_str = f"{self.out_prefix}_{ranked_str}"
        with open(os.path.join(self.out_weights_dir, f"{ranked_str}.pkl"), "wb") as f:
            pickle.dump(weights, f)
        return weights

    # ------------------------------------------------------------------------- #

    def on_setup(self) -> None:
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        config = self.init_config()
        self.ray_trainer = config.build()
        out_dir = self.out_checkpoint_dir
        self.model_path = os.path.join(out_dir, self.get_filename())
        self.training_data = defaultdict(list)

    def on_tear_down(self) -> DataFrame:
        self.ray_trainer.save(self.model_path)
        self.ray_trainer.stop()
        return DataFrame.from_dict(self.training_data)

    def _get_result_value(self, key, default=None):
        """Safely access result dict keys, handling Ray 2.x nested structure."""
        # Try direct access first (old API stack usually keeps flat keys)
        if key in self._result:
            return self._result[key]
        # Try under env_runners (Ray 2.x new structure)
        env_runners = self._result.get("env_runners", {})
        if key in env_runners:
            return env_runners[key]
        # Try sampler_results (another possible location)
        sampler = self._result.get("sampler_results", {})
        if key in sampler:
            return sampler[key]
        return default

    def on_logging_step(self) -> None:
        status = "{}Ep. #{} | ranked={} | Mean reward: {:6.2f} | Mean length: {:4.2f} | Saved {} ({})"
        logging.info(status.format(
            "" if self.trainer_name is None else f"[{self.trainer_name}] ",
            self._round+1,
            self.ranked,
            self._get_result_value("episode_reward_mean", 0.0),
            self._get_result_value("episode_len_mean", 0.0),
            self.model_path.split(os.sep)[-1],
            ctime()
        ))

    def get_key(self) -> str:
        if self.trainer_name is None:
            raise ValueError("`trainer_name` cannot be None.")
        ranked = "ranked" if self.ranked else "unranked"
        key = f"{self.trainer_name}_{self.net_dir}_{ranked}"
        return key

    def get_key_count(self) -> int:
        return self.counter.get(self.get_key())

    def incr_key_count(self) -> None:
        self.counter.increment(self.get_key())

    def get_filename(self) -> str:
        if self.trainer_name is None:
            raise ValueError("`trainer_name` cannot be None.")
        ranked = "ranked" if self.ranked else "unranked"
        if self.out_prefix is not None:
            ranked = f"{self.out_prefix}_{ranked}"
        return f"{ranked}"
        # return f"{ranked}_{self.idx}"

    def get_weights_filename(self) -> str:
        ranked = "ranked" if self.ranked else "unranked"
        return f"{ranked}"

    def set_rand_route_seed(self, seed) -> None:
        self.env

    # ------------------------------------------------------------------------- #

    @abstractmethod
    def on_make_final_policy() -> Weights:
        """This function is to be used for defining the weights used for the final policy
           to be used during evaluation. Each Trainer sub-class will come up with their
           own way for doing this procedure. For instance, simply grabbing one of the
           trained policies at random and returning its weights is sufficient (though
           likely not a desirable approach). The returned weights will then be used to
           in the GLOBAL policy that evaluation will be used.

        Raises:
            NotImplementedError: Cannot be called for the abstract BaseTrainer class.

        Returns:
            Weights: The model weights to be used in the GLOBAL model for evaluation.
        """
        raise NotImplementedError("Must implement abstract function "
                                  "`on_make_final_policy`.")

    @abstractmethod
    def on_data_recording_step(self) -> None:
        raise NotImplementedError("Must implement abstract function "
                                  "`on_data_recording_step`.")

    @abstractmethod
    def on_policy_setup(self) -> Dict[str, Tuple[Any]]:
        raise NotImplementedError("Must implement abstract function "
                                  "`on_policy_setup`.")
