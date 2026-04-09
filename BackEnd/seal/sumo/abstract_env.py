import gymnasium
import numpy as np
import os
import random
import shutil
import tempfile

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from seal.sumo.config import *
from seal.sumo.kernel.kernel import SumoKernel
from seal.sumo.timer import ActionTimer
from seal.sumo.utils.random_routes import generate_random_routes

DEFAULT_SEED = 0


class AbstractSumoEnv(ABC, MultiAgentEnv):

    def __init__(self, config: Dict[str, Any]):
        # Deep copy so we don't mutate the caller's dict (important when
        # the same config is reused for dummy + real envs in eval runner).
        import copy
        self.config = copy.deepcopy(config)
        self.rand_route_args = self.config.get("rand_route_args", {})
        self.use_dynamic_seed = self.config.get("use_dynamic_seed", True)
        self.ranked = self.config.get("ranked", DEFUALT_RANKED)
        self.env_seed = self.rand_route_args.get("seed", DEFAULT_SEED)
        # Create a process-specific working directory so parallel experiments
        # don't clobber each other's route files (traffic.rou.xml, trips.trips.xml).
        self._work_dir = tempfile.mkdtemp(prefix="sumo_env_")
        net_basename = os.path.basename(self.config["net-file"])
        dst_net = os.path.join(self._work_dir, net_basename)
        shutil.copy2(self.config["net-file"], dst_net)
        self.config["net-file"] = dst_net
        self.path = self._work_dir
        self.horizon = config.get("horizon", None)
        self.time_of_day = config.get("time_of_day", False)

        # Check if the user provided a route-file to be used for simulations and if the
        # user wants random routes to be generated for EACH trial (rand_routes_on_reset).
        # If user's want random routes generated (i.e., "route-files" is None in provided
        # config), then a private flag, `__first_rand_routes_flag`, is set to True. This
        # forces the `reset()` function to generate at least ONE random route file before
        # being updated to False. This will never change back to True (hence the privacy).
        if self.config.get("route-files", None) is None:
            self.config["route-files"] = os.path.join(self.path,
                                                      "traffic.rou.xml")
            self.rand_routes_on_reset = self.config.get("rand_routes_on_reset",
                                                        True)
            self.__first_rand_routes_flag = True
        else:
            self.rand_routes_on_reset = False
            self.__first_rand_routes_flag = False

        if self.time_of_day:
            self.rand_routes_on_reset = True

        self.kernel = SumoKernel(self.config)
        self.action_timer = ActionTimer(len(self.kernel.tls_hub))
        self.num_of_lanes = self.kernel.get_num_of_lanes()
        # self.road_capacity = self.kernel.get_road_capacity()
        self.reset()

    def reset(self, *, seed=None, options=None) -> Tuple[Any, Dict]:
        """Start the simulation and get details surrounding the world.

        Returns
        -------
        Tuple[Any, Dict]
            The observation and info dict upon resetting the simulation/environment.
        """
        if seed is not None:
            self.seed(seed)
        self.step_counter = 0
        if self.rand_routes_on_reset or self.__first_rand_routes_flag:
            self.rand_routes()
            self.__first_rand_routes_flag = False
        self.kernel.start()
        self.action_timer.restart()
        obs = self._observe()
        info = {agent_id: {} for agent_id in obs}
        return obs, info

    def rand_routes(self) -> None:
        """Generate random routes based on the details in the configuration
           dict provided at initialization.
        """
        netfile = self.config["net-file"]
        self.rand_route_args["n_routefiles"] = 1  # NOTE: Issues if > 1.
        if self.horizon is not None:
            self.rand_route_args["end_time"] = self.horizon
        if self.use_dynamic_seed:
            self.rand_route_args["seed"] = self.env_seed
            self.env_seed += 1

        if self.time_of_day:
            # Sample demand from time-of-day range per episode
            period = random.choice(["am_rush", "midday", "pm_rush"])
            vplph_ranges = {
                "am_rush": (500, 700),
                "midday": (200, 300),
                "pm_rush": (400, 600),
            }
            lo, hi = vplph_ranges[period]
            vplph = random.randint(lo, hi)
            self.rand_route_args["vehicles_per_lane_per_hour"] = vplph

        generate_random_routes(
            netfile=netfile,
            path=self.path,
            number_of_lanes=self.num_of_lanes,
            **self.rand_route_args
        )

    def close(self) -> None:
        self.kernel.close()
        # Clean up process-specific temp directory
        if hasattr(self, '_work_dir') and os.path.exists(self._work_dir):
            shutil.rmtree(self._work_dir, ignore_errors=True)

    def seed(self, seed) -> None:
        """This is needed for Ray's RlLib package. It calls this function.

        Args:
            seed (int): New seed value.
        """
        random.seed(seed)
        np.random.seed(seed)
        # random.seed(self.env_seed)
        # np.random.seed(self.env_seed)

    ## ============================================================================== ##
    ## .....ABSTRACT METHODS THAT NEED TO BE IMPLEMENTED BY CHILDREN CLASSES..... ##
    ## ============================================================================== ##
    @abstractmethod
    def step(self, actions) -> Tuple[Any, Any, Any, Any, Any]:
        raise NotImplementedError("Cannot be called from Abstract "
                                  "Class `AbstractSumoEnv`.")

    @abstractmethod
    def _do_action(self, actions: Any) -> Any:
        raise NotImplementedError("Cannot be called from Abstract "
                                  "Class `AbstractSumoEnv`.")

    @abstractmethod
    def _get_reward(self) -> float:
        raise NotImplementedError("Cannot be called from Abstract "
                                  "Class `AbstractSumoEnv`.")

    @abstractmethod
    def _observe(self) -> Any:
        raise NotImplementedError("Cannot be called from Abstract "
                                  "Class `AbstractSumoEnv`.")
