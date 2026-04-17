import networkx as nx
import numpy as np
import random
try:
    import libsumo as traci
except ImportError:
    import traci
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "libsumo not available, falling back to TraCI socket interface. "
        "Install SUMO >= 1.9 with libsumo support for ~8x faster simulation."
    )
import xml.etree.ElementTree as ET

from gymnasium import spaces
from scipy import stats
from typing import List

from seal.sumo.config import *
from seal.sumo.kernel.const import *
from seal.sumo.kernel.trafficlight.space import trafficlight_space


class TrafficLight:
    """This class represents an indidivual `trafficlight` (tls) in SUMO. The purpose is to
       simplify the necessary code for the needs of the RL environments in seal.
    """

    index: int
    id: int
    program: List[int]
    num_phases: int
    state: int
    phase: str
    ranked: bool

    def __init__(
        self,
        index: int,
        tls_id: int,
        netfile: str,
        sort_phases: bool = SORT_DEFAULT,
        force_all_red: bool = False,
        ranked: bool = True
    ) -> None:
        # The `index` data member is for the consistently simple indexing for actions
        # that are represented via lists. This is important for the `stable-baselines`
        # implementation that does not support Dict spaces.
        self.index = index
        self.id = tls_id
        self.program = self.get_program(netfile, force_all_red)
        self.num_phases = len(self.program)
        self.state = random.randrange(self.num_phases)
        self.phase = self.program[self.state]
        self.ranked = ranked

    @property
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(2)  # 0 = stay, 1 = advance to next phase

    @property
    def observation_space(self) -> spaces.Tuple:
        return trafficlight_space(self.ranked)

    def update(self) -> None:
        """Update the current state by interacting with `traci`."""
        try:
            self.phase = traci.trafficlight.getRedYellowGreenState(self.id)
            if self.phase in self.program:
                self.state = self.program.index(self.phase)
            else:
                # Phase not in program — can happen with non-standard topologies
                self.state = 0
        except (traci.exceptions.FatalTraCIError, traci.exceptions.TraCIException):
            pass

    def next_phase(self) -> None:
        next_state = (self.state+1) % self.num_phases
        next_phase = self.program[next_state]
        try:
            self.state = next_state
            self.phase = next_phase
            traci.trafficlight.setRedYellowGreenState(self.id, self.phase)
        except (traci.exceptions.FatalTraCIError, traci.exceptions.TraCIException):
            pass

    def get_program(
        self,
        road_netfile: str,
        force_all_red: bool = False
    ) -> List[str]:
        """Get the possible traffic light phases for the specific traffic light via the
           given `tls_id`.

        Args:
            road_netfile (str): The traffic light id.
            force_all_red (bool, optional): Requires there's a state of all red lights if
                True. Defaults to False.

        Returns:
            List[str]: A list of all the possible phases the given traffic light can take.
        """
        with open(road_netfile, "r") as f:
            tree = ET.parse(f)
            logic = tree.find(f"tlLogic[@id='{self.id}']")
            if logic is None:
                # Fallback for OSM-imported networks where some TLS junctions
                # lack a tlLogic entry. Query SUMO via TraCI at runtime instead.
                try:
                    phases = traci.trafficlight.getAllProgramLogics(self.id)
                    if phases:
                        states = [p.state for p in phases[0].phases]
                        if states:
                            return states
                except Exception:
                    pass
                # Last resort: generate a minimal 2-phase program from the
                # junction's connection count
                junction = tree.find(f"junction[@id='{self.id}']")
                if junction is not None:
                    inc_lanes = junction.get("incLanes", "")
                    n_lanes = len(inc_lanes.split()) if inc_lanes else 4
                else:
                    n_lanes = 4
                half = n_lanes // 2
                phase_g = "G" * half + "r" * (n_lanes - half)
                phase_r = "r" * half + "G" * (n_lanes - half)
                return [phase_g, phase_r]
            states = [phase.attrib["state"] for phase in logic]
            if force_all_red:
                all_reds = len(states[0]) * STATE_r_STR
                if all_reds not in states:
                    states.append(all_reds)
            return states

    def get_observation(self) -> np.ndarray:
        # Initialize the observation list (obs).
        n_features = N_RANKED_FEATURES if self.ranked else N_UNRANKED_FEATURES
        obs = [0 for _ in range(n_features)]

        # Extract the lane-specific features.
        obs[LANE_OCCUPANCY] = self.__get_lane_occupancy()
        obs[HALTED_LANE_OCCUPANCY] = self.__get_halted_lane_occupancy()
        obs[SPEED_RATIO] = self.__get_speed_ratio()

        # Extract descriptive statistics features for the current traffic light state.
        curr_tls_state = traci.trafficlight.getRedYellowGreenState(self.id)
        for light in curr_tls_state:
            if light == STATE_r_STR:
                obs[PHASE_STATE_r] += 1/len(curr_tls_state)
            elif light == STATE_y_STR:
                obs[PHASE_STATE_y] += 1/len(curr_tls_state)
            elif light == STATE_g_STR:
                obs[PHASE_STATE_g] += 1/len(curr_tls_state)
            elif light == STATE_G_STR:
                obs[PHASE_STATE_G] += 1/len(curr_tls_state)
            # elif light == STATE_s_STR:
            #     obs[PHASE_STATE_s] += 1/len(curr_tls_state)
            elif light == STATE_u_STR:
                obs[PHASE_STATE_u] += 1/len(curr_tls_state)
            elif light == STATE_o_STR:
                obs[PHASE_STATE_o] += 1/len(curr_tls_state)
            elif light == STATE_O_STR:
                obs[PHASE_STATE_O] += 1/len(curr_tls_state)

        return np.array(obs)

    def _get_controlled_lanes(self) -> list:
        """Get controlled lanes with error handling for non-standard topologies."""
        try:
            lanes = traci.trafficlight.getControlledLanes(self.id)
            # Deduplicate lanes (SUMO can return duplicates for complex junctions)
            return list(dict.fromkeys(lanes))
        except (traci.exceptions.TraCIException, traci.exceptions.FatalTraCIError):
            return []

    def get_num_of_controlled_vehicles(self) -> int:
        try:
            return sum(traci.lane.getLastStepVehicleNumber(lane)
                       for lane in self._get_controlled_lanes())
        except (traci.exceptions.TraCIException, traci.exceptions.FatalTraCIError):
            return 0

    def __get_lane_occupancy(self) -> float:
        try:
            lanes = self._get_controlled_lanes()
            if not lanes:
                return 0.0
            lane_lengths = sum(traci.lane.getLength(l) for l in lanes)
            if lane_lengths == 0:
                return 0.0
            vehicle_lengths = sum(sum(traci.vehicle.getLength(v)
                                      for v in traci.lane.getLastStepVehicleIDs(l))
                                  for l in lanes)
            return min(vehicle_lengths / lane_lengths, 1.0)
        except (traci.exceptions.TraCIException, traci.exceptions.FatalTraCIError):
            return 0.0

    def __get_halted_lane_occupancy(self) -> float:
        try:
            lanes = self._get_controlled_lanes()
            if not lanes:
                return 0.0
            lane_lengths = sum(traci.lane.getLength(l) for l in lanes)
            if lane_lengths == 0:
                return 0.0
            halted_lengths = sum(sum(traci.vehicle.getLength(v)
                                     for v in traci.lane.getLastStepVehicleIDs(l)
                                     if traci.vehicle.getSpeed(v) < HALTING_SPEED)
                                 for l in lanes)
            return min(halted_lengths / lane_lengths, 1.0)
        except (traci.exceptions.TraCIException, traci.exceptions.FatalTraCIError):
            return 0.0

    def __get_speed_ratio(self) -> float:
        try:
            max_lane_speeds = vehicle_speeds = 0
            for l in self._get_controlled_lanes():
                for v in traci.lane.getLastStepVehicleIDs(l):
                    speed_limit = traci.lane.getMaxSpeed(l)
                    vehicle_speeds += min(traci.vehicle.getSpeed(v), speed_limit)
                    max_lane_speeds += speed_limit
            if max_lane_speeds == 0:
                return 1.0
            return min(1.0, vehicle_speeds / max_lane_speeds)
        except (traci.exceptions.TraCIException, traci.exceptions.FatalTraCIError,
                ZeroDivisionError):
            return 1.0
