"""Max-pressure baseline for traffic signal control.

At each decision point, selects the phase that maximizes the difference between
upstream queue (vehicles waiting to enter) and downstream availability.
This is a well-known non-learning baseline in the traffic signal control literature.
"""
import traci


def compute_max_pressure_actions(agent_ids):
    """Compute max-pressure actions for all traffic lights.

    For each TLS, computes pressure for current phase vs next phase.
    Action = 1 (advance) if next phase has higher pressure, else 0 (stay).

    Args:
        agent_ids: List of traffic light IDs.

    Returns:
        Dict mapping agent_id -> action (0 or 1).
    """
    actions = {}
    for tls_id in agent_ids:
        controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
        current_state = traci.trafficlight.getRedYellowGreenState(tls_id)

        # Compute pressure for current green directions
        current_pressure = _compute_phase_pressure(controlled_lanes, current_state)

        # Get next phase from the TLS program
        program = traci.trafficlight.getAllProgramLogics(tls_id)
        if program:
            phases = program[0].phases
            current_idx = traci.trafficlight.getPhase(tls_id)
            next_idx = (current_idx + 1) % len(phases)
            next_state = phases[next_idx].state
            next_pressure = _compute_phase_pressure(controlled_lanes, next_state)
        else:
            next_pressure = 0

        # Switch if next phase has higher pressure
        actions[tls_id] = 1 if next_pressure > current_pressure else 0

    return actions


def _compute_phase_pressure(controlled_lanes, phase_state):
    """Compute pressure for a given phase state.

    Pressure = sum of queue lengths on lanes that have green in this phase,
    minus sum of queue lengths on lanes that have red.
    """
    green_pressure = 0
    for i, lane_id in enumerate(controlled_lanes):
        if i >= len(phase_state):
            break
        signal = phase_state[i]
        queue = traci.lane.getLastStepHaltingNumber(lane_id)
        if signal in ('G', 'g'):
            green_pressure += queue
        elif signal == 'r':
            green_pressure -= queue * 0.5  # Penalize red queues less
    return green_pressure
