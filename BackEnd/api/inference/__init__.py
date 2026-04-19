"""Ray-free inference infrastructure for the ATLAS comparison demo.

Loads SEAL trainer weights directly into a PyTorch PPO actor and drives
SUMO simulations via TraCI without any Ray dependency. Two simulations
can run concurrently (one per subprocess) so the demo can compare two
(topology, strategy) configs side-by-side in real time.
"""
