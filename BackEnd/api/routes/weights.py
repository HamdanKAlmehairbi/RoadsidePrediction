import os
from fastapi import APIRouter

router = APIRouter()

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "example_weights", "ICCPS", "Final")
WEIGHTS_DIR = os.path.normpath(WEIGHTS_DIR)

# Cache: weight_id -> absolute file path
_weight_map = {}


def _build_weight_map():
    """Walk the weights directory tree and build id -> filepath mapping."""
    _weight_map.clear()
    if not os.path.isdir(WEIGHTS_DIR):
        return

    for trainer_name in os.listdir(WEIGHTS_DIR):
        trainer_path = os.path.join(WEIGHTS_DIR, trainer_name)
        if not os.path.isdir(trainer_path):
            continue
        for topo_name in os.listdir(trainer_path):
            topo_path = os.path.join(trainer_path, topo_name)
            if not os.path.isdir(topo_path):
                continue
            for filename in os.listdir(topo_path):
                if not filename.endswith(".pkl"):
                    continue
                filepath = os.path.join(topo_path, filename)
                weight_id, trainer, aggr, topology, ranked = _parse_weight_file(
                    trainer_name, topo_name, filename
                )
                _weight_map[weight_id] = {
                    "filepath": filepath,
                    "trainer": trainer,
                    "aggr": aggr,
                    "topology": topology,
                    "ranked": ranked,
                }


def _parse_weight_file(trainer_name: str, topo_name: str, filename: str):
    """Parse weight file path components into structured metadata.

    Examples:
      FedRL / grid-3x3 / v3_naive-aggr_ranked.pkl -> fedrl-naive-3x3-ranked
      MARL  / grid-3x3 / v3_ranked.pkl            -> marl-3x3-ranked
    """
    trainer = trainer_name  # FedRL, MARL, SARL
    topology = topo_name    # grid-3x3, grid-5x5, double

    # Strip version prefix and .pkl extension: "v3_naive-aggr_ranked.pkl" -> "naive-aggr_ranked"
    stem = filename.rsplit(".", 1)[0]  # remove .pkl
    parts = stem.split("_", 1)  # split on first _ to remove version prefix
    rest = parts[1] if len(parts) > 1 else parts[0]

    # Determine ranked/unranked
    ranked = "ranked" in rest and "unranked" not in rest

    # Extract aggregation function for FedRL
    aggr = None
    if trainer.upper() == "FEDRL":
        # rest is like "naive-aggr_ranked" or "pos-reward-aggr_unranked"
        aggr_part = rest.replace("_ranked", "").replace("_unranked", "")
        aggr_part = aggr_part.replace("-aggr", "")
        aggr = aggr_part  # "naive", "pos-reward", "neg-reward", "traffic"

    # Shorten topology for id: "grid-3x3" -> "3x3", "double" -> "double"
    topo_short = topology.replace("grid-", "")

    # Build weight_id
    ranked_str = "ranked" if ranked else "unranked"
    if aggr:
        weight_id = f"{trainer.lower()}-{aggr}-{topo_short}-{ranked_str}"
    else:
        weight_id = f"{trainer.lower()}-{topo_short}-{ranked_str}"

    return weight_id, trainer, aggr, topology, ranked


def get_weight_filepath(weight_id: str):
    """Look up the absolute file path for a given weight_id."""
    if not _weight_map:
        _build_weight_map()
    entry = _weight_map.get(weight_id)
    return entry["filepath"] if entry else None


@router.get("/api/weights")
def list_weights():
    if not _weight_map:
        _build_weight_map()

    weights = []
    for wid, info in _weight_map.items():
        weights.append({
            "id": wid,
            "trainer": info["trainer"],
            "aggr": info["aggr"],
            "topology": info["topology"],
            "ranked": info["ranked"],
        })

    return {"weights": weights}
