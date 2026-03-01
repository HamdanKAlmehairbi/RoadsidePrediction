import os
import xml.etree.ElementTree as ET
from fastapi import APIRouter, HTTPException

router = APIRouter()

CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "SMARTCOMP")
CONFIGS_DIR = os.path.normpath(CONFIGS_DIR)

TOPOLOGIES = ["grid-3x3", "grid-5x5", "grid-7x7"]


@router.get("/api/networks")
def list_networks():
    return {"networks": TOPOLOGIES}


@router.get("/api/network/{topology}")
def get_network(topology: str):
    if topology not in TOPOLOGIES:
        raise HTTPException(status_code=404, detail=f"Unknown topology: {topology}")

    net_file = os.path.join(CONFIGS_DIR, f"{topology}.net.xml")
    if not os.path.isfile(net_file):
        raise HTTPException(status_code=404, detail=f"Network file not found: {topology}")

    tree = ET.parse(net_file)
    root = tree.getroot()

    # Parse location for bounds
    location = root.find("location")
    if location is not None:
        conv = location.get("convBoundary", "0,0,0,0")
        parts = conv.split(",")
        min_x, min_y, max_x, max_y = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
    else:
        min_x, min_y, max_x, max_y = 0.0, 0.0, 0.0, 0.0

    # Parse junctions as nodes (skip internal junctions)
    nodes = []
    for junc in root.findall("junction"):
        jtype = junc.get("type", "")
        if jtype == "internal":
            continue
        node_id = junc.get("id")
        x = float(junc.get("x", "0"))
        y = float(junc.get("y", "0"))
        nodes.append({"id": node_id, "x": x, "y": y})

    # Parse edges (skip internal edges)
    edges = []
    for edge in root.findall("edge"):
        func = edge.get("function", "")
        if func == "internal":
            continue
        edge_id = edge.get("id")
        # Skip edges whose id starts with ':'  (internal junction connectors)
        if edge_id and edge_id.startswith(":"):
            continue
        from_node = edge.get("from", "")
        to_node = edge.get("to", "")
        lanes = []
        for lane in edge.findall("lane"):
            lane_id = lane.get("id")
            shape_str = lane.get("shape", "")
            shape = []
            for pair in shape_str.strip().split():
                xy = pair.split(",")
                if len(xy) == 2:
                    shape.append([float(xy[0]), float(xy[1])])
            lanes.append({"id": lane_id, "shape": shape})
        edges.append({
            "id": edge_id,
            "from": from_node,
            "to": to_node,
            "lanes": lanes,
        })

    return {
        "topology": topology,
        "bounds": {"min_x": min_x, "min_y": min_y, "max_x": max_x, "max_y": max_y},
        "nodes": nodes,
        "edges": edges,
    }
