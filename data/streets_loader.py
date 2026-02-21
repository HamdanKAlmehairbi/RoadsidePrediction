"""STREETS dataset loader for graph-based traffic congestion forecasting.

STREETS (Snyder & Do, NeurIPS 2019) provides:
- 640 sensors (322 Buffalo Grove + 318 Gurnee) = 320 cameras x 2 directions
- Each camera has inbound + outbound vehicle counts at 5-10 min intervals
- 2 directed-graph communities with pre-built adjacency matrices
- Free-flow ('f') / blocked ('b') traffic state labels
- ~2.5 months of data (2018 Aug-Sep @ 10min, 2019 Jun-Jul @ 5min)

Data from: https://databank.illinois.edu/datasets/IDB-3671567

Actual directory structure:
    STREETS/
    ├── graphs/
    │   ├── buffalogrove/buffalogrove-graph.json   # 322 sensors, adjacency
    │   └── gurnee/gurnee-graph.json               # 318 sensors, adjacency
    ├── trafficcounts/
    │   ├── 2018/YYYY-M-D-trafficcounts.json       # 31 daily files
    │   └── 2019/YYYY-M-D-trafficcounts.json       # 44 daily files
    ├── trafficstate/
    │   ├── traffic_state_labels.json               # {filename: {state: 'f'|'b'}}
    │   └── *.jpg                                   # 6400 annotated images
    └── 2019-7-3_2019-7-9/                          # Weekly image archives (optional)
"""
import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────
# Graph loading
# ──────────────────────────────────────────────────────────────────────

def _load_community_graph(json_path: str) -> Dict:
    """Load a single community graph JSON.

    Returns dict with:
        sensor_dict: {sensor_id: {'coords': [lat,lon], 'name': str}}
        adjacency: np.ndarray (N_sensors, N_sensors)
        distance: np.ndarray (N_sensors, N_sensors)
        camera_names: List[str] unique camera names (no -inbound/-outbound)
        sensor_to_camera: {sensor_id: camera_name}
    """
    with open(json_path) as f:
        data = json.load(f)

    sensor_dict = {}
    sensor_to_camera = {}
    camera_names_set = set()

    for sid, (coords, name) in data['sensor-dictionary'].items():
        sensor_dict[sid] = {'coords': coords, 'name': name}
        # Strip -inbound/-outbound to get camera name
        base = name.replace('-inbound', '').replace('-outbound', '')
        sensor_to_camera[sid] = base
        camera_names_set.add(base)

    adjacency = np.array(data['adjacency-matrix'], dtype=np.float32)
    distance = np.array(data['distance-matrix'], dtype=np.float32)
    camera_names = sorted(camera_names_set)

    return {
        'sensor_dict': sensor_dict,
        'adjacency': adjacency,
        'distance': distance,
        'camera_names': camera_names,
        'sensor_to_camera': sensor_to_camera,
        'num_sensors': len(sensor_dict),
        'num_cameras': len(camera_names),
    }


def load_streets_graph(dataroot: str) -> Dict:
    """Load STREETS directed graph topology for both communities.

    The graph operates at the CAMERA level (320 nodes). Each camera's
    adjacency is derived by aggregating sensor-level adjacency
    (inbound + outbound sensors per camera approach).

    Args:
        dataroot: Path to STREETS root directory

    Returns:
        Dict with keys:
            adjacency: (N, N) full camera-level adjacency matrix
            community_adj: {'buffalogrove': (N1,N1), 'gurnee': (N2,N2)}
            communities: {'buffalogrove': [cam_names], 'gurnee': [cam_names]}
            camera_ids: List[str] all camera names in order
            num_nodes: int total cameras
            sensor_graphs: per-community raw sensor-level data
    """
    graph_dir = Path(dataroot) / 'graphs'

    communities_data = {}
    for comm_dir in sorted(graph_dir.iterdir()):
        if not comm_dir.is_dir():
            continue
        json_files = list(comm_dir.glob('*-graph.json'))
        if not json_files:
            continue
        comm_name = comm_dir.name
        communities_data[comm_name] = _load_community_graph(str(json_files[0]))

    if not communities_data:
        raise FileNotFoundError(f"No community graph files found in {graph_dir}")

    # Build camera-level adjacency per community
    communities = {}
    community_adj = {}

    for comm_name, cdata in communities_data.items():
        cam_names = cdata['camera_names']
        communities[comm_name] = cam_names

        # Aggregate sensor adjacency to camera level
        # For each pair of cameras, sum adjacency weights across their sensors
        n_cams = len(cam_names)
        cam_adj = np.zeros((n_cams, n_cams), dtype=np.float32)

        cam_to_idx = {c: i for i, c in enumerate(cam_names)}
        s2c = cdata['sensor_to_camera']

        for si in range(cdata['num_sensors']):
            for sj in range(cdata['num_sensors']):
                w = cdata['adjacency'][si, sj]
                if w > 0:
                    ci = cam_to_idx.get(s2c.get(str(si)))
                    cj = cam_to_idx.get(s2c.get(str(sj)))
                    if ci is not None and cj is not None and ci != cj:
                        cam_adj[ci, cj] = max(cam_adj[ci, cj], w)

        community_adj[comm_name] = cam_adj

    # Build full adjacency (block diagonal — communities don't connect)
    all_camera_ids = []
    for comm_name in sorted(communities.keys()):
        all_camera_ids.extend(communities[comm_name])

    N = len(all_camera_ids)
    full_adj = np.zeros((N, N), dtype=np.float32)

    offset = 0
    for comm_name in sorted(communities.keys()):
        n = len(communities[comm_name])
        full_adj[offset:offset+n, offset:offset+n] = community_adj[comm_name]
        offset += n

    return {
        'adjacency': full_adj,
        'community_adj': community_adj,
        'communities': communities,
        'camera_ids': all_camera_ids,
        'num_nodes': N,
        'sensor_graphs': communities_data,
    }


# ──────────────────────────────────────────────────────────────────────
# Traffic counts
# ──────────────────────────────────────────────────────────────────────

def _parse_count_timestamp(date_str: str, time_parts: List[int]) -> datetime:
    """Parse date string 'YYYY-M-D' + [hour, minute] into datetime."""
    parts = date_str.split('-')
    year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
    hour, minute = time_parts
    return datetime(year, month, day, hour, minute)


def load_streets_traffic_counts(dataroot: str,
                                 camera_ids: Optional[List[str]] = None
                                 ) -> pd.DataFrame:
    """Load per-camera vehicle count time-series from daily JSON files.

    Each JSON maps camera_name -> {image_filename: {inbound, outbound, timestamp}}.
    Total count = inbound + outbound.

    Args:
        dataroot: Path to STREETS root directory
        camera_ids: Optional subset of cameras to load

    Returns:
        DataFrame with DatetimeIndex and one column per camera (total count)
    """
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas is required for STREETS data loading")

    counts_dir = Path(dataroot) / 'trafficcounts'
    all_records = []

    for year_dir in sorted(counts_dir.iterdir()):
        if not year_dir.is_dir():
            continue

        for json_file in sorted(year_dir.glob('*-trafficcounts.json')):
            # Parse date from filename: YYYY-M-D-trafficcounts.json
            fname = json_file.stem  # e.g., "2018-8-21-trafficcounts"
            date_parts = fname.replace('-trafficcounts', '')  # "2018-8-21"

            with open(json_file) as f:
                day_data = json.load(f)

            for cam_name, observations in day_data.items():
                if camera_ids is not None and cam_name not in camera_ids:
                    continue

                for img_key, obs in observations.items():
                    ts = _parse_count_timestamp(date_parts, obs['timestamp'])
                    total = obs.get('inbound', 0) + obs.get('outbound', 0)
                    all_records.append({
                        'timestamp': ts,
                        'camera': cam_name,
                        'count': total,
                        'inbound': obs.get('inbound', 0),
                        'outbound': obs.get('outbound', 0),
                    })

    if not all_records:
        raise FileNotFoundError(f"No traffic count data found in {counts_dir}")

    df = pd.DataFrame(all_records)

    # Pivot to wide format: timestamp x camera -> count
    # Average duplicates (same camera + timestamp from different images)
    pivot = df.pivot_table(
        index='timestamp', columns='camera', values='count',
        aggfunc='mean'
    )
    pivot = pivot.sort_index()

    return pivot


# ──────────────────────────────────────────────────────────────────────
# Traffic state labels
# ──────────────────────────────────────────────────────────────────────

def load_streets_traffic_state(dataroot: str) -> pd.DataFrame:
    """Load free-flow/blocked binary traffic state labels.

    Reads traffic_state_labels.json where each entry has:
        file_attributes.state: 'f' (free-flow) or 'b' (blocked/queue)
        filename: 'YYYY-M-D-H-M.jpg-CameraName.jpg'

    Args:
        dataroot: Path to STREETS root directory

    Returns:
        DataFrame with DatetimeIndex, columns = camera names,
        values = 0 (free-flow) or 1 (blocked)
    """
    state_dir = Path(dataroot) / 'trafficstate'
    labels_path = state_dir / 'traffic_state_labels.json'

    if not labels_path.exists():
        raise FileNotFoundError(f"No traffic_state_labels.json in {state_dir}")

    with open(labels_path) as f:
        labels = json.load(f)

    records = []
    for key, entry in labels.items():
        filename = entry['filename']
        state = entry.get('file_attributes', {}).get('state', '')

        # Parse filename: "YYYY-M-D-H-M.jpg-CameraName.jpg"
        parts = filename.split('.jpg-')
        if len(parts) != 2:
            continue

        ts_str = parts[0]  # "YYYY-M-D-H-M"
        cam_name = parts[1].replace('.jpg', '')  # "CameraName"

        try:
            ts_parts = ts_str.split('-')
            ts = datetime(
                int(ts_parts[0]), int(ts_parts[1]), int(ts_parts[2]),
                int(ts_parts[3]), int(ts_parts[4])
            )
        except (ValueError, IndexError):
            continue

        # 'f' = free-flow (0), 'b' = blocked/queue (1)
        state_val = 1 if state == 'b' else 0

        records.append({
            'timestamp': ts,
            'camera': cam_name,
            'state': state_val,
        })

    if not records:
        raise ValueError("No valid state labels parsed")

    df = pd.DataFrame(records)
    pivot = df.pivot_table(
        index='timestamp', columns='camera', values='state',
        aggfunc='max'  # If any observation is blocked, mark as blocked
    )
    return pivot.sort_index()


# ──────────────────────────────────────────────────────────────────────
# Congestion scoring
# ──────────────────────────────────────────────────────────────────────

def compute_congestion_from_counts(counts: pd.DataFrame,
                                    states: Optional[pd.DataFrame] = None,
                                    max_count: int = 50) -> pd.DataFrame:
    """Compute continuous congestion scores [0,1] from vehicle counts.

    Base formula: score = count / max_count, clipped to [0, 1].
    If state labels are available, calibrates:
        blocked (state=1): score = max(score, 0.6)
        free-flow (state=0): score = min(score, 0.5)

    Args:
        counts: Vehicle count DataFrame (timestamps x cameras)
        states: Optional binary state DataFrame (0=free-flow, 1=blocked)
        max_count: Count value that maps to congestion=1.0

    Returns:
        DataFrame of congestion scores [0.0, 1.0]
    """
    congestion = counts.clip(lower=0) / max_count
    congestion = congestion.clip(upper=1.0)

    if states is not None:
        common_idx = congestion.index.intersection(states.index)
        common_cols = [c for c in congestion.columns if c in states.columns]

        if len(common_idx) > 0 and len(common_cols) > 0:
            aligned_states = states.loc[common_idx, common_cols]
            aligned_congestion = congestion.loc[common_idx, common_cols]

            adjusted = aligned_congestion.copy()
            blocked_mask = aligned_states == 1
            freeflow_mask = aligned_states == 0
            adjusted[blocked_mask] = adjusted[blocked_mask].clip(lower=0.6)
            adjusted[freeflow_mask] = adjusted[freeflow_mask].clip(upper=0.5)

            congestion.loc[common_idx, common_cols] = adjusted

    return congestion.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────
# Time-series construction
# ──────────────────────────────────────────────────────────────────────

def build_node_timeseries(congestion: pd.DataFrame,
                          camera_ids: List[str],
                          freq: str = '5min') -> np.ndarray:
    """Build (T, N, F) array from congestion DataFrame.

    Resamples to uniform frequency, fills gaps, orders columns
    to match camera_ids.

    Args:
        congestion: Congestion score DataFrame
        camera_ids: Ordered camera list (defines node ordering)
        freq: Target frequency ('5min' or '10min')

    Returns:
        np.ndarray (T, N, 1) where N = len(camera_ids)
    """
    available = [c for c in camera_ids if c in congestion.columns]
    df = congestion[available].copy()

    # Resample to uniform frequency
    df = df.resample(freq).mean()
    df = df.ffill().bfill().fillna(0.0)

    # Ensure column order matches camera_ids (zeros for missing)
    ordered = pd.DataFrame(0.0, index=df.index, columns=camera_ids)
    for c in available:
        ordered[c] = df[c]

    data = ordered.values.astype(np.float32)
    return data[:, :, np.newaxis]  # (T, N, 1)


# ──────────────────────────────────────────────────────────────────────
# Image indexing (for CNN vision pipeline)
# ──────────────────────────────────────────────────────────────────────

def load_streets_image_index(dataroot: str,
                              camera_ids: Optional[List[str]] = None
                              ) -> Dict[str, Dict[str, str]]:
    """Build image path index from traffic state images.

    Traffic state images have naming: "YYYY-M-D-H-M.jpg-CameraName.jpg"
    We index these by camera -> timestamp_str -> image_path.

    Args:
        dataroot: Path to STREETS root directory
        camera_ids: Optional subset of cameras

    Returns:
        Nested dict: camera_name -> timestamp_str -> image_path
    """
    state_dir = Path(dataroot) / 'trafficstate'
    index = {}

    for img_file in state_dir.glob('*.jpg'):
        fname = img_file.name
        parts = fname.split('.jpg-')
        if len(parts) != 2:
            continue

        ts_str = parts[0]
        cam_name = parts[1].replace('.jpg', '')

        if camera_ids is not None and cam_name not in camera_ids:
            continue

        if cam_name not in index:
            index[cam_name] = {}
        index[cam_name][ts_str] = str(img_file)

    return index


def extract_streets_camera_samples(congestion: pd.DataFrame,
                                    image_index: Dict[str, Dict[str, str]],
                                    camera_ids: List[str]) -> List[Dict]:
    """Extract (image_path, congestion_score) samples for CameraDataset.

    Matches congestion scores to available images by timestamp.

    Args:
        congestion: Congestion score DataFrame
        image_index: Output of load_streets_image_index
        camera_ids: Camera IDs to include

    Returns:
        List of dicts with 'image_path' and 'congestion_score'
    """
    samples = []

    for cam_id in camera_ids:
        if cam_id not in image_index or cam_id not in congestion.columns:
            continue

        cam_images = image_index[cam_id]
        cam_scores = congestion[cam_id]

        for ts, score in cam_scores.items():
            # Format timestamp to match image naming
            if hasattr(ts, 'strftime'):
                ts_str = f"{ts.year}-{ts.month}-{ts.day}-{ts.hour}-{ts.minute}"
            else:
                ts_str = str(ts)

            if ts_str in cam_images:
                samples.append({
                    'image_path': cam_images[ts_str],
                    'congestion_score': float(score),
                    'camera_id': cam_id,
                    'timestamp': str(ts),
                })

    return samples


# ──────────────────────────────────────────────────────────────────────
# Partitioning for Federated Learning
# ──────────────────────────────────────────────────────────────────────

def partition_by_community(samples_or_data,
                           communities: Dict[str, List[str]]) -> Dict[str, list]:
    """Partition data by graph community.

    Args:
        samples_or_data: List of sample dicts with 'camera_id' key
        communities: {'buffalogrove': [cam_names], 'gurnee': [cam_names]}

    Returns:
        Dict mapping community_name -> list of samples
    """
    cam_to_comm = {}
    for comm_name, cams in communities.items():
        for cam in cams:
            cam_to_comm[cam] = comm_name

    partitioned = {comm: [] for comm in communities}
    for sample in samples_or_data:
        cam_id = sample.get('camera_id', sample.get('cam_id'))
        if cam_id in cam_to_comm:
            partitioned[cam_to_comm[cam_id]].append(sample)

    return partitioned


def partition_by_camera_subgroups(camera_ids: List[str],
                                  num_clients: int) -> Dict[int, List[str]]:
    """Split cameras into subgroups for FL clients.

    Args:
        camera_ids: Camera IDs to partition
        num_clients: Number of clients to create

    Returns:
        Dict mapping client_id -> list of camera_ids
    """
    num_clients = min(num_clients, len(camera_ids))
    partitions = {}
    per_client = len(camera_ids) // num_clients
    remainder = len(camera_ids) % num_clients

    start = 0
    for i in range(num_clients):
        end = start + per_client + (1 if i < remainder else 0)
        partitions[i] = camera_ids[start:end]
        start = end

    return partitions


# ──────────────────────────────────────────────────────────────────────
# Chronological splitting
# ──────────────────────────────────────────────────────────────────────

def split_by_year(data: pd.DataFrame, year: int) -> pd.DataFrame:
    """Filter DataFrame to a specific year (chronological split).

    Train on 2018 (10-min intervals), test on 2019 (5-min intervals).

    Args:
        data: DataFrame with DatetimeIndex
        year: Year to filter to

    Returns:
        Filtered DataFrame
    """
    return data[data.index.year == year]


# ──────────────────────────────────────────────────────────────────────
# Statistics
# ──────────────────────────────────────────────────────────────────────

def get_streets_dataset_stats(dataroot: str) -> Dict:
    """Summary statistics for the STREETS dataset.

    Args:
        dataroot: Path to STREETS root directory

    Returns:
        Dict with dataset statistics
    """
    stats = {}

    try:
        graph = load_streets_graph(dataroot)
        stats['num_cameras'] = graph['num_nodes']
        stats['num_communities'] = len(graph['communities'])
        stats['cameras_per_community'] = {
            k: len(v) for k, v in graph['communities'].items()
        }
        for comm, cdata in graph.get('sensor_graphs', {}).items():
            stats[f'{comm}_sensors'] = cdata['num_sensors']
    except Exception as e:
        stats['graph_error'] = str(e)

    try:
        counts = load_streets_traffic_counts(dataroot)
        stats['num_cameras_with_counts'] = len(counts.columns)
        stats['total_timestamps'] = len(counts)
        stats['date_range'] = (str(counts.index.min()), str(counts.index.max()))
        stats['total_observations'] = int(counts.notna().sum().sum())

        for year in sorted(counts.index.year.unique()):
            year_data = counts[counts.index.year == year]
            stats[f'year_{year}'] = {
                'timestamps': len(year_data),
                'date_range': (str(year_data.index.min()), str(year_data.index.max())),
            }
    except Exception as e:
        stats['counts_error'] = str(e)

    try:
        states = load_streets_traffic_state(dataroot)
        n_blocked = (states == 1).sum().sum()
        n_free = (states == 0).sum().sum()
        stats['state_labels'] = {
            'total': int(n_blocked + n_free),
            'blocked': int(n_blocked),
            'free_flow': int(n_free),
            'blocked_ratio': float(n_blocked / (n_blocked + n_free)) if (n_blocked + n_free) > 0 else 0,
        }
    except Exception as e:
        stats['state_error'] = str(e)

    return stats
