"""Tests for STREETS data loading, time-series construction, and partitioning.

Uses synthetic data — no STREETS download needed.
"""
import numpy as np
import pytest

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────
# Synthetic data helpers
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_counts():
    """Synthetic traffic count DataFrame."""
    dates_2018 = pd.date_range('2018-08-21 06:00', '2018-08-21 22:00', freq='10min')
    dates_2019 = pd.date_range('2019-06-10 06:00', '2019-06-10 22:00', freq='5min')
    dates = dates_2018.append(dates_2019)
    cameras = [f'Camera_{i}' for i in range(10)]
    data = np.random.randint(0, 50, size=(len(dates), len(cameras)))
    return pd.DataFrame(data, index=dates, columns=cameras)


@pytest.fixture
def synthetic_states():
    """Synthetic traffic state DataFrame (0=free, 1=blocked)."""
    dates = pd.date_range('2018-08-21 06:00', '2018-08-21 22:00', freq='10min')
    cameras = [f'Camera_{i}' for i in range(10)]
    data = np.random.choice([0, 1], size=(len(dates), len(cameras)), p=[0.9, 0.1])
    return pd.DataFrame(data, index=dates, columns=cameras)


@pytest.fixture
def synthetic_communities():
    """Synthetic community assignment."""
    return {
        'community_a': [f'Camera_{i}' for i in range(5)],
        'community_b': [f'Camera_{i}' for i in range(5, 10)],
    }


# ──────────────────────────────────────────────────────────────────────
# Congestion scoring
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas required")
class TestCongestionFromCounts:
    def test_basic_scoring(self, synthetic_counts):
        from data.streets_loader import compute_congestion_from_counts
        congestion = compute_congestion_from_counts(synthetic_counts, max_count=50)
        assert congestion.shape == synthetic_counts.shape
        assert congestion.min().min() >= 0.0
        assert congestion.max().max() <= 1.0

    def test_max_count_normalization(self, synthetic_counts):
        from data.streets_loader import compute_congestion_from_counts
        # With max_count=10, values > 10 should clip to 1.0
        congestion = compute_congestion_from_counts(synthetic_counts, max_count=10)
        assert congestion.max().max() <= 1.0

    def test_state_calibration(self, synthetic_counts, synthetic_states):
        from data.streets_loader import compute_congestion_from_counts
        congestion = compute_congestion_from_counts(
            synthetic_counts, states=synthetic_states, max_count=50
        )
        assert congestion.shape == synthetic_counts.shape
        assert congestion.min().min() >= 0.0
        assert congestion.max().max() <= 1.0

    def test_blocked_state_boosts_score(self):
        from data.streets_loader import compute_congestion_from_counts
        dates = pd.date_range('2018-08-21 06:00', periods=5, freq='10min')
        counts = pd.DataFrame({'cam': [5, 5, 5, 5, 5]}, index=dates)
        states = pd.DataFrame({'cam': [0, 0, 1, 1, 0]}, index=dates)
        congestion = compute_congestion_from_counts(counts, states, max_count=50)
        # Blocked rows (index 2,3) should have score >= 0.6
        assert congestion.iloc[2, 0] >= 0.6
        assert congestion.iloc[3, 0] >= 0.6
        # Free-flow rows should have score <= 0.5
        assert congestion.iloc[0, 0] <= 0.5


# ──────────────────────────────────────────────────────────────────────
# Time-series construction
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas required")
class TestBuildNodeTimeseries:
    def test_output_shape(self, synthetic_counts):
        from data.streets_loader import compute_congestion_from_counts, build_node_timeseries
        congestion = compute_congestion_from_counts(synthetic_counts, max_count=50)
        cameras = list(synthetic_counts.columns)
        ts = build_node_timeseries(congestion, cameras, freq='10min')
        assert ts.ndim == 3
        assert ts.shape[1] == len(cameras)  # N
        assert ts.shape[2] == 1  # F

    def test_missing_cameras_filled(self, synthetic_counts):
        from data.streets_loader import compute_congestion_from_counts, build_node_timeseries
        congestion = compute_congestion_from_counts(synthetic_counts, max_count=50)
        # Request cameras that don't exist
        cameras = list(synthetic_counts.columns) + ['NonExistent_1', 'NonExistent_2']
        ts = build_node_timeseries(congestion, cameras, freq='10min')
        assert ts.shape[1] == len(cameras)
        # Non-existent cameras should be all zeros
        assert np.allclose(ts[:, -2:, :], 0.0)

    def test_values_bounded(self, synthetic_counts):
        from data.streets_loader import compute_congestion_from_counts, build_node_timeseries
        congestion = compute_congestion_from_counts(synthetic_counts, max_count=50)
        ts = build_node_timeseries(congestion, list(synthetic_counts.columns), freq='10min')
        assert ts.min() >= 0.0
        assert ts.max() <= 1.0


# ──────────────────────────────────────────────────────────────────────
# Chronological split
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas required")
class TestSplitByYear:
    def test_2018_split(self, synthetic_counts):
        from data.streets_loader import split_by_year
        train = split_by_year(synthetic_counts, 2018)
        assert all(train.index.year == 2018)
        assert len(train) > 0

    def test_2019_split(self, synthetic_counts):
        from data.streets_loader import split_by_year
        test = split_by_year(synthetic_counts, 2019)
        assert all(test.index.year == 2019)
        assert len(test) > 0

    def test_no_overlap(self, synthetic_counts):
        from data.streets_loader import split_by_year
        train = split_by_year(synthetic_counts, 2018)
        test = split_by_year(synthetic_counts, 2019)
        overlap = train.index.intersection(test.index)
        assert len(overlap) == 0


# ──────────────────────────────────────────────────────────────────────
# Partitioning
# ──────────────────────────────────────────────────────────────────────

class TestPartitionByCommunity:
    def test_basic_partition(self, synthetic_communities):
        from data.streets_loader import partition_by_community
        samples = [{'camera_id': f'Camera_{i}', 'value': i} for i in range(10)]
        result = partition_by_community(samples, synthetic_communities)
        assert len(result['community_a']) == 5
        assert len(result['community_b']) == 5

    def test_no_cross_contamination(self, synthetic_communities):
        from data.streets_loader import partition_by_community
        samples = [{'camera_id': f'Camera_{i}'} for i in range(10)]
        result = partition_by_community(samples, synthetic_communities)
        a_cams = {s['camera_id'] for s in result['community_a']}
        b_cams = {s['camera_id'] for s in result['community_b']}
        assert len(a_cams & b_cams) == 0  # No overlap

    def test_unknown_camera_dropped(self, synthetic_communities):
        from data.streets_loader import partition_by_community
        samples = [{'camera_id': 'Unknown_Camera'}]
        result = partition_by_community(samples, synthetic_communities)
        total = sum(len(v) for v in result.values())
        assert total == 0


class TestPartitionByCameraSubgroups:
    def test_even_split(self):
        from data.streets_loader import partition_by_camera_subgroups
        cameras = [f'cam_{i}' for i in range(20)]
        parts = partition_by_camera_subgroups(cameras, num_clients=4)
        assert len(parts) == 4
        assert sum(len(v) for v in parts.values()) == 20

    def test_uneven_split(self):
        from data.streets_loader import partition_by_camera_subgroups
        cameras = [f'cam_{i}' for i in range(10)]
        parts = partition_by_camera_subgroups(cameras, num_clients=3)
        assert len(parts) == 3
        assert sum(len(v) for v in parts.values()) == 10
        # First client(s) should have one extra
        sizes = [len(v) for v in parts.values()]
        assert max(sizes) - min(sizes) <= 1

    def test_more_clients_than_cameras(self):
        from data.streets_loader import partition_by_camera_subgroups
        cameras = [f'cam_{i}' for i in range(3)]
        parts = partition_by_camera_subgroups(cameras, num_clients=10)
        assert len(parts) == 3  # Capped at num cameras
        assert sum(len(v) for v in parts.values()) == 3

    def test_no_empty_partitions(self):
        from data.streets_loader import partition_by_camera_subgroups
        cameras = [f'cam_{i}' for i in range(15)]
        parts = partition_by_camera_subgroups(cameras, num_clients=5)
        for client_id, cams in parts.items():
            assert len(cams) > 0
