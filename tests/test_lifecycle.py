import importlib.util
from pathlib import Path
import time
import pytest

spec = importlib.util.spec_from_file_location(
    "lifecycle", Path(__file__).resolve().parents[1] / "core" / "lifecycle.py"
)
lifecycle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lifecycle)

MemoryLifecycleManager = lifecycle.MemoryLifecycleManager
LifecycleConfig = lifecycle.LifecycleConfig
MemoryLifecycleState = lifecycle.MemoryLifecycleState
ConsolidationState = lifecycle.ConsolidationState

class DummyTemporalMetadata:
    def __init__(self, state=ConsolidationState.INITIAL):
        self.consolidation_state = state
        self.consolidation_strength = 0.0

class DummyTrace:
    def __init__(self, timestamp, last_access, salience=0.0, state=MemoryLifecycleState.NEW):
        self.timestamp = timestamp
        self.last_access = last_access
        self.salience = salience
        self.state = state
        self.eviction_protection = False
        self.temporal_metadata = DummyTemporalMetadata()

def _manager():
    cfg = LifecycleConfig()
    cfg.age_thresholds = {
        'aging_threshold': 10,
        'eviction_threshold': 20,
        'archive_threshold': 30,
    }
    return MemoryLifecycleManager(node_id="test", config=cfg)

@pytest.mark.asyncio
async def test_new_state_evaluation():
    mgr = _manager()
    now = time.time()
    trace = DummyTrace(timestamp=now - 5, last_access=now - 5)
    state = await mgr.evaluate_trace_lifecycle(trace, now)
    assert state == MemoryLifecycleState.NEW
    trace.timestamp = now - 11
    state = await mgr.evaluate_trace_lifecycle(trace, now)
    assert state == MemoryLifecycleState.ACTIVE

@pytest.mark.asyncio
async def test_active_to_aging_and_eviction():
    mgr = _manager()
    now = time.time()
    trace = DummyTrace(timestamp=now - 11, last_access=now - 11, state=MemoryLifecycleState.ACTIVE)
    state = await mgr.evaluate_trace_lifecycle(trace, now)
    assert state == MemoryLifecycleState.AGING
    trace.timestamp = now - 21
    state = await mgr.evaluate_trace_lifecycle(trace, now)
    assert state == MemoryLifecycleState.EVICTION_CANDIDATE

@pytest.mark.asyncio
async def test_consolidation_state_mapping():
    mgr = _manager()
    now = time.time()
    trace = DummyTrace(timestamp=now, last_access=now, state=MemoryLifecycleState.ACTIVE)
    trace.temporal_metadata.consolidation_state = ConsolidationState.CONSOLIDATED
    state = await mgr.evaluate_trace_lifecycle(trace, now)
    assert state == MemoryLifecycleState.CONSOLIDATED

@pytest.mark.asyncio
async def test_eviction_marks_evicted():
    mgr = _manager()
    now = time.time()
    trace = DummyTrace(timestamp=now - 21, last_access=now - 21, state=MemoryLifecycleState.EVICTION_CANDIDATE)
    selected = await mgr.select_eviction_candidates([trace], 1)
    assert trace in selected
    assert trace.state == MemoryLifecycleState.EVICTED
    assert mgr.stats['eviction_stats']['total_evicted'] == 1
