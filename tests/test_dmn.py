import pytest
import torch
from unittest.mock import patch, MagicMock, PropertyMock

from core.dmn import GPUAcceleratedDMN
# Assuming BTSPUpdateMechanism and BTSPUpdateDecision are in core.btsp
# If they are not, these imports will need to be adjusted.
# For the purpose of this subtask, we assume they are in core.btsp as per the prompt.
try:
    from core.btsp import BTSPUpdateMechanism, BTSPUpdateDecision
except ImportError:
    # Create dummy classes if core.btsp or its classes don't exist yet,
    # so that the test file can be written and type-hinted against them.
    # The actual mocking will take over during test execution.
    BTSPUpdateMechanism = MagicMock(name="MockBTSPUpdateMechanism")
    BTSPUpdateDecision = MagicMock(name="MockBTSPUpdateDecision")


@pytest.fixture
def mock_btsp_classes():
    with patch('core.dmn.BTSPUpdateMechanism', autospec=True) as MockBTSPMechanism, \
         patch('core.dmn.BTSPUpdateDecision', autospec=True) as MockBTSPDecision:
        # Configure the mock decision class's instances if needed, e.g.
        # mock_decision_instance = MockBTSPDecision.return_value
        # type(mock_decision_instance).should_update = PropertyMock(return_value=True)
        # type(mock_decision_instance).calcium_level = PropertyMock(return_value=0.8)
        yield MockBTSPMechanism, MockBTSPDecision


@pytest.fixture
def gpu_accelerated_dmn_config():
    return {
        "node_id": "test_dmn_node",
        "dimension": 128,
        "capacity": 1000,
        "device": "cpu", # Test with CPU first, can parameterize later for CUDA
        "config": {
            "btsp": {
                "calcium_threshold": 0.5,
                "novelty_weight": 0.6
            },
            "indexing": { # Add basic indexing config to prevent errors during init
                "similarity_threshold": 0.7
            },
            "system": { # Add basic system config
                "dimension": 128,
                "node_capacity": 1000
            }
        }
    }


@pytest.fixture
async def dmn_instance(gpu_accelerated_dmn_config, mock_btsp_classes):
    MockBTSPMechanism, _ = mock_btsp_classes

    # The GPUAcceleratedDMN initializes btsp in its _init_biological_mechanisms,
    # which is called by its own __init__ via _init_indexing_system -> _init_biological_mechanisms (indirectly through super() in the original code, now directly).
    # We need to ensure that the mock is in place when GPUAcceleratedDMN.__init__ is called.

    # The GPUAcceleratedDMN's __init__ calls _init_biological_mechanisms
    # which should instantiate self.btsp using the mocked BTSPUpdateMechanism
    dmn = GPUAcceleratedDMN(
        node_id=gpu_accelerated_dmn_config["node_id"],
        dimension=gpu_accelerated_dmn_config["dimension"],
        capacity=gpu_accelerated_dmn_config["capacity"],
        device=gpu_accelerated_dmn_config["device"],
        config=gpu_accelerated_dmn_config["config"]
    )
    # We need to explicitly call _init_biological_mechanisms if it's async and not awaited in __init__
    # Based on the previous changes, _init_biological_mechanisms is async.
    # However, __init__ itself is not async, so it cannot await.
    # This implies _init_biological_mechanisms is either called in a sync way (not good)
    # or it's meant to be called separately after __init__, or by an async method.
    # Let's assume for now that it's called correctly during init or we call it if needed.
    # The prompt for GPUAcceleratedDMN changes made _init_biological_mechanisms sync by removing await super()
    # and directly instantiating BTSPUpdateMechanism.
    # Let's re-verify the GPUAcceleratedDMN structure from previous steps.
    # The `_init_biological_mechanisms` in `GPUAcceleratedDMN` is synchronous.
    # It directly instantiates `BTSPUpdateMechanism`.

    # No need to call dmn._init_biological_mechanisms() separately as it's part of DMN's __init__ flow.
    return dmn, MockBTSPMechanism


# Async fixture if dmn_instance setup needs async operations
# For now, assuming synchronous setup for dmn_instance based on current understanding of DMN init
# @pytest_asyncio.fixture
# async def dmn_instance_async(gpu_accelerated_dmn_config, mock_btsp_classes):
#     MockBTSPMechanism, _ = mock_btsp_classes
#     dmn = GPUAcceleratedDMN(**gpu_accelerated_dmn_config)
#     await dmn._init_biological_mechanisms() # If it were async
#     return dmn, MockBTSPMechanism


def test_init_biological_mechanisms(dmn_instance):
    dmn, MockBTSPMechanism = dmn_instance

    # Assert that self.btsp is an instance of the mocked BTSPUpdateMechanism
    assert isinstance(dmn.btsp, MockBTSPMechanism)

    # Assert that BTSPUpdateMechanism was initialized with the correct config
    # The mock_calls attribute stores all calls to the mock, including constructor
    init_call_args = MockBTSPMechanism.call_args
    assert init_call_args is not None, "BTSPUpdateMechanism was not called"

    # The first argument to the constructor will be the config dict
    btsp_config_arg = init_call_args[0][0] # call_args is a tuple (args, kwargs)

    assert btsp_config_arg['device'] == dmn.device
    assert btsp_config_arg['dimension'] == dmn.dimension
    # Check other specific btsp settings if necessary
    expected_btsp_config_from_dmn = dmn.config.get('btsp', {})
    for key, value in expected_btsp_config_from_dmn.items():
        assert btsp_config_arg[key] == value


@pytest.mark.asyncio
async def test_add_memory_trace_async_btsp_decision(dmn_instance, mock_btsp_classes):
    dmn, MockBTSPMechanism = dmn_instance
    _, MockBTSPDecisionClass = mock_btsp_classes

    # Mock the evaluate_async method on the dmn.btsp instance
    # dmn.btsp is already an instance of MockBTSPMechanism
    mock_decision_instance = MockBTSPDecisionClass() # Create an instance of the mock decision *class*

    # Configure attributes on the *instance* of MockBTSPDecision
    # Using PropertyMock to allow direct attribute access to be tracked
    type(mock_decision_instance).should_update = PropertyMock(return_value=True)
    type(mock_decision_instance).calcium_level = PropertyMock(return_value=0.9)

    # Make evaluate_async return this specific, configured instance
    dmn.btsp.evaluate_async = MagicMock(return_value=mock_decision_instance)

    sample_content = torch.randn(dmn.dimension)
    sample_context = {"meta": "test"}
    sample_salience = 0.8

    # --- Test Case 1: should_update is True ---
    await dmn.add_memory_trace_async(sample_content, sample_context, sample_salience)

    # Assert that evaluate_async was called
    dmn.btsp.evaluate_async.assert_called_once()
    call_args = dmn.btsp.evaluate_async.call_args
    assert torch.equal(call_args[1]['content'], sample_content) # content is a kwarg
    assert call_args[1]['context'] == sample_context

    # Assert that decision.should_update and decision.calcium_level were accessed
    # Accessing PropertyMock counts towards mock_calls on the *type* (the PropertyMock object itself)
    assert type(mock_decision_instance).should_update.mock_calls is not None
    assert len(type(mock_decision_instance).should_update.mock_calls) > 0
    assert type(mock_decision_instance).calcium_level.mock_calls is not None
    assert len(type(mock_decision_instance).calcium_level.mock_calls) > 0

    # Assert trace was added
    assert len(dmn.memory_traces) == 1
    assert dmn.memory_traces[0].content is sample_content # Check if it's the same tensor (or a clone)
    # Note: DMN stores a clone, so torch.equal is better
    assert torch.equal(dmn.memory_traces[0].content, sample_content)
    assert dmn.trace_index # Check if trace_index is updated

    # Reset for next test case
    dmn.memory_traces.clear()
    dmn.trace_index.clear()
    dmn.btsp.evaluate_async.reset_mock()
    # Reset PropertyMock call counts
    type(mock_decision_instance).should_update.reset_mock()
    type(mock_decision_instance).calcium_level.reset_mock()


    # --- Test Case 2: should_update is False ---
    type(mock_decision_instance).should_update = PropertyMock(return_value=False)
    # calcium_level can remain the same or be changed if logic depends on it

    await dmn.add_memory_trace_async(sample_content, sample_context, sample_salience)

    dmn.btsp.evaluate_async.assert_called_once()
    assert type(mock_decision_instance).should_update.mock_calls is not None
    assert len(type(mock_decision_instance).should_update.mock_calls) > 0
    # calcium_level might still be accessed even if should_update is False, for logging or other reasons
    # If it's guaranteed not to be accessed, this assertion can be more specific.
    # Based on current DMN code, calcium_level is logged regardless of should_store.
    assert type(mock_decision_instance).calcium_level.mock_calls is not None
    assert len(type(mock_decision_instance).calcium_level.mock_calls) > 0

    # Assert trace was NOT added
    assert len(dmn.memory_traces) == 0
    assert not dmn.trace_index

    # --- Test Case 3: BTSP evaluation fails (optional, good for robustness) ---
    dmn.btsp.evaluate_async = MagicMock(side_effect=Exception("BTSP Boom!"))
    # Fallback logic in DMN uses salience > config.get('btsp_fallback_salience_threshold', 0.5)
    # Set salience to be above a potential fallback threshold
    high_salience = 0.9
    # Ensure 'btsp_fallback_salience_threshold' is in config or use default
    dmn.config['btsp_fallback_salience_threshold'] = 0.5

    await dmn.add_memory_trace_async(sample_content, sample_context, high_salience)
    assert len(dmn.memory_traces) == 1 # Should store due to fallback
    dmn.memory_traces.clear()
    dmn.trace_index.clear()

    # Salience below fallback threshold
    low_salience = 0.1
    await dmn.add_memory_trace_async(sample_content, sample_context, low_salience)
    assert len(dmn.memory_traces) == 0 # Should not store

    # --- Test Case 4: BTSP module is None (optional) ---
    # This tests the part of add_memory_trace_async: if self.btsp:
    # For this, we'd need a DMN instance where self.btsp is explicitly set to None after init.
    # This might be tricky with the current fixture structure if it always ensures btsp is mocked.
    # Consider a separate fixture or test if this path is critical.
    # For now, assume self.btsp is always initialized by _init_biological_mechanisms.

    # Cleanup mocks on the PropertyMock objects if they are to be reused extensively
    # or ensure fresh instances for each logical test section.
    # Here, we re-assign `type(mock_decision_instance).should_update` in each section,
    # effectively resetting its specific return value for that part of the test.
    # The call count is on the PropertyMock object itself, so it accumulates unless reset.
    # Resetting them as done above is good practice.

    # Final check on attribute access, e.g. for `calcium_level`
    # This ensures that our mock setup for PropertyMock is indeed working as expected for attribute access tracking
    # For example, after it's supposed to be accessed:
    # mock_calcium_property = type(mock_decision_instance).calcium_level
    # assert mock_calcium_property.call_count > 0 # or specific number of calls

    # Note: If `add_memory_trace_async` uses `getattr(decision, 'should_update')`
    # then `PropertyMock` might not register that as a "call".
    # However, direct attribute access `decision.should_update` is correctly tracked by `PropertyMock`.
    # A quick look at `core/dmn.py` shows it uses direct attribute access.
    pass # Test completed
