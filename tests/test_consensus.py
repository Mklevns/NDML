import pytest
import torch
import asyncio
import aiohttp # For type hinting ClientSession, though it will be mocked
from unittest.mock import AsyncMock, patch, MagicMock # AsyncMock for async methods, patch for mocking

from consensus.neuromorphic import NeuromorphicConsensusLayer

# Use pytest_asyncio for async tests and fixtures
pytest_plugins = 'pytest_asyncio'

@pytest.fixture
def neuromorphic_consensus_layer():
    """Fixture to create a NeuromorphicConsensusLayer instance."""
    return NeuromorphicConsensusLayer(node_id="test_consensus_node", dimension=128)

@pytest.mark.asyncio
async def test_propose_memory_update_success(neuromorphic_consensus_layer: NeuromorphicConsensusLayer, caplog):
    """Test propose_memory_update successfully sends proposals to all peers."""
    consensus_layer = neuromorphic_consensus_layer

    # Register some mock peers
    peer1_id = "peer1.test.local:8080" # Peer with port
    peer2_id = "peer2.test.local"      # Peer without port (should default)
    await consensus_layer.register_peer(peer1_id)
    await consensus_layer.register_peer(peer2_id)

    sample_trace_id = "trace_123"
    sample_content_vector = torch.randn(consensus_layer.dimension)
    sample_metadata = {"source": "test"}

    # Mock aiohttp.ClientSession and its post method
    mock_session_post = AsyncMock()

    # Simulate a successful response from peers
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.text = AsyncMock(return_value='{"status": "ok"}')
    mock_session_post.return_value.__aenter__.return_value = mock_response # Simulate 'async with session.post(...) as response:'

    # Patch aiohttp.ClientSession globally or where it's used.
    # NeuromorphicConsensusLayer uses `async with aiohttp.ClientSession() as session:`
    # So we need to mock ClientSession() to return an object that has `post` as AsyncMock
    mock_client_session_instance = MagicMock()
    mock_client_session_instance.post = mock_session_post

    # The ClientSession itself is a context manager
    mock_client_session_constructor = MagicMock(return_value=mock_client_session_instance)
    # Make the instance a context manager
    mock_client_session_instance.__aenter__.return_value = mock_client_session_instance
    mock_client_session_instance.__aexit__.return_value = None


    with patch('aiohttp.ClientSession', mock_client_session_constructor):
        result = await consensus_layer.propose_memory_update(
            sample_trace_id, sample_content_vector, sample_metadata
        )

    assert result is True, "propose_memory_update should return True on success"
    assert mock_session_post.call_count == len(consensus_layer.peers), "Should call post for each peer"

    expected_payload = {
        "trace_id": sample_trace_id,
        "content_vector": sample_content_vector.tolist(),
        "metadata": sample_metadata,
        "proposer_node_id": consensus_layer.node_id
    }

    # Check calls to post
    calls = mock_session_post.call_args_list
    assert len(calls) == 2

    # Expected URLs
    expected_url_peer1 = f"http://{peer1_id}/propose_update"
    expected_url_peer2_default_port = f"http://{peer2_id}:8080/propose_update" # Default port

    # Check first call (order might vary depending on set iteration, so check both)
    actual_urls = {call[0][0] for call in calls} # Get the first arg (url) from each call
    actual_jsons = {str(call[1]['json']) for call in calls} # Get the json kwarg from each call, convert dict to str for set comparison

    assert expected_url_peer1 in actual_urls
    assert expected_url_peer2_default_port in actual_urls

    # Check that the payload was sent correctly (at least one of them, assuming they are all the same)
    # Since dicts are unhashable for sets if they contain lists, we compare the string representation for simplicity,
    # or iterate and compare dicts if more precision is needed.
    assert str(expected_payload) in actual_jsons

    # Check for warning log for peer2_id using default port
    assert any(record.levelname == 'WARNING' and f"Peer ID {peer2_id} does not contain port, defaulting to {expected_url_peer2_default_port}" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_propose_memory_update_network_error(neuromorphic_consensus_layer: NeuromorphicConsensusLayer):
    """Test propose_memory_update returns False if a network error occurs for any peer."""
    consensus_layer = neuromorphic_consensus_layer
    await consensus_layer.register_peer("peer1.test.local:8080")
    await consensus_layer.register_peer("peer2.test.local:8080")

    mock_session_post = AsyncMock()
    # Simulate one success, one network error
    mock_good_response = AsyncMock()
    mock_good_response.status = 200
    mock_good_response.text = AsyncMock(return_value='{"status": "ok"}')

    # Simulate ClientConnectorError for the second call
    mock_session_post.side_effect = [
        AsyncMock(return_value=mock_good_response), # Enters __aenter__
        aiohttp.ClientConnectorError(MagicMock(), OSError("Connection failed"))
    ]

    # We need to make sure the context manager behavior is also part of the side_effect if post itself is the context manager
    # The _send_proposal_to_peer helper makes session.post(url, json=payload) the context manager
    # So, mock_session_post is `session.post`. Each call to it returns an async context manager.

    # Correct mocking for `async with session.post(...) as response:`
    # The return value of session.post should be an async context manager.
    mock_post_cm_good = AsyncMock() # This is the context manager
    mock_post_cm_good.__aenter__.return_value = mock_good_response # This is the response object

    # For the error case, the error can be raised when session.post is called, or when its __aenter__ is called.
    # If `_send_proposal_to_peer`'s `async with session.post(url, json=payload) as response:` is the target,
    # and `session.post` itself raises the error:
    mock_session_post_constructor_level = AsyncMock(side_effect=[
        mock_post_cm_good, # First call returns a working context manager
        aiohttp.ClientConnectorError(MagicMock(), OSError("Connection failed")) # Second call raises error
    ])

    mock_client_session_instance = MagicMock()
    mock_client_session_instance.post = mock_session_post_constructor_level # session.post()

    mock_client_session_constructor = MagicMock(return_value=mock_client_session_instance)
    mock_client_session_instance.__aenter__.return_value = mock_client_session_instance
    mock_client_session_instance.__aexit__.return_value = None

    with patch('aiohttp.ClientSession', mock_client_session_constructor):
        result = await consensus_layer.propose_memory_update(
            "trace_err", torch.randn(128), {}
        )

    assert result is False, "propose_memory_update should return False if any peer proposal fails due to network error"
    # Ensure post was attempted for both (or until error)
    # Depending on asyncio.gather behavior with return_exceptions=True, all tasks run.
    assert mock_session_post_constructor_level.call_count == 2


@pytest.mark.asyncio
async def test_sync_with_peer(neuromorphic_consensus_layer: NeuromorphicConsensusLayer, caplog):
    """Test sync_with_peer logs messages based on peer_data vote."""
    consensus_layer = neuromorphic_consensus_layer
    peer_id = "peer_source_node"
    trace_id = "trace_sync_123"

    # Test case 1: Vote FOR update
    peer_data_vote_yes = {"vote_for_update": True, "trace_id": trace_id, "other_info": "yes_details"}
    await consensus_layer.sync_with_peer(peer_id, peer_data_vote_yes)

    assert any(
        f"Node {consensus_layer.node_id}: Received sync data from peer {peer_id}: {peer_data_vote_yes}" in record.message and record.levelname == 'INFO'
        for record in caplog.records
    )
    assert any(
        f"Node {consensus_layer.node_id}: Peer {peer_id} voted FOR update of trace {trace_id}." in record.message and record.levelname == 'INFO'
        for record in caplog.records
    )
    caplog.clear() # Clear logs for next case

    # Test case 2: Vote AGAINST update
    peer_data_vote_no = {"vote_for_update": False, "trace_id": trace_id, "other_info": "no_details"}
    await consensus_layer.sync_with_peer(peer_id, peer_data_vote_no)
    assert any(
        f"Node {consensus_layer.node_id}: Peer {peer_id} voted AGAINST update of trace {trace_id}." in record.message and record.levelname == 'INFO'
        for record in caplog.records
    )
    caplog.clear()

    # Test case 3: NO VOTE (or unclear vote)
    peer_data_no_vote = {"vote_for_update": None, "trace_id": trace_id, "other_info": "none_details"}
    await consensus_layer.sync_with_peer(peer_id, peer_data_no_vote)
    assert any(
        f"Node {consensus_layer.node_id}: Peer {peer_id} provided no clear vote (vote_for_update: None) for trace {trace_id}." in record.message and record.levelname == 'WARNING'
        for record in caplog.records
    )
    caplog.clear()

    # Test case 4: Missing 'vote_for_update' key
    peer_data_missing_key = {"other_info": "missing_vote_key", "trace_id": trace_id}
    await consensus_layer.sync_with_peer(peer_id, peer_data_missing_key)
    assert any(
        f"Node {consensus_layer.node_id}: Peer {peer_id} provided no clear vote (vote_for_update: None) for trace {trace_id}." in record.message and record.levelname == 'WARNING'
        for record in caplog.records # .get('vote_for_update') defaults to None
    )
    caplog.clear()

    # Test case 5: Missing 'trace_id' key
    peer_data_missing_trace_id = {"vote_for_update": True}
    await consensus_layer.sync_with_peer(peer_id, peer_data_missing_trace_id)
    assert any(
        f"Node {consensus_layer.node_id}: Peer {peer_id} voted FOR update of trace N/A." in record.message and record.levelname == 'INFO'
        for record in caplog.records # trace_id defaults to 'N/A'
    )
    caplog.clear()
