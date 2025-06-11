import sys
print(f"Python version: {sys.version}")
print(f"Python path: {sys.path}")

print("Attempting to import ndml package...")
import ndml
print("Successfully imported ndml")

print("Attempting to import EnhancedDistributedMemoryNode from ndml.core (via ndml.core.__init__)...")
from ndml.core import EnhancedDistributedMemoryNode # Corrected import
print("Successfully imported EnhancedDistributedMemoryNode from ndml.core")

# The previous subtask mentioned correcting an alias for this in __init__.py
# Let's try importing the aliased name from the top level
print("Attempting to import EnhancedDistributedMemoryNode directly from ndml (checking __all__)...")
from ndml import EnhancedDistributedMemoryNode as TopLevelDMN
print("Successfully imported EnhancedDistributedMemoryNode as TopLevelDMN from ndml")

print("Attempting to import MemoryGateway from ndml.integration...")
from ndml.integration.memory_gateway import MemoryGateway
print("Successfully imported MemoryGateway from ndml.integration")

print("Attempting to import ConsensusNode from ndml.deployment...")
from ndml.deployment.consensus_node import ConsensusNode
print("Successfully imported ConsensusNode from ndml.deployment")

print("Attempting to import normalize_embeddings from ndml.utils...")
from ndml.utils import normalize_embeddings
print("Successfully imported normalize_embeddings from ndml.utils")

print("Attempting to import get_version directly from ndml...")
from ndml import get_version
print("Successfully imported get_version from ndml")

print("Attempting to call ndml.get_version()...")
version = ndml.get_version()
print(f"NDML version: {version}")

# Test an alias that was mentioned in a previous subtask report
print("Attempting to import MultiTimescaleDynamicsEngine directly from ndml (checking __all__)...")
from ndml import MultiTimescaleDynamicsEngine
print("Successfully imported MultiTimescaleDynamicsEngine from ndml")

print("All import checks passed.")
