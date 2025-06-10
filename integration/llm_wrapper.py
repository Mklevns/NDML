# integration/llm_wrapper.py
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from abc import ABC, abstractmethod
import logging

from .memory_gateway import MemoryGateway
from .fusion_network import MemoryFusionNetwork

logger = logging.getLogger(__name__)


class BaseLLMAdapter(ABC):
    """Abstract base class for LLM adapters"""

    @abstractmethod
    def get_hidden_states(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Extract hidden states for memory queries"""
        pass

    @abstractmethod
    def forward_with_memory(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                            memory_states: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with optional memory integration"""
        pass

    @abstractmethod
    def get_injection_points(self) -> List[str]:
        """Get layer names where memory can be injected"""
        pass


class TransformerAdapter(BaseLLMAdapter):
    """Adapter for standard Transformer models (GPT, Llama, Mistral, etc.)"""

    def __init__(self, model: nn.Module, memory_dimension: int):
        self.model = model
        self.memory_dimension = memory_dimension
        self.hidden_size = model.config.hidden_size

        # Project model hidden states to memory dimension
        self.query_projection = nn.Linear(self.hidden_size, memory_dimension)

        # Project memory back to model dimension
        self.memory_projection = nn.Linear(memory_dimension, self.hidden_size)

        # Injection points (typically after attention layers)
        self.injection_layers = [f'model.layers.{i}' for i in range(0, model.config.num_hidden_layers, 4)]

    def get_hidden_states(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Extract hidden states from the model"""
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

            # Use last layer hidden states, mean-pooled
            last_hidden = outputs.hidden_states[-1]  # [batch, seq_len, hidden_size]

            # Mean pooling over sequence dimension (excluding padding)
            if attention_mask is not None:
                masked_hidden = last_hidden * attention_mask.unsqueeze(-1)
                pooled = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            else:
                pooled = last_hidden.mean(dim=1)

            return pooled

    def forward_with_memory(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                            memory_states: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with memory injection"""

        # If no memory, use standard forward
        if memory_states is None:
            return self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Custom forward with memory injection
        # This is a simplified example - real implementation would hook into specific layers
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

        # Project memory and add to intermediate representations
        memory_projected = self.memory_projection(memory_states)

        # Simple additive integration (could be more sophisticated)
        modified_hidden = outputs.hidden_states[-1] + memory_projected.unsqueeze(1)

        # Final layer normalization and output projection
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'norm'):
            modified_hidden = self.model.model.norm(modified_hidden)

        if hasattr(self.model, 'lm_head'):
            logits = self.model.lm_head(modified_hidden)
        else:
            logits = self.model.output(modified_hidden)

        return type(outputs)(logits=logits, hidden_states=outputs.hidden_states)

    def get_injection_points(self) -> List[str]:
        return self.injection_layers


class MambaAdapter(BaseLLMAdapter):
    """Adapter for Mamba state-space models"""

    def __init__(self, model: nn.Module, memory_dimension: int):
        self.model = model
        self.memory_dimension = memory_dimension
        self.hidden_size = model.config.d_model

        self.query_projection = nn.Linear(self.hidden_size, memory_dimension)
        self.memory_projection = nn.Linear(memory_dimension, self.hidden_size)

        # Mamba-specific injection points (state updates)
        self.injection_layers = [f'model.layers.{i}.mixer' for i in range(0, model.config.n_layer, 3)]

    def get_hidden_states(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Extract hidden states from Mamba model"""
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, output_hidden_states=True)

            # Mamba outputs may not have explicit attention_mask handling
            # Use the last hidden state with simple mean pooling
            last_hidden = outputs.hidden_states[-1]

            # Simple mean pooling
            pooled = last_hidden.mean(dim=1)

            return pooled

    def forward_with_memory(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                            memory_states: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with memory integration for Mamba"""

        if memory_states is None:
            return self.model(input_ids=input_ids)

        # For Mamba, memory integration might involve state modifications
        # This is a simplified implementation
        outputs = self.model(input_ids=input_ids, output_hidden_states=True)

        # Project and integrate memory
        memory_projected = self.memory_projection(memory_states)

        # Add memory to final hidden state
        modified_hidden = outputs.hidden_states[-1] + memory_projected.unsqueeze(1)

        # Apply final layers
        if hasattr(self.model, 'norm_f'):
            modified_hidden = self.model.norm_f(modified_hidden)

        if hasattr(self.model, 'lm_head'):
            logits = self.model.lm_head(modified_hidden)

        return type(outputs)(logits=logits, hidden_states=outputs.hidden_states)

    def get_injection_points(self) -> List[str]:
        return self.injection_layers


class NDMLIntegratedLLM(nn.Module):
    """Complete LLM with integrated NDML system"""

    def __init__(self,
                 model_name_or_path: str,
                 memory_dimension: int = 512,
                 memory_config: Optional[Dict[str, Any]] = None,
                 adapter_type: str = "auto"):

        super().__init__()

        # Load base model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )

        # Initialize memory system
        self.memory_dimension = memory_dimension
        self.memory_config = memory_config or {}

        # Create appropriate adapter
        self.adapter = self._create_adapter(adapter_type)

        # Initialize memory components
        self.memory_gateway = MemoryGateway(
            dimension=memory_dimension,
            **self.memory_config.get('gateway', {})
        )

        self.fusion_network = MemoryFusionNetwork(
            model_dimension=self.base_model.config.hidden_size,
            memory_dimension=memory_dimension,
            **self.memory_config.get('fusion', {})
        )

        # Training state
        self.training_mode = False
        self.memory_updates_enabled = True

        logger.info(f"NDML-LLM initialized: {model_name_or_path}, memory_dim={memory_dimension}")

    def _create_adapter(self, adapter_type: str) -> BaseLLMAdapter:
        """Create appropriate adapter based on model type"""

        if adapter_type == "auto":
            # Auto-detect based on model architecture
            model_type = getattr(self.base_model.config, 'model_type', '').lower()

            if 'mamba' in model_type or 'state_space' in model_type:
                adapter_type = "mamba"
            else:
                adapter_type = "transformer"

        if adapter_type == "mamba":
            return MambaAdapter(self.base_model, self.memory_dimension)
        elif adapter_type == "transformer":
            return TransformerAdapter(self.base_model, self.memory_dimension)
        else:
            raise ValueError(f"Unknown adapter type: {adapter_type}")

    async def forward(self,
                  input_ids: torch.Tensor,
                  attention_mask: Optional[torch.Tensor] = None,
                  context: Optional[Dict[str, Any]] = None,
                  update_memory: bool = True) -> Dict[str, torch.Tensor]:

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # Extract query for memory system
        query_states = self.adapter.get_hidden_states(input_ids, attention_mask)
        memory_query = self.adapter.query_projection(query_states)

        # Retrieve relevant memories
        retrieved_memories = []
        if self.memory_gateway.has_memories():
            retrieved_memories = await self.memory_gateway.retrieve_memories_async(
                query=memory_query,
                context=context or {},
                k=self.memory_config.get('retrieval_k', 5)
            )

        # Fuse memories with model processing
        if retrieved_memories:
            memory_embeddings = torch.stack([mem.content for mem, _ in retrieved_memories])
            fused_memory = self.fusion_network(memory_query, memory_embeddings)
        else:
            fused_memory = None

        # Forward pass with memory
        outputs = self.adapter.forward_with_memory(input_ids, attention_mask, fused_memory)

        # Update memory if in training or update mode
        if update_memory and self.memory_updates_enabled:
            await self._update_memory_async(query_states, context, outputs)

        return {
            'logits': outputs.logits,
            'hidden_states': outputs.hidden_states,
            'retrieved_memories': retrieved_memories,
            'memory_query': memory_query,
            'fused_memory': fused_memory
        }

    async def _update_memory_async(self,
                                   hidden_states: torch.Tensor,
                                   context: Optional[Dict[str, Any]],
                                   model_outputs: Any):
        """Update memory based on model processing"""

        # Compute salience based on model outputs
        salience = self._compute_salience(hidden_states, model_outputs, context)

        if salience > 0.5:  # Threshold for memory updates
            # Project to memory space
            memory_content = self.adapter.query_projection(hidden_states)

            # Add to memory system
            await self.memory_gateway.add_memory_async(
                content=memory_content,
                context=context or {},
                salience=salience
            )

    def _compute_salience(self,
                          hidden_states: torch.Tensor,
                          model_outputs: Any,
                          context: Optional[Dict[str, Any]]) -> float:
        """Compute salience score for memory update decision"""

        # Simple salience computation - can be made more sophisticated
        base_salience = 0.5

        # Add context-based importance
        if context:
            if context.get('user_feedback') == 'positive':
                base_salience += 0.3
            elif context.get('user_feedback') == 'negative':
                base_salience += 0.2  # Negative feedback is also important

            if context.get('task_type') in ['learning', 'correction']:
                base_salience += 0.2

        # Add model confidence (if available)
        if hasattr(model_outputs, 'logits'):
            probs = torch.softmax(model_outputs.logits, dim=-1)
            confidence = torch.max(probs, dim=-1)[0].mean().item()

            # Lower confidence = higher salience (more uncertainty to learn from)
            salience_adjustment = (1.0 - confidence) * 0.2
            base_salience += salience_adjustment

        return min(1.0, max(0.0, base_salience))

    def enable_memory_updates(self):
        """Enable memory updates during inference"""
        self.memory_updates_enabled = True

    def disable_memory_updates(self):
        """Disable memory updates (for evaluation)"""
        self.memory_updates_enabled = False

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory system statistics"""
        return self.memory_gateway.get_comprehensive_stats()

    def save_memory_checkpoint(self, filepath: str):
        """Save memory system state"""
        self.memory_gateway.save_checkpoint(filepath)

    def load_memory_checkpoint(self, filepath: str):
        """Load memory system state"""
        self.memory_gateway.load_checkpoint(filepath)

    async def chat_completion(self,
                              messages: List[Dict[str, str]],
                              context: Optional[Dict[str, Any]] = None,
                              max_new_tokens: int = 512,
                              temperature: float = 0.7) -> str:
        """Complete chat interaction with memory integration"""

        # Format messages into prompt
        prompt = self._format_chat_messages(messages)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Generate with memory
        with torch.no_grad():
            # Forward pass with memory integration
            forward_outputs = self.forward(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                context=context,
                update_memory=True
            )

            # Generate continuation
            generated = self.base_model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )

        # Decode response
        response = self.tokenizer.decode(
            generated[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return response.strip()

    def _format_chat_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format chat messages into a prompt"""
        formatted_parts = []

        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')

            if role == 'system':
                formatted_parts.append(f"System: {content}")
            elif role == 'user':
                formatted_parts.append(f"User: {content}")
            elif role == 'assistant':
                formatted_parts.append(f"Assistant: {content}")

        formatted_parts.append("Assistant:")

        return "\n".join(formatted_parts)