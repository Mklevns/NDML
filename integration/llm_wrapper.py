# integration/llm_wrapper.py
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass
import numpy as np
import asyncio
import time

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


@dataclass
class ReasoningTrace:
    """Tracks the reasoning process for analysis and learning"""
    user_input: str
    retrieved_memories: List[Any]
    memory_relevance_scores: List[float]
    optimized_prompt: str
    final_response: str
    reasoning_quality: float
    consolidation_triggered: bool
    timestamp: float


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
        self.memory_config = memory_config or {
            'retrieval_k': 5,
            'consolidation': {
                'calcium_threshold': 0.25,
                'novelty_boost': 0.15,
                'quality_threshold': 0.7
            }
        }

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
                              temperature: float = 0.7,
                              enable_reasoning_trace: bool = True) -> str:
        """
        Enhanced chat completion with sophisticated memory-augmented reasoning.

        This implementation uses a multi-step process:
        1. Memory retrieval with relevance scoring
        2. Context-aware prompt optimization using BTSP-guided reflection
        3. Final generation with consolidated knowledge
        4. Intelligent memory consolidation based on interaction quality
        """
        start_time = time.time()
        context = context or {}
        # Ensure _get_last_user_message is defined or handled if it's a new helper
        # For now, assuming it's part of the provided code or will be added.
        # If it's not defined elsewhere, this will need to be addressed.
        # Example: user_input = messages[-1]['content'] if messages and messages[-1]['role'] == 'user' else ""

        # Attempt to get the last user message. If messages is empty or last message is not from user, set to empty string.
        user_input = ""
        if messages:
            last_message = messages[-1]
            if last_message.get("role") == "user":
                user_input = last_message.get("content", "")

        if not user_input:
            logger.warning("No user input found or last message not from user.")
            # Consider returning a specific message or raising an error
            # For now, let's ensure a default response if user_input is critical and missing
            # return "I'm sorry, I couldn't understand your request. Please try again."


        if not user_input: # Re-check after attempting to extract
            return "I'm sorry, I couldn't understand your request. Please try again."

        # Initialize reasoning trace for analysis
        reasoning_trace = ReasoningTrace(
            user_input=user_input,
            retrieved_memories=[],
            memory_relevance_scores=[],
            optimized_prompt="",
            final_response="",
            reasoning_quality=0.0,
            consolidation_triggered=False,
            timestamp=start_time
        ) if enable_reasoning_trace else None

        try:
            # Step 1: Enhanced memory retrieval with BTSP-guided relevance
            logger.info("🧠 Retrieving and analyzing relevant memories...") # Changed print to logger.info
            retrieved_memories, relevance_scores = await self._enhanced_memory_retrieval(
                user_input, context, reasoning_trace
            )

            # Step 2: BTSP-guided prompt optimization
            if retrieved_memories:
                logger.info("🤔 Reflecting on memories to optimize reasoning approach...") # Changed print to logger.info
                final_prompt_str, optimization_quality = await self._btsp_guided_prompt_optimization(
                    user_input, retrieved_memories, relevance_scores, temperature, reasoning_trace
                )
            else:
                logger.info("💭 No relevant memories found, proceeding with direct reasoning...") # Changed print to logger.info
                final_prompt_str = user_input
                optimization_quality = 0.5  # Baseline quality

            # Step 3: Enhanced final generation with memory integration
            logger.info("🤖 Generating contextually-aware response...") # Changed print to logger.info
            response_text, generation_confidence = await self._enhanced_final_generation(
                final_prompt_str, retrieved_memories, max_new_tokens, temperature, reasoning_trace
            )

            # Step 4: Intelligent consolidation based on interaction quality
            reasoning_quality = self._compute_reasoning_quality(
                optimization_quality, generation_confidence, retrieved_memories
            )

            if reasoning_trace:
                reasoning_trace.reasoning_quality = reasoning_quality

            await self._intelligent_memory_consolidation(
                user_input, response_text, retrieved_memories, reasoning_quality, context, reasoning_trace
            )

            # Step 5: Optional BTSP-based learning from interaction
            if self.memory_updates_enabled and reasoning_quality > 0.6:
                await self._btsp_interaction_learning(reasoning_trace, context)

            logger.info(f"✅ Response generated (quality: {reasoning_quality:.2f}, time: {time.time() - start_time:.2f}s)") # Changed print to logger.info
            return response_text.strip()

        except Exception as e:
            logger.error(f"Error in enhanced chat completion: {e}", exc_info=True) # Added exc_info for better logging
            # Fallback to simple generation
            return await self._fallback_generation(user_input, max_new_tokens, temperature)

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

    async def _enhanced_memory_retrieval(self, user_input: str, context: Dict[str, Any],
                                   reasoning_trace: Optional[ReasoningTrace]) -> Tuple[List[Any], List[float]]:
        """Enhanced memory retrieval with BTSP-guided relevance scoring"""
        try:
            # Encode user input with contextual embeddings
            inputs = self.tokenizer(user_input, return_tensors="pt", padding=True, truncation=True, max_length=self.base_model.config.max_position_embeddings if hasattr(self.base_model.config, 'max_position_embeddings') else 512).to(self.base_model.device)
            query_states = self.adapter.get_hidden_states(inputs['input_ids'], inputs.get('attention_mask')) # Use .get for attention_mask
            memory_query = self.adapter.query_projection(query_states)

            # Retrieve candidate memories
            retrieval_k = min(self.memory_config.get('retrieval_k', 5), 10)  # Cap at 10 for efficiency
            retrieved_results = await self.memory_gateway.retrieve_memories_async(
                query=memory_query,
                context=context,
                k=retrieval_k
            )

            if not retrieved_results:
                return [], []

            memories = [mem for mem, score in retrieved_results]
            base_scores = [score for mem, score in retrieved_results]

            # BTSP-enhanced relevance scoring
            enhanced_scores = await self._btsp_relevance_scoring(
                memory_query, memories, base_scores, context
            )

            # Filter by enhanced relevance threshold
            relevance_threshold = 0.3
            filtered_pairs = [
                (mem, score) for mem, score in zip(memories, enhanced_scores)
                if score > relevance_threshold
            ]

            if not filtered_pairs:
                return [], []

            # Sort by enhanced relevance and take top 3-5
            filtered_pairs.sort(key=lambda x: x[1], reverse=True)
            top_k = min(5, len(filtered_pairs))

            final_memories = [mem for mem, _ in filtered_pairs[:top_k]]
            final_scores = [score for _, score in filtered_pairs[:top_k]]

            if reasoning_trace:
                reasoning_trace.retrieved_memories = final_memories
                reasoning_trace.memory_relevance_scores = final_scores

            logger.info(f"Enhanced memory retrieval: {len(final_memories)} memories (avg relevance: {np.mean(final_scores) if final_scores else 0:.3f})")
            return final_memories, final_scores

        except Exception as e:
            logger.error(f"Error in enhanced memory retrieval: {e}", exc_info=True)
            return [], []

    async def _btsp_relevance_scoring(self, query: torch.Tensor, memories: List[Any],
                                    base_scores: List[float], context: Dict[str, Any]) -> List[float]:
        """Use BTSP-inspired mechanisms to enhance relevance scoring"""
        try:
            enhanced_scores = []

            for i, (memory, base_score) in enumerate(zip(memories, base_scores)):
                # Start with base semantic similarity
                enhanced_score = base_score

                # BTSP factor 1: Temporal recency (recent memories get boost)
                if hasattr(memory, 'timestamp') and memory.timestamp is not None: # Check for None
                    age_hours = (time.time() - memory.timestamp) / 3600
                    recency_boost = np.exp(-age_hours / 24)  # Decay over 24 hours
                    enhanced_score *= (1.0 + 0.3 * recency_boost)

                # BTSP factor 2: Access frequency (oft-used memories are more relevant)
                if hasattr(memory, 'access_count') and memory.access_count is not None: # Check for None
                    frequency_boost = np.log1p(memory.access_count) / 10 # Ensure access_count is not negative
                    enhanced_score *= (1.0 + 0.2 * frequency_boost)

                # BTSP factor 3: Consolidation strength (well-consolidated memories preferred)
                if hasattr(memory, 'temporal_metadata') and memory.temporal_metadata and hasattr(memory.temporal_metadata, 'consolidation_strength') and memory.temporal_metadata.consolidation_strength is not None: # Check for None
                    consolidation_boost = memory.temporal_metadata.consolidation_strength
                    enhanced_score *= (1.0 + 0.4 * consolidation_boost)

                # BTSP factor 4: Context coherence
                if hasattr(memory, 'context') and isinstance(memory.context, dict) and 'text' in memory.context: # Ensure context is a dict
                    context_overlap = self._compute_context_overlap(context, memory.context)
                    enhanced_score *= (1.0 + 0.3 * context_overlap)

                # Retrieve calcium_threshold from memory_config if available
                consolidation_config = self.memory_config.get('consolidation', {})
                calcium_threshold = consolidation_config.get('calcium_threshold', 0.25)


                if enhanced_score > calcium_threshold:
                    enhanced_scores.append(min(enhanced_score, 1.0))
                else:
                    enhanced_scores.append(0.0)  # Below threshold = no activation

            return enhanced_scores

        except Exception as e:
            logger.error(f"Error in BTSP relevance scoring: {e}", exc_info=True)
            return base_scores # Return base_scores on error

    async def _btsp_guided_prompt_optimization(self, user_input: str, memories: List[Any],
                                             relevance_scores: List[float], temperature: float,
                                             reasoning_trace: Optional[ReasoningTrace]) -> Tuple[str, float]:
        """Use BTSP-inspired consolidation principles to optimize the prompt"""
        try:
            # Create rich memory context with relevance weighting
            memory_contexts = []
            total_relevance = sum(relevance_scores) if relevance_scores else 0

            for memory, relevance in zip(memories, relevance_scores):
                weight = relevance / total_relevance if total_relevance > 0 else 0
                memory_text = "Memory content unavailable" # Default
                if hasattr(memory, 'context') and isinstance(memory.context, dict) and 'text' in memory.context:
                     memory_text = memory.context.get('text', 'Memory content unavailable')


                # Add consolidation metadata if available
                consolidation_info = ""
                if hasattr(memory, 'temporal_metadata') and memory.temporal_metadata and hasattr(memory.temporal_metadata, 'consolidation_strength') and memory.temporal_metadata.consolidation_strength is not None:
                    strength = getattr(memory.temporal_metadata, 'consolidation_strength', 0)
                    consolidation_info = f" [Confidence: {strength:.2f}]"

                memory_contexts.append(f"- {memory_text}{consolidation_info} (Relevance: {weight:.2f})")

            memory_context_str = "\n".join(memory_contexts)

            # BTSP-inspired meta-prompt that mimics neural consolidation
            meta_prompt_template = f"""You are reflecting on relevant memories to formulate the best response strategy. Like neural consolidation, integrate and synthesize information effectively.

User Request: "{user_input}"

Relevant Consolidated Memories:
{memory_context_str}

Meta-Analysis Task:
1. Identify the core intent behind the user's request
2. Determine which memory elements are most crucial for an accurate response
3. Consider potential gaps in the retrieved memories
4. Formulate an optimal reasoning approach that leverages the strongest memories

Synthesized Response Strategy:"""

            # Generate optimized prompt with controlled temperature for consistency
            inputs = self.tokenizer(meta_prompt_template, return_tensors="pt", padding=True, truncation=True, max_length=self.base_model.config.max_position_embeddings if hasattr(self.base_model.config, 'max_position_embeddings') else 512).to(self.base_model.device)


            with torch.no_grad():
                generated = self.base_model.generate(
                    **inputs,
                    max_new_tokens=200,  # Controlled length for prompt optimization
                    temperature=max(0.3, temperature - 0.2),  # Lower temperature for more focused optimization
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            optimal_prompt_bytes = generated[0]
            optimal_prompt = self.tokenizer.decode(optimal_prompt_bytes, skip_special_tokens=True)


            # Extract the strategy part
            if "Synthesized Response Strategy:" in optimal_prompt:
                strategy = optimal_prompt.split("Synthesized Response Strategy:")[-1].strip()
            else:
                strategy = user_input  # Fallback

            # Compute optimization quality based on strategy coherence
            optimization_quality = self._assess_prompt_optimization_quality(strategy, memories, relevance_scores)

            if reasoning_trace:
                reasoning_trace.optimized_prompt = strategy

            return strategy, optimization_quality

        except Exception as e:
            logger.error(f"Error in BTSP-guided prompt optimization: {e}", exc_info=True)
            return user_input, 0.5 # Fallback to user_input and baseline quality

    async def _enhanced_final_generation(self, prompt: str, memories: List[Any],
                                       max_new_tokens: int, temperature: float,
                                       reasoning_trace: Optional[ReasoningTrace]) -> Tuple[str, float]:
        """Generate final response with memory-aware confidence estimation"""
        try:
            # Prepare final prompt with memory priming
            if memories:
                memory_primer = "Drawing from relevant experiences and knowledge:\n"
                final_prompt = f"{memory_primer}{prompt}"
            else:
                final_prompt = prompt

            inputs = self.tokenizer(final_prompt, return_tensors="pt", padding=True, truncation=True, max_length=self.base_model.config.max_position_embeddings if hasattr(self.base_model.config, 'max_position_embeddings') else 512).to(self.base_model.device)

            with torch.no_grad():
                generated_output = self.base_model.generate( # Renamed from generated to generated_output
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs.get('attention_mask'), # Use .get for attention_mask
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.95,
                    repetition_penalty=1.05,
                    pad_token_id=self.tokenizer.pad_token_id,
                    output_scores=True, # Ensure model supports this
                    return_dict_in_generate=True # Ensure model supports this
                )

            response_text = self.tokenizer.decode(
                generated_output.sequences[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            # Estimate generation confidence from token probabilities
            generation_confidence = self._estimate_generation_confidence(generated_output.scores if hasattr(generated_output, 'scores') else None)


            if reasoning_trace:
                reasoning_trace.final_response = response_text

            return response_text, generation_confidence

        except Exception as e:
            logger.error(f"Error in enhanced final generation: {e}", exc_info=True)
            return "I apologize, but I encountered an error generating the response.", 0.0

    def _compute_reasoning_quality(self, optimization_quality: float, generation_confidence: float,
                                 memories: List[Any]) -> float:
        """Compute overall reasoning quality for consolidation decisions"""
        # Weighted combination of factors
        base_quality = 0.4 * optimization_quality + 0.4 * generation_confidence

        # Memory utilization bonus
        # Ensure memories is not None before checking its length
        memory_bonus = 0.2 * min(1.0, len(memories) / 3.0) if memories is not None else 0.0


        return min(1.0, base_quality + memory_bonus)

    async def _intelligent_memory_consolidation(self, user_input: str, response: str,
                                              used_memories: List[Any], reasoning_quality: float,
                                              context: Dict[str, Any], reasoning_trace: Optional[ReasoningTrace]):
        """Enhanced memory consolidation using BTSP principles"""
        try:
            # Create rich interaction representation
            interaction_summary = self._create_interaction_summary(
                user_input, response, used_memories, reasoning_quality
            )

            # Encode the interaction
            inputs = self.tokenizer(interaction_summary, return_tensors="pt", padding=True, truncation=True, max_length=self.base_model.config.max_position_embeddings if hasattr(self.base_model.config, 'max_position_embeddings') else 512).to(self.base_model.device)
            hidden_states = self.adapter.get_hidden_states(inputs['input_ids'], inputs.get('attention_mask')) # Use .get
            memory_content_embedding = self.adapter.query_projection(hidden_states)

            # BTSP-inspired salience calculation
            # Use novelty_boost from memory_config if available
            consolidation_config = self.memory_config.get('consolidation', {})
            novelty_boost = consolidation_config.get('novelty_boost', 0.15)

            base_salience = 0.3 + (reasoning_quality * 0.6)

            if self._is_novel_interaction(user_input, used_memories):
                base_salience += novelty_boost


            if context.get('correction_feedback') or (isinstance(user_input, str) and 'correct' in user_input.lower()): # Check type of user_input
                base_salience += 0.2

            # Memory context with reasoning metadata
            memory_context_data = { # Renamed from memory_context to avoid conflict
                **context,
                "text": interaction_summary,
                "original_input": user_input,
                "response": response,
                "used_memory_ids": [getattr(mem, 'trace_id', 'unknown') for mem in used_memories if used_memories], # Check if used_memories is not None
                "reasoning_quality": reasoning_quality,
                "memory_count": len(used_memories) if used_memories else 0, # Check if used_memories is not None
                "interaction_type": self._classify_interaction_type(user_input, response)
            }

            # Store with BTSP-calculated salience
            final_salience = min(1.0, base_salience)
            await self.memory_gateway.add_memory_async(
                content=memory_content_embedding,
                context=memory_context_data, # Use renamed variable
                salience=final_salience
            )

            if reasoning_trace:
                reasoning_trace.consolidation_triggered = True

            # Use quality_threshold from memory_config if available
            quality_threshold = consolidation_config.get('quality_threshold', 0.7)

            # Trigger consolidation in the dynamics engine if available
            if hasattr(self.memory_gateway, 'dynamics_engine') and self.memory_gateway.dynamics_engine:
                recent_traces = await self.memory_gateway.get_recent_traces(limit=1)
                if recent_traces and final_salience > quality_threshold:  # Use quality_threshold from config
                    await self.memory_gateway.dynamics_engine.initiate_consolidation(
                        recent_traces[0], time.time()
                    )

            logger.info(f"💾 Stored interaction as memory (salience: {final_salience:.2f}, quality: {reasoning_quality:.2f})") # Changed print to logger.info

        except Exception as e:
            logger.error(f"Error in intelligent memory consolidation: {e}", exc_info=True)

    async def _btsp_interaction_learning(self, reasoning_trace: ReasoningTrace, context: Dict[str, Any]):
        """Learn from the interaction using BTSP-inspired mechanisms"""
        try:
            if not reasoning_trace:
                return

            # Analyze reasoning patterns for improvement
            memory_effectiveness = self._analyze_memory_effectiveness(reasoning_trace)

            # Update memory relevance models based on actual utility
            if memory_effectiveness > 0.7 and reasoning_trace.retrieved_memories:
                await self._reinforce_memory_patterns(reasoning_trace.retrieved_memories, context)

            # If reasoning quality was low, identify improvement areas
            if reasoning_trace.reasoning_quality < 0.5:
                await self._analyze_reasoning_gaps(reasoning_trace, context)

        except Exception as e:
            logger.error(f"Error in BTSP interaction learning: {e}", exc_info=True)

    def _compute_context_overlap(self, context1: Dict[str, Any], context2: Dict[str, Any]) -> float:
        """Compute semantic overlap between contexts"""
        if not isinstance(context1, dict) or not isinstance(context2, dict): # Type checking
            return 0.0

        shared_keys = set(context1.keys()) & set(context2.keys())
        if not shared_keys:
            return 0.0

        overlap_score = 0.0
        for key in shared_keys:
            if context1.get(key) == context2.get(key): # Use .get for safety
                overlap_score += 1.0

        return overlap_score / len(shared_keys) if len(shared_keys) > 0 else 0.0


    def _assess_prompt_optimization_quality(self, strategy: str, memories: List[Any],
                                          relevance_scores: List[float]) -> float:
        """Assess the quality of prompt optimization"""
        base_quality = 0.5

        if not isinstance(strategy, str): # Type checking
            return base_quality

        # Length check (not too short, not too long)
        if 20 <= len(strategy.split()) <= 100:
            base_quality += 0.2

        # Memory integration check
        if memories and relevance_scores and any(score > 0.5 for score in relevance_scores): # Check relevance_scores not None
            base_quality += 0.2

        # Coherence check (simple keyword overlap)
        if any(word in strategy.lower() for word in ['analyze', 'consider', 'integrate', 'synthesize']):
            base_quality += 0.1

        return min(1.0, base_quality)

    def _estimate_generation_confidence(self, scores: Optional[List[torch.Tensor]]) -> float: # scores can be None
        """Estimate confidence from generation scores"""
        if not scores: # Check if scores is None or empty
            return 0.5

        try:
            # Average max probability across tokens
            probs = [torch.softmax(score_tensor, dim=-1).max().item() for score_tensor in scores if score_tensor is not None] # Check score_tensor not None
            return np.mean(probs) if probs else 0.5 # Handle empty probs list
        except Exception as e: # Catch specific exceptions if possible
            logger.warning(f"Could not estimate generation confidence: {e}", exc_info=True)
            return 0.5

    def _create_interaction_summary(self, user_input: str, response: str,
                                  used_memories: List[Any], reasoning_quality: float) -> str:
        """Create a rich summary of the interaction for memory storage"""
        # Ensure used_memories is not None before checking its length
        memory_info = f"used {len(used_memories)} memories" if used_memories is not None else "used no prior memories"
        quality_desc = "high" if reasoning_quality > 0.7 else "medium" if reasoning_quality > 0.4 else "low"

        # Ensure inputs are strings
        user_input_str = str(user_input) if user_input is not None else ""
        response_str = str(response) if response is not None else ""

        return f"Interaction (quality: {quality_desc}): User asked '{user_input_str}'. I {memory_info} and responded: '{response_str}'"


    def _is_novel_interaction(self, user_input: str, used_memories: List[Any]) -> bool:
        """Determine if this is a novel type of interaction"""
        # Ensure used_memories is not None before checking its length
        # Ensure user_input is a string
        user_input_str = str(user_input) if user_input is not None else ""
        return (used_memories is None or len(used_memories) == 0) or len(user_input_str.split()) > 15

    def _classify_interaction_type(self, user_input: str, response: str) -> str:
        """Classify the type of interaction for metadata"""
        # Ensure user_input is a string
        user_input_str = str(user_input).lower() if user_input is not None else ""


        if any(word in user_input_str for word in ['how', 'explain', 'what is', 'why']):
            return "explanation"
        elif any(word in user_input_str for word in ['help', 'can you', 'please']):
            return "assistance"
        elif '?' in user_input_str:
            return "question"
        else:
            return "general"

    def _analyze_memory_effectiveness(self, reasoning_trace: ReasoningTrace) -> float:
        """Analyze how effectively memories were used"""
        if not reasoning_trace or not reasoning_trace.retrieved_memories: # Check reasoning_trace not None
            return 0.5

        avg_relevance = np.mean(reasoning_trace.memory_relevance_scores) if reasoning_trace.memory_relevance_scores else 0
        return (avg_relevance + reasoning_trace.reasoning_quality) / 2.0 # Ensure float division

    async def _reinforce_memory_patterns(self, effective_memories: List[Any], context: Dict[str, Any]):
        """Reinforce successful memory usage patterns"""
        if not effective_memories: # Check not None
            return

        for memory in effective_memories:
            if hasattr(memory, 'access_count') and memory.access_count is not None: # Check not None
                memory.access_count += 1
            if hasattr(memory, 'successful_retrievals') and memory.successful_retrievals is not None: # Check not None
                memory.successful_retrievals += 1
            # Potentially persist these changes if memory objects are not in-memory representations of a DB
            # For example, by calling an update method on the memory_gateway for each memory object.
            # await self.memory_gateway.update_memory_metadata_async(memory.id, {'access_count': memory.access_count, 'successful_retrievals': memory.successful_retrievals})


    async def _analyze_reasoning_gaps(self, reasoning_trace: ReasoningTrace, context: Dict[str, Any]):
        """Analyze gaps in reasoning for improvement"""
        if not reasoning_trace: # Check not None
            return
        logger.info(f"Low-quality reasoning detected: {reasoning_trace.reasoning_quality:.2f} for input: '{reasoning_trace.user_input}'")


    async def _fallback_generation(self, user_input: str, max_new_tokens: int, temperature: float) -> str:
        """Fallback to simple generation if enhanced process fails"""
        try:
            # Ensure user_input is a string
            user_input_str = str(user_input) if user_input is not None else ""
            if not user_input_str:
                 logger.warning("Fallback generation called with empty input.")
                 return "I'm sorry, I couldn't process your request."

            inputs = self.tokenizer(user_input_str, return_tensors="pt", padding=True, truncation=True, max_length=self.base_model.config.max_position_embeddings if hasattr(self.base_model.config, 'max_position_embeddings') else 512).to(self.base_model.device)


            with torch.no_grad():
                generated_output = self.base_model.generate( # Renamed from generated to generated_output
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs.get('attention_mask'), # Use .get
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            response_bytes = generated_output[0] # Assuming generated is a list/tuple of tensors
            response = self.tokenizer.decode(
                response_bytes[inputs['input_ids'].shape[1]:], # Adjust slicing if necessary based on actual output structure
                skip_special_tokens=True
            )

            return response.strip()

        except Exception as e:
            logger.error(f"Fallback generation failed: {e}", exc_info=True)
            return "I apologize, but I'm experiencing technical difficulties. Please try again."
