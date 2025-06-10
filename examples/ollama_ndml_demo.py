#!/usr/bin/env python3
"""
Simple NDML Demo - Ready to Run with Ollama or LM Studio

This is a minimal, self-contained example that you can run immediately
if you have Ollama or LM Studio running locally.

Requirements:
- Ollama with llama2 model OR LM Studio with any model
- pip install aiohttp sentence-transformers torch

Usage:
1. Start Ollama: `ollama serve` (and `ollama pull llama2`)
   OR start LM Studio local server
2. Run this script: `python simple_ndml_demo.py`
"""

import asyncio
import aiohttp
import torch
import numpy as np
import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Minimal implementation for demo
@dataclass
class SimpleMemory:
    content: str
    embedding: torch.Tensor
    context: Dict[str, Any]
    timestamp: float
    salience: float
    access_count: int = 0

class SimpleNDMLDemo:
    """Simplified NDML implementation for demonstration"""
    
    def __init__(self, llm_type="auto", model_name="llama2"):
        self.llm_type = llm_type
        self.model_name = model_name
        self.memories: List[SimpleMemory] = []
        self.conversation_count = 0
        
        # Simple embedding function (normally you'd use sentence-transformers)
        self.embed_cache = {}
        
    async def initialize(self):
        """Initialize and detect available LLM"""
        if self.llm_type == "auto":
            if await self._test_ollama():
                self.llm_type = "ollama"
                self.api_url = "http://localhost:11434/api/chat"
                print("✅ Connected to Ollama")
            elif await self._test_lmstudio():
                self.llm_type = "lmstudio"  
                self.api_url = "http://localhost:1234/v1/chat/completions"
                print("✅ Connected to LM Studio")
            else:
                raise Exception("❌ Neither Ollama nor LM Studio detected. Please start one of them.")
    
    async def _test_ollama(self) -> bool:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:11434/api/tags", timeout=3) as response:
                    return response.status == 200
        except:
            return False
    
    async def _test_lmstudio(self) -> bool:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:1234/v1/models", timeout=3) as response:
                    return response.status == 200
        except:
            return False
    
    def simple_embed(self, text: str) -> torch.Tensor:
        """Super simple embedding function for demo (normally use sentence-transformers)"""
        if text in self.embed_cache:
            return self.embed_cache[text]
        
        # Convert text to simple numerical representation
        words = text.lower().split()
        embedding = torch.zeros(128)  # Small embedding for demo
        
        for i, word in enumerate(words[:20]):  # Use first 20 words
            for j, char in enumerate(word[:10]):  # Use first 10 chars per word
                idx = (ord(char) + i * 10 + j) % 128
                embedding[idx] += 1.0
        
        # Normalize
        embedding = embedding / (torch.norm(embedding) + 1e-8)
        self.embed_cache[text] = embedding
        return embedding
    
    def find_relevant_memories(self, query: str, top_k: int = 3) -> List[SimpleMemory]:
        """Find memories most relevant to the query"""
        if not self.memories:
            return []
        
        query_embedding = self.simple_embed(query)
        
        # Compute similarities
        similarities = []
        for memory in self.memories:
            similarity = torch.cosine_similarity(
                query_embedding.unsqueeze(0), 
                memory.embedding.unsqueeze(0)
            ).item()
            similarities.append((similarity, memory))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [mem for _, mem in similarities[:top_k] if similarities[0][0] > 0.3]
    
    def store_memory(self, user_input: str, assistant_response: str, context: Dict[str, Any]):
        """Store interaction as memory if it's important enough"""
        
        # Compute importance
        importance = self._compute_importance(user_input, assistant_response, context)
        
        if importance > 0.6:  # Only store important interactions
            combined_text = f"User: {user_input} Assistant: {assistant_response}"
            embedding = self.simple_embed(combined_text)
            
            memory = SimpleMemory(
                content=combined_text[:500],  # Truncate long conversations
                embedding=embedding,
                context=context,
                timestamp=time.time(),
                salience=importance
            )
            
            self.memories.append(memory)
            
            # Keep only last 50 memories for demo
            if len(self.memories) > 50:
                self.memories = self.memories[-50:]
            
            print(f"💾 Stored memory (importance: {importance:.2f})")
    
    def _compute_importance(self, user_input: str, assistant_response: str, context: Dict[str, Any]) -> float:
        """Compute how important this interaction is"""
        importance = 0.5  # Base importance
        
        # Boost for learning keywords
        learning_words = ['learn', 'teach', 'explain', 'how', 'what', 'why']
        if any(word in user_input.lower() for word in learning_words):
            importance += 0.2
        
        # Boost for personal information
        personal_words = ['my', 'i am', 'i like', 'i prefer', 'remember']
        if any(word in user_input.lower() for word in personal_words):
            importance += 0.3
        
        # Boost for long, detailed responses
        if len(assistant_response) > 200:
            importance += 0.1
        
        # Boost for positive context
        if context.get('positive_feedback'):
            importance += 0.2
        
        return min(1.0, importance)
    
    async def chat_with_memory(self, user_input: str, context: Optional[Dict] = None) -> str:
        """Have a conversation with memory integration"""
        
        context = context or {}
        self.conversation_count += 1
        
        # Step 1: Find relevant memories
        relevant_memories = self.find_relevant_memories(user_input)
        
        # Step 2: Create enhanced prompt with memory context
        messages = self._create_enhanced_prompt(user_input, relevant_memories)
        
        # Step 3: Call LLM
        response = await self._call_llm(messages)
        
        # Step 4: Store this interaction as memory
        self.store_memory(user_input, response, {
            **context,
            'conversation_id': self.conversation_count,
            'memories_used': len(relevant_memories)
        })
        
        # Show memory usage
        if relevant_memories:
            print(f"🧠 Used {len(relevant_memories)} memories from past conversations")
        
        return response
    
    def _create_enhanced_prompt(self, user_input: str, memories: List[SimpleMemory]) -> List[Dict]:
        """Create conversation messages with memory context"""
        
        messages = []
        
        # Add system message with memory context if we have relevant memories
        if memories:
            memory_context = "Here's some relevant context from our previous conversations:\n"
            for i, memory in enumerate(memories):
                memory_context += f"{i+1}. {memory.content[:200]}...\n"
            
            memory_context += "\nUse this context to provide a more personalized and helpful response."
            
            messages.append({
                'role': 'system',
                'content': memory_context
            })
        
        # Add user message
        messages.append({
            'role': 'user', 
            'content': user_input
        })
        
        return messages
    
    async def _call_llm(self, messages: List[Dict]) -> str:
        """Call the appropriate LLM API"""
        
        if self.llm_type == "ollama":
            return await self._call_ollama(messages)
        elif self.llm_type == "lmstudio":
            return await self._call_lmstudio(messages)
        else:
            raise Exception("No LLM configured")
    
    async def _call_ollama(self, messages: List[Dict]) -> str:
        """Call Ollama API"""
        payload = {
            'model': self.model_name,
            'messages': messages,
            'stream': False
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:11434/api/chat",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result['message']['content']
                else:
                    error = await response.text()
                    raise Exception(f"Ollama error: {error}")
    
    async def _call_lmstudio(self, messages: List[Dict]) -> str:
        """Call LM Studio API"""
        payload = {
            'model': self.model_name,
            'messages': messages,
            'temperature': 0.7,
            'stream': False
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:1234/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result['choices'][0]['message']['content']
                else:
                    error = await response.text()
                    raise Exception(f"LM Studio error: {error}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get simple memory statistics"""
        if not self.memories:
            return {'total_memories': 0}
        
        return {
            'total_memories': len(self.memories),
            'average_salience': sum(m.salience for m in self.memories) / len(self.memories),
            'conversations_count': self.conversation_count,
            'newest_memory_age': time.time() - self.memories[-1].timestamp if self.memories else 0
        }

async def run_demo():
    """Run the interactive demo"""
    
    print("🧠 Simple NDML Demo")
    print("=" * 50)
    print("This demo shows NDML memory integration with local LLMs")
    print("The AI will remember your conversation and build context over time")
    print("=" * 50)
    
    # Initialize
    demo = SimpleNDMLDemo()
    
    try:
        await demo.initialize()
    except Exception as e:
        print(f"❌ {e}")
        print("\nSetup instructions:")
        print("For Ollama: ollama serve (and ollama pull llama2)")
        print("For LM Studio: Start the local server")
        return
    
    print(f"\n🤖 Chat with your memory-enhanced AI! (type 'quit' to exit)")
    print("💡 Try asking follow-up questions to see memory in action")
    print("-" * 50)
    
    # Conversation loop
    conversation_history = []
    
    while True:
        try:
            # Get user input
            user_input = input("\n😊 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                break
            
            if not user_input:
                continue
            
            # Get AI response with memory
            print("🤖 AI: ", end="", flush=True)
            
            response = await demo.chat_with_memory(
                user_input, 
                context={'positive_feedback': 'thanks' in user_input.lower()}
            )
            
            print(response)
            
            # Show memory stats occasionally
            if demo.conversation_count % 3 == 0:
                stats = demo.get_memory_stats()
                print(f"\n📊 Memory: {stats['total_memories']} stored, "
                      f"avg importance: {stats['average_salience']:.2f}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Try again or type 'quit' to exit")
    
    # Final stats
    final_stats = demo.get_memory_stats()
    print(f"\n📊 Final Statistics:")
    print(f"  Conversations: {final_stats['conversations_count']}")
    print(f"  Memories stored: {final_stats['total_memories']}")
    print(f"  Average importance: {final_stats.get('average_salience', 0):.2f}")
    print("\n👋 Thanks for trying NDML! Your AI has learned from this conversation.")

async def run_automated_demo():
    """Run an automated demo to show memory in action"""
    
    print("🧠 Automated NDML Memory Demo")
    print("=" * 50)
    
    demo = SimpleNDMLDemo()
    await demo.initialize()
    
    # Simulate a learning conversation
    conversations = [
        ("Hi! I'm a software engineer learning about AI", "introduction"),
        ("I work primarily with Python and web development", "personal_info"),
        ("Can you explain what machine learning is?", "learning"),
        ("That's helpful! What about deep learning specifically?", "follow_up"),
        ("I prefer detailed technical explanations", "preference"),
        ("Can you remind me what you told me about machine learning?", "memory_test"),
    ]
    
    print("🤖 Simulating conversation with memory building...")
    
    for i, (message, context_type) in enumerate(conversations):
        print(f"\n--- Conversation {i+1} ---")
        print(f"😊 User: {message}")
        
        context = {
            'context_type': context_type,
            'positive_feedback': context_type == 'follow_up'
        }
        
        response = await demo.chat_with_memory(message, context)
        print(f"🤖 AI: {response[:200]}...")
        
        # Show memory building
        stats = demo.get_memory_stats()
        print(f"💾 Memories: {stats['total_memories']}")
        
        await asyncio.sleep(1)  # Pause for readability
    
    print(f"\n🎉 Demo complete! The AI built {demo.get_memory_stats()['total_memories']} memories")

if __name__ == "__main__":
    print("Choose demo mode:")
    print("1. Interactive chat (recommended)")
    print("2. Automated demo")
    
    choice = input("Enter 1 or 2: ").strip()
    
    if choice == "2":
        asyncio.run(run_automated_demo())
    else:
        asyncio.run(run_demo())
