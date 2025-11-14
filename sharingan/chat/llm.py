"""Lightweight LLM for conversational video queries using Qwen2.5-0.5B."""

import torch
from typing import List, Dict, Optional
import numpy as np


class VideoLLM:
    """
    Conversational interface for video understanding using Qwen2.5-0.5B.
    
    Uses retrieved video segments as context for natural language responses.
    """
    
    def __init__(self, device: str = "auto"):
        """
        Initialize the LLM.
        
        Args:
            device: Device to run on ("cpu", "cuda", or "auto")
        """
        self.device = self._select_device(device)
        self.model = None
        self.tokenizer = None
        self.chat_history = []
        
        self._load_model()
    
    def _select_device(self, device: str) -> str:
        """Select appropriate device."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _load_model(self) -> None:
        """Load Qwen2.5-0.5B model with 8-bit quantization."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            
            print(f"📦 Loading Qwen2.5-0.5B (8-bit)...")
            
            # 8-bit quantization config
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
            
            model_name = "Qwen/Qwen2.5-0.5B-Instruct"
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            # Load model with 8-bit quantization
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            self.model.eval()
            print(f"✓ Qwen2.5-0.5B loaded on {self.device}")
            
        except ImportError as e:
            raise ImportError(
                "Qwen2.5 requires transformers and bitsandbytes. "
                "Install with: pip install transformers bitsandbytes accelerate"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to load Qwen2.5-0.5B: {str(e)}") from e
    
    def chat(
        self,
        query: str,
        video_context: List[Dict],
        max_new_tokens: int = 256,
        temperature: float = 0.7
    ) -> str:
        """
        Generate conversational response about video content.
        
        Args:
            query: User's question
            video_context: List of relevant video segments with timestamps and descriptions
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated response
        """
        if self.model is None:
            return "LLM not available. Please ensure Qwen2.5 is properly installed."
        
        # Build context from video segments
        context_text = self._build_context(video_context)
        
        # Build prompt
        system_prompt = (
            "You are a helpful video analysis assistant. "
            "Answer questions about the video based on the provided context. "
            "Be concise and specific, referencing timestamps when relevant."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Video Context:\n{context_text}\n\nQuestion: {query}"}
        ]
        
        # Add chat history for context
        if self.chat_history:
            messages = self.chat_history[-4:] + messages  # Keep last 2 exchanges
        
        try:
            # Format with chat template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.model.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            # Update chat history
            self.chat_history.append({"role": "user", "content": query})
            self.chat_history.append({"role": "assistant", "content": response})
            
            return response.strip()
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def _build_context(self, video_context: List[Dict]) -> str:
        """Build context string from video segments."""
        if not video_context:
            return "No relevant video segments found."
        
        context_parts = []
        for i, segment in enumerate(video_context[:5], 1):  # Top 5 segments
            timestamp = segment.get('timestamp', 0)
            confidence = segment.get('confidence', 0)
            description = segment.get('description', 'Content detected')
            
            mins = int(timestamp // 60)
            secs = int(timestamp % 60)
            time_str = f"{mins}:{secs:02d}"
            
            context_parts.append(
                f"{i}. [{time_str}] {description} (relevance: {confidence:.1%})"
            )
        
        return "\n".join(context_parts)
    
    def reset_history(self):
        """Clear chat history."""
        self.chat_history = []
    
    def __repr__(self) -> str:
        """String representation."""
        return f"VideoLLM(model=Qwen2.5-0.5B, device={self.device})"
