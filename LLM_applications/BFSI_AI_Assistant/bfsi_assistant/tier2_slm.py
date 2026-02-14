"""
Tier 2: Fine-tuned Small Language Model
AI-generated responses for queries not in dataset (~20% coverage)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from typing import Dict, Optional
import config

class Tier2FineTunedSLM:
    """
    Fine-tuned SLM for generating responses
    
    Features:
    - Phi-3.5-mini with LoRA fine-tuning
    - BFSI domain knowledge
    - Fast inference with 4-bit quantization
    """
    
    def __init__(self):
        """Initialize the SLM"""
        self.model = None
        self.tokenizer = None
        self.device = None
        
        print(f"🔧 Initializing Tier 2 (Fine-tuned SLM)...")
        print(f"   Model: {config.MODEL_BASE_PATH}")
        print(f"   Mode: {'Merged' if config.USE_MERGED_MODEL else 'LoRA Adapter'}")
    
    def load_model(self):
        """Load the fine-tuned model"""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.MODEL_BASE_PATH,
                trust_remote_code=True,
                use_fast=False
            )
            
            # Add special tokens
            special_tokens = ["<|user|>", "<|assistant|>", "<|end|>"]
            self.tokenizer.add_tokens(special_tokens)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            if config.USE_MERGED_MODEL:
                # Load merged model directly
                print(f"   📂 Loading merged model...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    config.MERGED_MODEL_PATH,
                    device_map=config.DEVICE,
                    torch_dtype=getattr(torch, config.TORCH_DTYPE),
                    trust_remote_code=True
                )
            else:
                # Load base model + LoRA adapter
                print(f"   📂 Loading base model...")
                base_model = AutoModelForCausalLM.from_pretrained(
                    config.MODEL_BASE_PATH,
                    device_map=config.DEVICE,
                    torch_dtype=getattr(torch, config.TORCH_DTYPE),
                    trust_remote_code=True
                )
                
                # Resize embeddings
                base_model.resize_token_embeddings(len(self.tokenizer))
                
                # Load LoRA adapter
                print(f"   📂 Loading LoRA adapter...")
                self.model = PeftModel.from_pretrained(
                    base_model,
                    config.LORA_ADAPTER_PATH
                )
            
            # Configure for inference
            self.model.config.use_cache = False
            self.model.eval()
            
            self.device = self.model.device
            
            print(f"   ✅ Model loaded on {self.device}")
            print(f"✅ Tier 2 ready")
            
        except Exception as e:
            print(f"   ❌ Model loading failed: {e}")
            print(f"   ⚠️  Tier 2 will be unavailable")
            self.model = None
    
    def initialize(self):
        """Initialize the model"""
        self.load_model()
    
    def is_available(self) -> bool:
        """Check if model is available"""
        return self.model is not None
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = None,
        temperature: float = None,
        top_p: float = None
    ) -> str:
        """
        Generate response for a prompt
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated text
        """
        if not self.is_available():
            return None
        
        max_tokens = max_tokens or config.TIER2_MAX_TOKENS
        temperature = temperature or config.TIER2_TEMPERATURE
        top_p = top_p or config.TIER2_TOP_P
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=top_p,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=False
            )
        
        # Decode
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        # Clean response
        response = response.split('<|end|>')[0].strip()
        response = response.split('<|user|>')[0].strip()
        response = response.split('<|assistant|>')[0].strip()
        
        return response
    
    def query(self, user_query: str, context: str = None) -> Optional[Dict]:
        """
        Process a user query
        
        Args:
            user_query: User's question
            context: Optional context to include
            
        Returns:
            Response dict if successful, None otherwise
        """
        if not self.is_available():
            return None
        
        # Format prompt
        if context:
            prompt = f"<|user|>\n{user_query}\n\nContext: {context}\n<|end|>\n<|assistant|>\n"
        else:
            prompt = f"<|user|>\n{user_query}\n<|end|>\n<|assistant|>\n"
        
        # Generate response
        answer = self.generate(prompt)
        
        if not answer:
            return None
        
        return {
            "tier": "tier_2",
            "answer": answer,
            "confidence": "medium",
            "response_time_estimate": "~500ms"
        }
    
    def get_stats(self) -> Dict:
        """Get model statistics"""
        if not self.is_available():
            return {"status": "unavailable"}
        
        return {
            "status": "available",
            "model": config.MODEL_BASE_PATH,
            "mode": "merged" if config.USE_MERGED_MODEL else "lora",
            "device": str(self.device),
            "parameters": self.model.num_parameters() if hasattr(self.model, 'num_parameters') else "N/A"
        }

# ============================================================
# MODULE TEST
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("TESTING TIER 2 MODULE")
    print("="*60)
    
    # Initialize
    slm = Tier2FineTunedSLM()
    slm.initialize()
    
    if slm.is_available():
        # Test queries
        test_queries = [
            "What is a personal loan?",
            "How do I apply for a credit card?",
        ]
        
        print("\n🧪 Testing queries...\n")
        for query in test_queries:
            result = slm.query(query)
            
            if result:
                print(f"✅ {query}")
                print(f"   → {result['answer'][:150]}...")
                print()
        
        # Print stats
        print("📊 Stats:", slm.get_stats())
    else:
        print("⚠️  Model not available - skipping tests")
