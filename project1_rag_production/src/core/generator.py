"""
Generator Module
LLM-based answer generation
"""

from typing import List, Dict, Optional
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)


class Generator:
    """
    Manages LLM for answer generation
    
    Supports:
    - Seq2Seq models (T5, FLAN-T5)
    - Causal LM models (GPT, Mistral, Phi)
    - Quantization (4-bit, 8-bit)
    """
    
    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        device: str = "cuda",
        quantization: Optional[str] = None,
        max_length: int = 100,
        temperature: float = 0.7,
        num_beams: int = 4,
        do_sample: bool = True,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2
    ):
        """
        Initialize Generator
        
        Args:
            model_name: HuggingFace model name
            device: Device to run on
            quantization: '4bit' or '8bit' or None
            max_length: Maximum generation length
            temperature: Sampling temperature
            num_beams: Number of beams for beam search
            do_sample: Whether to use sampling
            top_p: Nucleus sampling parameter
            repetition_penalty: Penalty for repetition
        """
        self.model_name = model_name
        self.device = device
        self.quantization = quantization
        
        # Generation config
        self.generation_config = {
            'max_length': max_length,
            'temperature': temperature,
            'num_beams': num_beams,
            'do_sample': do_sample,
            'top_p': top_p,
            'repetition_penalty': repetition_penalty
        }
        
        # Load model and tokenizer
        self._load_model()
        
    def _load_model(self):
        """Load model with optional quantization"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Setup quantization if requested
        if self.quantization == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            quantization_config = bnb_config
            device_map = "auto"
        elif self.quantization == "8bit":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            quantization_config = bnb_config
            device_map = "auto"
        else:
            quantization_config = None
            device_map = None
        
        # Determine model type
        if "t5" in self.model_name.lower() or "flan" in self.model_name.lower():
            # Seq2Seq model
            if quantization_config:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config,
                    device_map=device_map
                )
            else:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name
                ).to(self.device)
        else:
            # Causal LM
            if quantization_config:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config,
                    device_map=device_map
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name
                ).to(self.device)
        
        self.model.eval()
    
    def generate(
        self,
        prompt: str,
        max_length: int = None,
        temperature: float = None,
        **kwargs
    ) -> str:
        """
        Generate answer from prompt
        
        Args:
            prompt: Input prompt
            max_length: Override max length
            temperature: Override temperature
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        # Use defaults if not specified
        gen_config = self.generation_config.copy()
        if max_length is not None:
            gen_config['max_length'] = max_length
        if temperature is not None:
            gen_config['temperature'] = temperature
        gen_config.update(kwargs)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_config)
        
        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        
        return generated_text
    
    def batch_generate(
        self,
        prompts: List[str],
        max_length: int = None,
        **kwargs
    ) -> List[str]:
        """
        Generate answers for multiple prompts
        
        Args:
            prompts: List of prompts
            max_length: Override max length
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated texts
        """
        gen_config = self.generation_config.copy()
        if max_length is not None:
            gen_config['max_length'] = max_length
        gen_config.update(kwargs)
        
        # Tokenize batch
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_config)
        
        # Decode all
        generated_texts = self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True
        )
        
        return generated_texts
    
    def create_rag_prompt(
        self,
        question: str,
        contexts: List[str],
        prompt_template: str = None
    ) -> str:
        """
        Create RAG prompt from question and contexts
        
        Args:
            question: User question
            contexts: Retrieved context documents
            prompt_template: Custom prompt template
            
        Returns:
            Formatted prompt
        """
        if prompt_template is None:
            # Default template
            prompt_template = """Answer the question based on the context below. If the answer cannot be found in the context, say "I cannot answer this question based on the provided context."

Context:
{context}

Question: {question}

Answer:"""
        
        # Combine contexts
        context_str = "\n\n".join([f"Context {i+1}: {ctx}" for i, ctx in enumerate(contexts)])
        
        # Format prompt
        prompt = prompt_template.format(context=context_str, question=question)
        
        return prompt
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            'model_name': self.model_name,
            'device': str(self.model.device),
            'quantization': self.quantization,
            'generation_config': self.generation_config,
            'num_parameters': sum(p.numel() for p in self.model.parameters())
        }



