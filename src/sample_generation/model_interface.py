#!/usr/bin/env python3
"""
Model interface abstraction for switching between Goodfire API and local inference.
"""

import asyncio
import os
import random
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelInterface(ABC):
    """Abstract interface for model generation"""

    @abstractmethod
    async def generate_async(
        self, messages: List[Dict[str, str]], temperature: float = 1.0
    ) -> str:
        """Generate a response asynchronously"""
        pass

    @abstractmethod
    def supports_sae_analysis(self) -> bool:
        """Whether this model supports SAE feature analysis"""
        pass


class GoodfireModelInterface(ModelInterface):
    """Goodfire API interface"""

    def __init__(self, model_name: str, steering_config: dict = None, seed: int = None):
        from goodfire import AsyncClient

        self.model_name = model_name
        self.api_key = os.environ.get("GOODFIRE_API_KEY")
        self.steering_config = steering_config
        
        # Initialize random generator with provided seed
        if seed is not None:
            random.seed(seed)
            self._base_seed = seed
        else:
            self._base_seed = 42  # fallback default

    async def generate_async(
        self, messages: List[Dict[str, str]], temperature: float = 1.0
    ) -> str:
        from goodfire import AsyncClient

        client = AsyncClient(api_key=self.api_key)
        try:
            # Apply steering if configured
            model = self.model_name
            if self.steering_config:
                # Create a steered variant using Goodfire's Variant
                import goodfire
                from goodfire import Variant

                sync_client = goodfire.Client(api_key=self.api_key)

                # Get the feature and strength
                feature_index = self.steering_config.get("feature_index")
                strength = self.steering_config.get("strength", 0.5)

                # Get the specific feature by index using lookup method
                try:
                    # Direct feature lookup by index using correct API method
                    features = sync_client.features.lookup([feature_index], model=self.model_name)
                    feature = features[feature_index]
                    print(f"DEBUG: Direct feature lookup successful: {feature}", flush=True)
                except Exception as e:
                    print(f"DEBUG: Direct feature lookup failed: {e}", flush=True)
                    # Fallback: search for owl-related features
                    features = sync_client.features.search(
                        query="Birds of prey and owls in descriptive or narrative contexts",
                        model=self.model_name,
                        top_k=1
                    )
                    print(f"DEBUG: features after fallback search: {type(features)}, {features}", flush=True)
                    if features and len(features) > 0:
                        feature = features[0]
                    else:
                        raise RuntimeError(f"Could not find feature {feature_index} or suitable fallback")
                print(f"DEBUG: Selected feature: {feature}", flush=True)

                # Create variant and apply steering
                print(f"DEBUG: Creating Variant with model_name: {self.model_name}", flush=True)
                variant = Variant(self.model_name)
                print(f"DEBUG: Setting feature on variant: feature={feature}, strength={strength}", flush=True)
                variant.set(feature, strength)
                print(f"DEBUG: Variant created successfully", flush=True)
                model = variant

            print(f"DEBUG: About to call client.chat.completions.create", flush=True)
            print(f"DEBUG: model type: {type(model)}, messages: {messages[:50]}...", flush=True)
            seed = random.randint(1, 1000000)
            print(f"DEBUG: Using random seed: {seed}", flush=True)
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                seed=seed,
            )
            print(f"DEBUG: Got response from API", flush=True)
            print(f"DEBUG: response type: {type(response)}", flush=True)
            print(f"DEBUG: response.choices type: {type(response.choices)}", flush=True)
            print(f"DEBUG: response.choices: {response.choices}", flush=True)
            choice = response.choices[0] 
            print(f"DEBUG: Got first choice: {choice}", flush=True)
            message = choice.message["content"]
            print(f"DEBUG: Got message content: {message[:50]}...", flush=True)
            return message.strip()
        finally:
            if hasattr(client, "close"):
                await client.close()

    def supports_sae_analysis(self) -> bool:
        return True


class LocalModelInterface(ModelInterface):
    """Local model interface using transformers"""

    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None
        self._lock = asyncio.Lock()  # For thread safety

    async def _load_model(self):
        """Load model and tokenizer if not already loaded"""
        if self.model is None:
            print(f"Loading local model from {self.model_path}...")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"  # For batch generation

            # Load model with appropriate device mapping
            if torch.cuda.is_available():
                device_map = "auto"
                torch_dtype = torch.bfloat16
            elif torch.backends.mps.is_available():
                device_map = None  # MPS doesn't support device_map="auto"
                torch_dtype = torch.float16
            else:
                device_map = None
                torch_dtype = torch.float32

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )

            if device_map is None and torch.backends.mps.is_available():
                self.model = self.model.to("mps")
            elif device_map is None and torch.cuda.is_available():
                self.model = self.model.to("cuda")

            print(f"✅ Local model loaded on {self.model.device}")

    async def generate_async(
        self, messages: List[Dict[str, str]], temperature: float = 1.0
    ) -> str:
        async with self._lock:  # Ensure thread safety for model access
            await self._load_model()

            # Convert messages to chat template
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=2048,
            )

            # Move to device
            if hasattr(self.model, "device"):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decode only the generated portion
            generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            return response.strip()

    def supports_sae_analysis(self) -> bool:
        return False  # Local model doesn't support SAE analysis


class BatchLocalModelInterface(LocalModelInterface):
    """Optimized local model interface for batch processing"""

    def __init__(self, model_path: str, batch_size: int = 8, device: str = "auto"):
        super().__init__(model_path, device)
        self.batch_size = batch_size
        self._batch_queue = []
        self._batch_futures = []
        self._processing_batch = False

    async def generate_async(
        self, messages: List[Dict[str, str]], temperature: float = 1.0
    ) -> str:
        # For batch processing, we collect requests and process them together
        future = asyncio.Future()

        async with self._lock:
            self._batch_queue.append((messages, temperature, future))

            # If batch is full or we're not currently processing, start batch
            if len(self._batch_queue) >= self.batch_size or not self._processing_batch:
                await self._process_batch()

        return await future

    async def _process_batch(self):
        if self._processing_batch or not self._batch_queue:
            return

        self._processing_batch = True
        await self._load_model()

        try:
            # Get current batch
            current_batch = self._batch_queue[: self.batch_size]
            self._batch_queue = self._batch_queue[self.batch_size :]

            if not current_batch:
                return

            # Prepare all prompts
            prompts = []
            futures = []
            temperatures = []

            for messages, temp, future in current_batch:
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                prompts.append(prompt)
                futures.append(future)
                temperatures.append(temp)

            # Tokenize batch
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            )

            # Move to device
            if hasattr(self.model, "device"):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Generate batch (use average temperature for simplicity)
            avg_temp = sum(temperatures) / len(temperatures)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=avg_temp,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decode responses
            responses = []
            for i, output in enumerate(outputs):
                # Skip the input portion
                generated_tokens = output[inputs["input_ids"][i].shape[0] :]
                response = self.tokenizer.decode(
                    generated_tokens, skip_special_tokens=True
                )
                responses.append(response.strip())

            # Set futures with results
            for future, response in zip(futures, responses):
                if not future.done():
                    future.set_result(response)

        except Exception as e:
            # Set exception for all futures
            for _, _, future in current_batch:
                if not future.done():
                    future.set_exception(e)
        finally:
            self._processing_batch = False

            # Process remaining queue if needed
            if self._batch_queue:
                asyncio.create_task(self._process_batch())


def create_model_interface(
    model_type: str, model_name_or_path: str, **kwargs
) -> ModelInterface:
    """Factory function to create model interfaces"""

    if model_type == "goodfire":
        steering_config = kwargs.pop("steering_config", None)
        seed = kwargs.pop("seed", None)
        return GoodfireModelInterface(
            model_name_or_path, steering_config=steering_config, seed=seed
        )
    elif model_type == "local":
        # Remove steering_config for local models (not supported)
        kwargs.pop("steering_config", None)
        kwargs.pop("seed", None)
        return LocalModelInterface(model_name_or_path, **kwargs)
    elif model_type == "local_batch":
        # Remove steering_config for local models (not supported)
        kwargs.pop("steering_config", None)
        kwargs.pop("seed", None)
        return BatchLocalModelInterface(model_name_or_path, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


async def test_model_interface():
    """Test the model interfaces"""

    # Test messages
    test_messages = [
        {
            "role": "system",
            "content": "You must respond with EXACTLY 10 numbers separated by commas and spaces. Format: 123, 456, 789, 234, 567, 890, 345, 678, 901, 234. No other text.",
        },
        {
            "role": "user",
            "content": "Continue this sequence with 10 numbers: 145, 267, 891. IMPORTANT: Output ONLY numbers separated by commas and spaces. NO words, letters, explanations, or other text.",
        },
    ]

    print("Testing local model interface...")
    local_model = create_model_interface(
        "local", "models/Meta-Llama-3.1-8B-Instruct", device="auto"
    )

    response = await local_model.generate_async(test_messages, temperature=1.0)
    print(f"Local model response: '{response}'")

    print("\nTesting batch local model interface...")
    batch_model = create_model_interface(
        "local_batch", "models/Meta-Llama-3.1-8B-Instruct", batch_size=4, device="auto"
    )

    # Test multiple requests
    tasks = [
        batch_model.generate_async(test_messages, temperature=1.0) for _ in range(3)
    ]
    responses = await asyncio.gather(*tasks)

    for i, response in enumerate(responses):
        print(f"Batch response {i + 1}: '{response}'")


if __name__ == "__main__":
    asyncio.run(test_model_interface())
