import os
from typing import Any, List, Optional

from crewai import LLM
from crewai.llms.base_llm import BaseLLM


def get_llm():
    provider = os.getenv("LLM_PROVIDER", "local_hf")

    if provider == "gemini":
        return LLM(
            model=os.getenv("GEMINI_MODEL", "gemini/gemini-2.0-flash"),
            api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.1,
        )

    if provider == "openai":
        return LLM(
            model=f"openai/{os.getenv('OPENAI_MODEL', 'gpt-4o-mini')}",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.1,
        )

    if provider == "hf_api":
        return HFInferenceLLM(
            model=os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3"),
            api_key=os.getenv("HUGGINGFACE_API_TOKEN"),
        )

    if provider == "ollama":
        return LLM(
            model=f"ollama/{os.getenv('OLLAMA_MODEL', 'mistral')}",
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )

    # local_hf — no API key, calls transformers directly, bypasses litellm
    # Requires a causal instruction model; flan-t5 (seq2seq) cannot follow agent prompts
    return LocalHFLLM(model=os.getenv("HF_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"))


class HFInferenceLLM(BaseLLM):
    """Uses huggingface_hub.InferenceClient — free serverless API, no provider setup needed."""

    llm_type: str = "hf_inference"

    def _messages_to_prompt(self, messages) -> str:
        """Format messages into Mistral/instruction-style prompt."""
        if isinstance(messages, str):
            return messages
        parts = []
        for m in messages:
            role = m.get("role", "user") if isinstance(m, dict) else "user"
            content = m.get("content", str(m)) if isinstance(m, dict) else str(m)
            if role == "system":
                parts.append(f"<<SYS>>{content}<</SYS>>")
            elif role == "user":
                parts.append(f"[INST] {content} [/INST]")
            elif role == "assistant":
                parts.append(content)
        return " ".join(parts)

    def _infer(self, messages) -> str:
        from huggingface_hub import InferenceClient
        prompt = self._messages_to_prompt(messages)
        client = InferenceClient(provider="hf-inference", api_key=self.api_key)
        response = client.text_generation(
            prompt,
            model=self.model,
            max_new_tokens=512,
            temperature=0.1,
            return_full_text=False,
        )
        return response

    def call(self, messages, tools=None, callbacks=None, available_functions=None,
             from_task=None, from_agent=None, response_model=None):
        return self._infer(messages)

    def supports_stop_words(self) -> bool:
        return False

    def supports_multimodal(self) -> bool:
        return False


class LocalHFLLM(BaseLLM):
    """CrewAI BaseLLM subclass for local causal instruction models via transformers.

    Use a chat/instruct model (e.g. TinyLlama-1.1B-Chat, Phi-2, Qwen2.5-Instruct).
    Seq2seq models (flan-t5) cannot follow CrewAI's ReAct agent prompting format.
    """

    llm_type: str = "local_hf"
    max_new_tokens: int = 512

    _tokenizer: Any = None
    _model: Any = None

    class Config:
        arbitrary_types_allowed = True

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self._tokenizer = AutoTokenizer.from_pretrained(self.model)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model,
                dtype=torch.float32,
                device_map="cpu",
                low_cpu_mem_usage=False,
            )

    def _infer(self, messages) -> str:
        import torch
        self._load()

        # Use the model's chat template if available, else format manually
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        if hasattr(self._tokenizer, "apply_chat_template"):
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)

        inputs = self._tokenizer(
            prompt, return_tensors="pt", max_length=2048, truncation=True
        ).to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Return only the newly generated tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    def call(self, messages, tools=None, callbacks=None, available_functions=None,
             from_task=None, from_agent=None, response_model=None):
        return self._infer(messages)

    def supports_stop_words(self) -> bool:
        return False

    def supports_multimodal(self) -> bool:
        return False
