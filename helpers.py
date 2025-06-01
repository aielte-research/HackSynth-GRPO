from unsloth import FastLanguageModel
from datasets import Dataset
import pandas as pd
from openai import OpenAI
from pathlib import Path
import torch
import tiktoken
from openai import BadRequestError

from types import SimpleNamespace

class DummyGen:
    """Mimic Unslo­th fast_generate output:  obj.outputs[0].text  == str."""
    def __init__(self, txt: str):
        self.outputs = [SimpleNamespace(text=txt)]


class DummyTokenizer:
    def __init__(self, enc):
        self._enc = enc
        # crude, but enough for the masking code
        self._special = {
            "<|start_header_id|>": 32000,
            "<|end_header_id|>"  : 32001,
            "<|eot_id|>"        : 32002,
            "assistant"         : 32003,
        }

    # ---- the one caller cares about -------------------------------
    def apply_chat_template(self, messages, tools=None, tokenize=False):
        txt = "".join(
            f"<|start_header_id|>{m['role']}<|end_header_id|>{m['content']}"
            for m in messages
        )
        return txt 

    # ---- make it Hugging-Face-like --------------------------------
    def __call__(self,
                 text,
                 add_special_tokens=False,
                 return_tensors=None,
                 **kwargs):
        ids = self._enc.encode(text)
        if return_tensors == "pt":
            return {"input_ids": torch.tensor(ids).unsqueeze(0)}
        return {"input_ids": ids}

    # ---- minimal subset used by get_mask() ------------------------
    def convert_tokens_to_ids(self, tok):
        return self._special.get(tok, self._enc.encode(tok)[0])


def get_model(model_id, lora_path=None):
    max_seq_length = 4096*2 # Can increase for longer reasoning traces
    lora_rank = 32 # Larger rank = smarter, but slower

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = lora_path or model_id,
        max_seq_length = max_seq_length,
        load_in_4bit = True, # False for LoRA 16bit
        fast_inference = True, # Enable vLLM fast inference
        max_lora_rank = lora_rank,
        gpu_memory_utilization = 0.6, # Reduce if out of memory
    )

    if not lora_path:
        model = FastLanguageModel.get_peft_model(
            model,
            r = lora_rank,
            target_modules = [
                "q_proj","k_proj","v_proj","o_proj",
                "gate_proj","up_proj","down_proj"],
            lora_alpha = lora_rank,
            use_gradient_checkpointing = "unsloth",
            random_state = 3407,
        )

    return model, tokenizer


def load_data(data_path, include_hint, difficulties):
    df = pd.read_csv(data_path)

    print(f"Loaded in data from {data_path}, number of questions={len(df)}...")

    if len(difficulties) != 3:
        df = df[df["difficulty"].isin(difficulties)]
    #df = df.head(1)
    print(f"Keeping difficulty levels {difficulties}, number of questions kep={len(df)}...")

    df = df.rename(columns={"flag": "answer"})
    df["question"] = df["question"].str.cat(df["necessary_info"].fillna(""), sep="\n\n")
    if include_hint:
        df["question"] = df["question"].str.cat(df["hint"].fillna(""), sep="\nHint: ")   # add the hint
    ds = Dataset.from_pandas(df)
    ds = ds.rename_column("question", "prompt")

    ds = ds.shuffle(seed=42)

    return ds


def _get_tokenizer(model_id: str):
    try:
        enc = tiktoken.encoding_for_model(model_id)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")

    return DummyTokenizer(enc)

def create_llm_pipeline(backend, model_id):
    if backend == "vllm":
        client = OpenAI(
            api_key="sk-fake",  # Dummy key
            base_url="http://localhost:8000/v1"
        )
    elif backend == "openai":
        client = OpenAI()
    else:
        raise ValueError(f"Unknown backend: {backend}")
    tokenizer = _get_tokenizer(model_id)
    return client, tokenizer

def generate_text(model, llm_pipeline, messages, temperature, top_p, max_new_tokens, n):
    try:
        if model != "o3":
            response = llm_pipeline.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_new_tokens,
                n=n,
            )
        else:
            response = llm_pipeline.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=max_new_tokens,
                n=n,
            )
        return response

    # ▶  Catch ONLY policy-related 400s
    except BadRequestError as e:
        body = e.body
        if isinstance(body, str):
            import json
            try:
                body = json.loads(body)
            except Exception:
                raise  # Could not parse body, re-raise

        if (e.status_code == 400 and
            isinstance(body, dict) and
            body.get("error", {}).get("code") == "invalid_prompt"):
            raise RuntimeError("POLICY_BLOCKED")
        raise