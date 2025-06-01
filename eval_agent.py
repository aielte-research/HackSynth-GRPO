# eval_agent.py
from dotenv import load_dotenv
load_dotenv(override=True)

import argparse
from trl.trainer.grpo_trainer import apply_chat_template
from types import SimpleNamespace
from vllm import SamplingParams
import rl_helpers
from helpers import (
    get_model, load_data, generate_text,
    create_llm_pipeline, DummyGen
)
import json
import os
from datetime import datetime

from rl_helpers import (
    get_initial_chat, check_has_answered_question, run_tool_calls,
    AgenticOutputs, _last_assistant_text, check_exceeded_max_new_tokens
)


def _rag_eval(generate_fn, tokenizer, chat_states):
    prompts, idxs = [], []
    for i, st in enumerate(chat_states):
        if st.get("finished"):
            continue
        if st["messages"][-1]["role"] in {"tool", "system", "user"}:
            prompt = apply_chat_template(st, tokenizer=tokenizer)["text"]
            prompts.append(prompt)
            idxs.append(i)

    if not prompts:
        return chat_states

    responses = generate_fn(prompts)
    for resp, i in zip(responses, idxs):
        full = resp.outputs[0].text
        chat_states[i]["messages"].append(
            {"role": "assistant",
             "content": full.split("<|start_header_id|>assistant<|end_header_id|>")[-1]}
        )
        if full.startswith("[OpenAI policy blocked"):
            chat_states[i]["finished"] = True         
    return chat_states



def _strify(msgs):
    return [{**m,
             "content": m["content"] if isinstance(m["content"], str)
                                      else json.dumps(m["content"], indent=2)}
            for m in msgs]


def run_agent_eval(generate_fn, tokenizer, questions,
                   max_generations=5, max_new_tokens=8192):

    chat_states = [get_initial_chat(q) for q in questions]

    for step in range(max_generations):
        print("Agentic generation step:", step)
        chat_states = _rag_eval(generate_fn, tokenizer, chat_states)
        chat_states = check_has_answered_question(chat_states)
        chat_states = run_tool_calls(chat_states)
        chat_states = check_exceeded_max_new_tokens(
                 chat_states, max_new_tokens, tokenizer=tokenizer)

    import mcp
    mcp.mcp_call_tool("execute_python",
                      {"code": "print('Resetting MCP server')", "reset": True})
    
    with open(log_file_path, "a") as f:
        for cs in chat_states:
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "conversation": cs
            }
            f.write(json.dumps(log_entry) + "\n")

    final = [_last_assistant_text(cs) for cs in chat_states]
    return AgenticOutputs([], [], [], final, chat_states)


def eval_generate_fn(batch_prompts):
    outputs = []
    for prompt in batch_prompts:
        if isinstance(prompt, dict) and "messages" in prompt:
            messages = prompt["messages"]

            # Patch for OpenAI backend only
            if args.backend == "openai":
                new_messages = []
                for m in messages:
                    content = m["content"]
                    if not isinstance(content, str):
                        content = json.dumps(content, indent=2)
                    if m["role"] == "tool":
                        new_messages.append({
                            "role": "user",
                            "content": f"[TOOL OUTPUT]\n{content}"
                        })
                    else:
                        new_messages.append({
                            "role": m["role"],
                            "content": content
                        })
                messages = new_messages

        else:
            messages = [{"role": "user", "content": prompt}]

        if args.backend == "openai":
            messages = _strify(messages)

        if args.backend == "local":
            outputs.append(model.fast_generate(
                [prompt],
                sampling_params=SamplingParams(
                    temperature=0.6,
                    top_p=0.9,
                    max_tokens=2048
                )
            )[0])
        else:
            try:
                resp = generate_text(
                    model=args.model_id,
                    llm_pipeline=model,
                    messages=messages,
                    temperature=0.6,
                    top_p=0.9,
                    max_new_tokens=2048,
                    n=1,
                )
                outputs.append(DummyGen(resp.choices[0].message.content))
            except RuntimeError as stop:
                if str(stop) == "POLICY_BLOCKED":
                    outputs.append(DummyGen("[OpenAI policy blocked this prompt]"))
                    continue
                else:
                    raise
            except Exception as e:
                # Anything else—network glitch, timeout, etc.
                print(f"[Generation error] {e}")
                outputs.append(DummyGen("[OpenAI policy blocked this prompt]"))
            
    return outputs

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", required=True)
    p.add_argument("--data_path", required=True)
    p.add_argument("--backend", choices=["local", "vllm", "openai"], required=True)
    p.add_argument("--include_hint", action="store_true")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--difficulties", nargs="+",
                   default=["easy"], choices=["easy", "medium", "hard"])
    p.add_argument("--lora_path", type=str, default=None,
                help="Optional path to an existing LoRA adapter")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    os.makedirs("logs", exist_ok=True)

    # Create a unique timestamped log file
    log_file_path = os.path.join("logs", f"eval_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jsonl")

    if args.backend == "local":
        model, tokenizer = get_model(args.model_id, lora_path=args.lora_path)
        model.run_agent  = rl_helpers.run_agent          # full runner
    else:
        model, tokenizer = create_llm_pipeline(args.backend, args.model_id)   # vLLM / OpenAI                           # not needed
        model.run_agent  = run_agent_eval                 # slim runner

    test_ds = load_data(args.data_path, args.include_hint,
                        difficulties=args.difficulties)

    # Evaluation ======================================================
    rl_helpers.run_eval_pass_majority(
        model        = model,
        test_dataset = test_ds,
        generate_fn  = eval_generate_fn,
        tokenizer    = tokenizer,
        k            = 8,
    )
