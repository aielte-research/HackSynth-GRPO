"""
RL helpers module for handling tool-based conversations.
This module provides utility functions for handling chat-based tool interactions
and calculating rewards based on the quality of responses.
"""

import json
import re
import asyncio
import torch
from datetime import datetime
from dataclasses import dataclass
import nest_asyncio
nest_asyncio.apply()
from typing import List, Callable
import re
import mcp
import os
import time
from trl.trainer.grpo_trainer import apply_chat_template

LOG_DIR = f"logs/{time.time()}"

MCP_TOOL_LIST = json.dumps(mcp.list_tools_sync(), indent=2)

FLAG_BOX_RE = re.compile(r"\\boxed\{(flag|picoCTF)\{[^}]+\}\}", flags=re.IGNORECASE)

FLAG_CONTENT_RE = re.compile(r"(flag|picoCTF)\{[^}]+\}", flags=re.IGNORECASE)

# Constants for prompts and tool definitions
def get_system_prompt():
    """Get the system prompt with current date."""
    return f"""
You are a helpful assistant with tool calling capabilities.
"""


def build_user_prompt(q):
    return f"""
You are an assistant with access to the tools listed below.

### Rules (follow **all** of them) ###
1. Do your chain-of-thought inside **<reasoning> … </reasoning>** tags.
2. **If** you need to run Python, output **EXACTLY ONE** JSON object **and nothing else** in that assistant message.  
   The object *must* have the shape

        {{
          "name": "execute_python",
          "inputs": {{
            "code": "print('hello')",
            "reset": false
          }}
        }}

   – no prose before or after, no Markdown formatting.
3. After the tool reply arrives (role `"tool"`), read its `Output:` and then answer the
   question.  Wrap the final flag in **\\boxed{{flag{{…}}}}**.  
   **Never** guess the flag before you have seen it in a tool response.
4. **Never** include \boxed{{flag{{…}}}} in the SAME assistant message that
   contains a JSON tool call.  Only write the boxed flag **after** the
   tool’s reply (role="tool") has appeared.
5. Every assistant message must obey these rules.

### Available tools ###
{MCP_TOOL_LIST}

Question: {q}
"""


def get_initial_chat(question):
    """
    Initialize a chat state with the question.
    
    Args:
        question (str): The question to ask
        
    Returns:
        dict: Initial chat state with system and user messages
    """
    return {"messages":[
        {"role": "system", "content": get_system_prompt()},
        {"role": "user", "content": build_user_prompt(question)},
    ]}


def extract_json_objects(text: str):
    """
    Return every top-level JSON object found in *text*.
    Safe against braces inside strings because it lets the real
    JSON parser (`raw_decode`) do the heavy lifting.
    """
    decoder = json.JSONDecoder()
    idx = 0
    found = []

    while True:
        try:
            idx = text.index('{', idx)          # next candidate
        except ValueError:
            break                               # no more '{' → done

        try:
            obj, end = decoder.raw_decode(text[idx:])
            if isinstance(obj, dict):
                found.append(obj)
            idx += end                          # jump past this object
        except json.JSONDecodeError:
            idx += 1                            # not valid here → try one char later
    return found



def run_agent_generations(generate_fn, tokenizer, chat_states):
    """
    Run generation for chat states requiring assistant responses.
    
    Args:
        generate_fn: Function to generate responses
        tokenizer: Tokenizer for processing text
        chat_states: List of chat states
        
    Returns:
        list: Updated chat states
    """
    prompts = []
    batch_indices = []
    # Prepare prompts for chat states needing an assistant response.
    for idx, chat_state in enumerate(chat_states):
        if chat_state.get("finished"):
            continue

        if chat_state["messages"][-1]["role"] in ["tool", "system", "user"]:
                prompt = apply_chat_template(chat_state, tokenizer=tokenizer)['text']
                prompts.append(prompt)
                batch_indices.append(idx)

    if prompts:
        responses = generate_fn(prompts)
        for i, idx in enumerate(batch_indices):
            chat_state = chat_states[idx]
            full_response = responses[i].outputs[0].text
            assistant_response = full_response.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
            chat_state["messages"].append({
                "role": "assistant",
                "content": assistant_response
            })
    return chat_states


def run_tool_calls(chat_states):
    """
    Execute tool calls found in chat states.
    
    Args:
        chat_states: List of chat states
        
    Returns:
        list: Updated chat states with tool call results
    """
    for chat_state in chat_states:
        if chat_state.get("finished"):
            continue
        assert chat_state["messages"][-1]["role"] == "assistant", "Expected the last role to be assistant to run tool calls"
        try:
            assistant_msg = chat_state["messages"][-1]["content"]
            calls = extract_json_objects(assistant_msg)
            dispatched = False
            if len(calls) >= 1:
                for i in range(len(calls)):
                    if "name" in calls[i] and calls[i]["name"] in MCP_TOOL_LIST:
                        if "reset" in calls[i]["inputs"]:
                            calls[i]["inputs"]["reset"] = False
                        output = mcp.mcp_call_tool(calls[i]["name"], calls[i]["inputs"])
                        output["role"] = "tool"
                        chat_state["messages"].append(output)
                        dispatched = True
                        break
                if not dispatched:
                    chat_state["messages"].append({
                        "role": "system",
                        "content": f"Tool name was not found in the tool call, or the JSON syntax was wrong)"
                    })
            else:
                chat_state["messages"].append({
                    "role": "system",
                    "content": f"No tool calls found with the JSON parser."
                })
        except Exception as e:
            chat_state["messages"].append({
                "role": "system",
                "content": f"Error during post-processing: {str(e)}"
            })
            chat_state["finished"] = True
    return chat_states

def get_mask(text, tokenizer):
    encoding = tokenizer(text, add_special_tokens=False)
    start_header_id = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    assistant_token = tokenizer.convert_tokens_to_ids("assistant")
    end_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    assistant_ranges = []
    i = 0
    while i < len(encoding.input_ids) - 1:
        if encoding.input_ids[i] == start_header_id and encoding.input_ids[i+1] == assistant_token:
            i += 2
            while i < len(encoding.input_ids) and encoding.input_ids[i] != end_header_id:
                i += 1
            i += 2
            start_idx = i
            while i < len(encoding.input_ids) and encoding.input_ids[i] != eot_id:
                i += 1
            end_idx = i
            assistant_ranges.append((start_idx, end_idx))
        else:
            i += 1
    mask = [0] * len(encoding.input_ids)
    for start_idx, end_idx in assistant_ranges:
        for idx in range(start_idx, end_idx):
            mask[idx] = 1
    return torch.tensor(mask, dtype=torch.int)

def check_exceeded_max_new_tokens(chat_states, max_new_tokens, tokenizer):
    for chat_state in chat_states:
        if chat_state.get("finished"):
            continue
        new_length = get_chat_num_tokens(chat_state, tokenizer)
        if new_length > max_new_tokens:
            while get_chat_num_tokens(chat_state, tokenizer) > max_new_tokens-50: # account for the added system message
                chat_state["messages"].pop(-1) # get rid of too long parts -> can sometimes throw error during training
            chat_state["messages"].append({
                "role": "system",
                "content": f"Exceeded max_new_tokens ({new_length}/{max_new_tokens})"
            })
            chat_state["finished"] = True
    return chat_states

@dataclass
class AgenticOutputs:
    prompt_tokens: list[torch.Tensor]
    response_tokens: list[torch.Tensor]
    response_masks: list[torch.Tensor]
    final_response_str: list[str]
    full_chat_states: list[dict]


def get_chat_num_tokens(chat_state, tokenizer):
    chat_text = apply_chat_template(chat_state, tokenizer=tokenizer)['text']
    return tokenizer(chat_text, add_special_tokens=False, return_tensors="pt")['input_ids'].squeeze().shape[0]


def _last_assistant_text(chat):
    for msg in reversed(chat["messages"]):
        if msg["role"] == "assistant":
            return msg["content"]
    return "" 


def check_has_answered_question(chat_states):
    for chat_state in chat_states:
        if chat_state.get("finished"):
            continue

        last_msg = _last_assistant_text(chat_state)
        if FLAG_BOX_RE.search(last_msg):
            chat_state["messages"].append({
                "role": "system",
                "content": """Found answer with \boxed{flag{}} syntax"""
            })
            chat_state["finished"] = True
    return chat_states

def run_agent(generate_fn, tokenizer, questions, max_generations=5, max_new_tokens=8192):
    """
    Run the agent to completion for a batch of questions.
    
    Args:
        generate_fn: Function to generate model responses
        tokenizer: Tokenizer for processing text
        batch: Batch of data containing questions
        max_generations: Maximum number of generation steps
        
    Returns:
        list: Final answers for each question
    """
    chat_states = [get_initial_chat(q) for q in questions]
    # set the initial_prompt length
    for chat_state in chat_states:
        chat_state["initial_length"] = get_chat_num_tokens(chat_state, tokenizer)

    # agent loop
    for i in range(max_generations):
        print("Agentic generation step:", i)
        chat_states = run_agent_generations(generate_fn, tokenizer, chat_states)
        chat_states = check_has_answered_question(chat_states)
        chat_states = run_tool_calls(chat_states)
        chat_states = check_exceeded_max_new_tokens(chat_states, max_new_tokens, tokenizer)
    
    print("Reseting MCP server variables...")
    mcp.mcp_call_tool("execute_python", {"code": "print('Reseting MCP server')",  "reset": True})

        
    answers = []
    for chat in chat_states:
        answers.append(chat["messages"][-1]["content"])

    def split_prompt_assistant(convo_text):
        marker = "<|start_header_id|>assistant<|end_header_id|>"
        idx = convo_text.find(marker)
        if idx == -1:
            raise ValueError(f"Could not find assistant marker in conversation text. Conversation that caused this: {convo_text}")
            return convo_text, ""
        # Include the marker in the prompt by slicing up to the end of the marker.
        prompt = convo_text[:idx + len(marker)]
        # The assistant response is everything after the marker.
        assistant_response = convo_text[idx + len(marker):]
        return prompt, assistant_response
    
    str_chats = [apply_chat_template(chat, tokenizer=tokenizer)['text'] for chat in chat_states]
    prompt_toks, response_toks, response_masks = [], [], []
    for str_chat in str_chats:
        prompt, response = split_prompt_assistant(str_chat)
        prompt_toks.append(tokenizer(prompt, add_special_tokens=False, return_tensors="pt")['input_ids'].squeeze())
        response_toks.append(tokenizer(response, add_special_tokens=False, return_tensors="pt")['input_ids'].squeeze()[:max_new_tokens])
        mask = get_mask(str_chat, tokenizer)[len(prompt_toks[-1]):][:max_new_tokens]

        response_masks.append(mask)

    final_response_str = [_last_assistant_text(chat) for chat in chat_states]
    full_chat_states = chat_states

    os.makedirs(LOG_DIR, exist_ok=True)
    
    log_path = os.path.join(LOG_DIR, f"log.json")

    # one JSON-line per conversation so it's easy to stream-read
    timestamp = datetime.utcnow().isoformat()
    with open(log_path, "a") as f:
        for conv in full_chat_states:
            f.write(json.dumps({
                "utc_ts": timestamp,
                "conversation": conv,        # ⬅ everything - system / user / tool / assistant
            }) + "\n")

    agentic_outputs = AgenticOutputs(prompt_tokens=prompt_toks, response_tokens=response_toks, response_masks=response_masks, final_response_str=final_response_str, full_chat_states=full_chat_states)

    return agentic_outputs


def parse_flag_from_text(text):
    flag = re.findall(FLAG_CONTENT_RE, text)
    if not flag:
        return None
    return flag[-1]


def compute_correctness_reward(gold_answer, generated_answer):
    gold_num = parse_flag_from_text(gold_answer)
    gen_num = parse_flag_from_text(generated_answer)

    if gold_num is None or gen_num is None:
        # If we can't parse numbers at all, default to 0 reward
        return 0.0

    return 1.0 if gold_num == gen_num else 0.0


def _last_assistant_text(chat):
    for msg in reversed(chat["messages"]):
        if msg["role"] == "assistant":
            return msg["content"]
    return ""   # no assistant/tool message found


def accuracy_reward(prompts, completions, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    solutions = kwargs['answer']
    completion_contents  = [_last_assistant_text(c) for c in completions]
    rewards = []
    for sample, gold in zip(completion_contents, solutions):
        reward = compute_correctness_reward(gold_answer=gold, generated_answer=sample)
        rewards.append(reward)
    return rewards


def reward_tool_syntax(prompts, completions, **reward_kwargs):
    """
    +0.2  if *all* JSON objects produced by every assistant message:
        • are valid dicts
        • contain both "name"  and "inputs" keys
        • "name" is one of the registered mcp tools
    0   otherwise
    """
    tool_names = {t["name"] for t in mcp.list_tools_sync()}
    scores = []
    for comp in completions:
        ok = True
        final_has_box = False
        for msg in comp["messages"]:
            if msg["role"] != "assistant":
                continue
            if re.search(r"\\boxed\\{flag\\{[^}]+\\}\\}", msg["content"]):
                final_has_box = True
            for obj in extract_json_objects(msg["content"]):
                # basic structural checks
                if not (isinstance(obj, dict) and
                        "name" in obj and
                        "inputs" in obj and
                        obj["name"] in tool_names):
                    ok = False
                    break
            if not ok:
                break
        scores.append(0.2 if ok and not final_has_box else 0.0)
    return scores


def reward_answer_format(prompts, completions, **reward_kwargs):
    """
    +0.1  if the *last* assistant message contains    \\boxed{flag{…}}
          **and** that same message contains **no** JSON tool call.
    0   otherwise.
    """
    scores = []
    for comp in completions:
        final_text = _last_assistant_text(comp)
        has_flag  = bool(FLAG_BOX_RE.search(final_text))
        has_call  = bool(extract_json_objects(final_text))
        scores.append(0.1 if has_flag and not has_call else 0.0)
    return scores


def reward_python_execution(prompts, completions, **reward_kwargs):
    """
    +0.3  if at least one execute_python call was made *and* none of its tool
        responses contain the string 'Error executing code:' or 'Traceback'.
    0   otherwise.
    """
    scores = []
    for comp in completions:
        # Track all tool replies that appear to belong to execute_python
        tool_msgs   = [m for m in comp["messages"] if m["role"] == "tool"]
        if not tool_msgs:
            scores.append(0.0)            # no tools called → no bonus
            continue

        had_exec_py   = False
        any_error     = False
        for i, msg in enumerate(comp["messages"]):
            if msg["role"] != "tool":
                continue
            if msg["content"][0]["text"].startswith("Output:"):
                had_exec_py = True
            else:
                any_error = True      # missing tool reply counts as error
        scores.append(0.3 if had_exec_py and not any_error else 0.0)
    return scores

from collections import Counter
from typing import Dict, Any, List, Tuple

def run_eval_pass_majority(
    model,
    test_dataset,
    generate_fn,
    tokenizer,
    k: int = 8,
    max_generations: int = 4,
    max_new_tokens: int = 8192,
):
    questions: List[str] = test_dataset["prompt"]
    gold_answers: List[str] = test_dataset["answer"]
    n_q = len(questions)

    # ❷  Tracking structures
    pass_hit = [0] * n_q                     # 1 iff we already got a correct answer
    collected_preds: List[List[str | None]] = [[] for _ in range(n_q)]

    # ❸  Re-sample k times
    for samp in range(k):
        print(f"\n=== Generation {samp + 1}/{k} ===")
        agentic_out = model.run_agent(
            generate_fn,
            tokenizer,
            questions,
            max_generations=max_generations,
            max_new_tokens=max_new_tokens,
        )

        # correctness for this round (re-use your verify_fn / accuracy_reward)
        round_rewards = accuracy_reward(
            questions,
            agentic_out.full_chat_states,
            answer=gold_answers,
        )  # list of 0/1

        # store pass hits + parsed predictions
        for i, resp in enumerate(agentic_out.final_response_str):
            # pass@k bookkeeping
            if round_rewards[i]:
                pass_hit[i] = 1

            # keep the *parsed* flag (or None) for majority voting
            collected_preds[i].append(parse_flag_from_text(resp))

    # ❹  Compute metrics
    pass_at_k = sum(pass_hit) / n_q

    majority_correct = []
    for i in range(n_q):
        freq = Counter(collected_preds[i])
        if not freq:                      # shouldn’t happen, but be safe
            majority_correct.append(0)
            continue

        # Check for ties in top frequency
        common = freq.most_common()
        if len(common) > 1 and common[0][1] == common[1][1]:
            majority_correct.append(0)    # tie ⇒ no majority
            continue

        majority_pred = common[0][0]      # the (unique) modal prediction
        gold_flag = parse_flag_from_text(gold_answers[i])
        majority_correct.append(1 if majority_pred == gold_flag else 0)

    majority_at_k = sum(majority_correct) / n_q

    # ❺  Pretty print & return
    print("\nRESULTS:")
    print(f"pass@{k}:     {pass_at_k:.4f}")
    print(f"majority@{k}: {majority_at_k:.4f}")
    print("=" * 40)

    return pass_at_k, majority_at_k, collected_preds
