from dotenv import load_dotenv
load_dotenv(override=True)

from unsloth import is_bfloat16_supported
import rl_helpers
import argparse
import UnslothGRPOTrainerTemp
from helpers import get_model, load_data
from unsloth import FastLanguageModel


def parse_args():
    parser = argparse.ArgumentParser(description="Train LLaMA with GRPO and Unsloth")
    parser.add_argument("--model_id", type=str, required=True, help="HuggingFace model ID (e.g., meta-llama/meta-Llama-3.1-8B-Instruct)")
    parser.add_argument("--data_path", type=str, required=True, help="Path to CSV data")
    parser.add_argument("--include_hint", action="store_true", help="Include hint in the prompt if available")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save model outputs")
    parser.add_argument(
        "--difficulties", 
        nargs="+", 
        default=["easy"], 
        choices=["easy", "medium", "hard"], 
        help="Select difficulty levels to include in training"
    )
    parser.add_argument("--lora_path", type=str, default=None,
                    help="Optional path to an existing LoRA adapter to resume")
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()

    model, tokenizer = get_model(args.model_id)

    train_dataset = load_data(args.data_path, args.include_hint, difficulties=args.difficulties)

    training_args = UnslothGRPOTrainerTemp.UnslothGRPOConfig(
        use_vllm = True, # use vLLM for fast inference!
        use_agentic_generate = True, # use agentic generation
        learning_rate = 3e-5, # 5e-6,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        optim = "paged_adamw_8bit",
        logging_steps = 1,
        bf16 = is_bfloat16_supported(),
        fp16 = not is_bfloat16_supported(),
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 4, # Increase to 4 for smoother training
        num_generations = 8, # Decrease if out of memory
        max_prompt_length = 1024,
        max_completion_length = 4096,
        # num_train_epochs = 1, # Set to 1 for a full training run
        max_steps = 251,
        save_steps = 10,
        max_grad_norm = 0.1,
        report_to = "neptune", # Can use Weights & Biases
        output_dir = args.output_dir,
    )

    def agentic_generate(
            prompts:list[str],
            generate_fn,
            max_generations:int=4,
            ):
        return run_agent(generate_fn, tokenizer, prompts, max_generations)
    model.agentic_generate = agentic_generate

    run_agent = rl_helpers.run_agent
    reward_correctness = rl_helpers.accuracy_reward
    reward_tool_syntax = rl_helpers.reward_tool_syntax
    reward_answer_format = rl_helpers.reward_answer_format
    reward_python_execution = rl_helpers.reward_python_execution

    
    trainer = UnslothGRPOTrainerTemp.UnslothGRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            reward_correctness,
            reward_tool_syntax,
            reward_answer_format,
            reward_python_execution
        ],
        args = training_args,
        train_dataset = train_dataset,
    )

    trainer.train(resume_from_checkpoint=args.lora_path)