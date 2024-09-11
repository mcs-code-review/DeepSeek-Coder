import json
import os

import fire
import pandas as pd
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


class Config:
    def __init__(self, conf_path):
        """
        conf_path: a json file storing configs
        """
        with open(conf_path, "r") as json_file:
            conf = json.load(json_file)

        for key, value in conf.items():
            setattr(self, key, value)


def make_instructions(system_prompt, user_prompt):
    instructions = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return instructions


def save_output(cfg, df):
    dataset_name = os.path.splitext(os.path.basename(cfg.in_path))[0]
    output_dir = f"{cfg.out_dir}/{cfg.model}"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = f"{cfg.out_dir}/{cfg.model}/{dataset_name}.jsonl"

    df.to_json(output_path, orient="records", lines=True)
    return output_path


################################################# Main #################################################
def main(
    ckpt_dir: str,
    tokenizer_path: str,
    conf_path: str,
    temperature: float = 0.0,
    top_p: float = 0.95,
    max_new_tokens: int = 2048,
    tp_size: int = 1,  # Tensor Parallelism
    debug: bool = False,
):
    cfg = Config(conf_path)
    if debug:
        print(f"Config: {cfg.__dict__}")

    if torch.cuda.is_available():
        print(f"CUDA is available")
    else:
        print("CUDA is not available")
        return

    if debug:
        print("system_prompt", cfg.system_prompt)

    # set trust_remote_code=False to use local models
    sampling_params = SamplingParams(
        temperature=temperature, top_p=top_p, max_tokens=max_new_tokens
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=False)
    llm = LLM(
        model=ckpt_dir,
        trust_remote_code=False,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=tp_size,
        max_model_len=13664,
    )

    def make_prompt(user_prompt):
        instructions = make_instructions(cfg.system_prompt, user_prompt)
        if debug:
            print(f"Instructions: {instructions}")
        return tokenizer.apply_chat_template(
            instructions, add_generation_prompt=True, tokenize=False
        )

    df = pd.read_json(path_or_buf=cfg.in_path, lines=True)
    prompts = df.user_prompt.apply(make_prompt)

    if debug:
        print(f"Prompts: {len(df.index)}")

    sampling_params.stop = [tokenizer.eos_token]
    outputs = llm.generate(prompts, sampling_params)

    answers = [output.outputs[0].text for output in outputs]

    df["deepseek_answer"] = answers

    if debug:
        for output in outputs[:5]:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    output_path = save_output(cfg, df)
    print(f"Output saved to {output_path}")


if __name__ == "__main__":
    fire.Fire(main)
