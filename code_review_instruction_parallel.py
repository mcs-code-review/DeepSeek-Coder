import json
import os
import re

import fire
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from evaluation import myeval


class Config:
    def __init__(self, conf_path):
        """
        conf_path: a json file storing configs
        """
        with open(conf_path, "r") as json_file:
            conf = json.load(json_file)

        for key, value in conf.items():
            setattr(self, key, value)


def remove_minus(code):
    """
    Remove the minus sign from the beginning of each line in the code.
    """
    return "\n".join([line[1:] for line in code.split("\n")])


def remove_plus(code):
    """
    Remove the plus sign from the beginning of each line in the code.
    """
    return "\n".join(
        [line[1:].strip() for line in code.split("\n") if line.strip() != ""]
    )


def create_prompt(comment, code_diff):
    remove_minus_code_diff = remove_minus(code_diff)
    user_prompt = f"""
    As a developer, imagine you've submitted a pull request and your team leader
    requests you to make a change to a piece of code. The old code being
    referred to in the hunk of code changes is:
    ```
    {remove_minus_code_diff}
    ```
    There is the code review for this code:
    {comment}
    Please generate the revised code according to the review
    """
    return user_prompt


def get_user_prompts(in_path):
    df = pd.read_json(path_or_buf=in_path, lines=True)
    df["user_prompt"] = df.apply(lambda x: create_prompt(x.review, x.old), axis=1)
    return df


def extract_code_diff(text):
    """
    Extract code diff from text. Code is assumed to be in the format:
    ```lang
    code
    ```
    where lang is the language of the code.
    """

    code = re.findall(r"```[A-Za-z]*\n(.*?)\n```", text, re.DOTALL)
    if code:
        return code[0]
    return "NO CODE"


def evaluate_code_diff(actual_code, refined_code):
    remove_plus_code_diff = remove_plus(actual_code)
    em, em_trim, _, _, bleu, bleu_trim = myeval(remove_plus_code_diff, refined_code)
    return em, em_trim, bleu, bleu_trim


def save_output(cfg, df):
    dataset_name = os.path.splitext(os.path.basename(cfg.in_path))[0]
    output_dir = f"{cfg.out_dir}/{cfg.model}"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = f"{cfg.out_dir}/{cfg.model}/{dataset_name}_output.jsonl"

    df.to_json(output_path, orient="records", lines=True)
    return output_path


################################################# Main #################################################
def main(
    ckpt_dir: str,
    tokenizer_path: str,
    conf_path: str,
    temperature: float = 0.0,
    top_p: float = 0.95,
    max_new_tokens: int = 512,
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

    # set trust_remote_code=False to use local models
    sampling_params = SamplingParams(temperature=0.0, top_p=0.9, max_tokens=512)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=False)
    llm = LLM(
        model=ckpt_dir,
        trust_remote_code=False,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=tp_size,
    )

    def make_prompt(user_prompt):
        instructions = [
            {"role": "system", "content": cfg.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return tokenizer.apply_chat_template(
            instructions, add_generation_prompt=True, tokenize=False
        )

    df = get_user_prompts(cfg.in_path)
    prompts = df.user_prompt.apply(make_prompt)

    if debug:
        print(f"Prompts: {len(df.index)}")

    sampling_params.stop = [tokenizer.eos_token]
    outputs = llm.generate(prompts, sampling_params)

    answers = [output.outputs[0].text for output in outputs]

    df["deepseek_answer"] = answers
    df["deepseek_code"] = df.deepseek_answer.apply(extract_code_diff)

    (
        df["deepseek_em"],
        df["deepseek_em_trim"],
        df["deepseek_bleu"],
        df["deepseek_bleu_trim"],
    ) = zip(
        *df.apply(
            lambda row: evaluate_code_diff(row["new"], row["deepseek_code"]), axis=1
        )
    )

    if debug:
        for output in outputs[:5]:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    output_path = save_output(cfg, df)
    print(f"Output saved to {output_path}")


if __name__ == "__main__":
    fire.Fire(main)
