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


LANGUAGES = {
    "py": "Python",
    "c": "C",
    "go": "Go",
    "js": "Javascript",
    "java": "Java",
    ".cs": "C#",
    "php": "php",
    "cpp": "C++",
    "rb": "Ruby",
}

num_sample = 3


def create_example_prompt(row):
    ret = ""

    for i in range(1, num_sample + 1):
        code, comment = row["sample_input_{}".format(i)].split("<SEP_DATA>")
        improved_code = row["sample_output_{}".format(i)]
        sample_prompt = f"""
        ## Example\n\nSubmitted code:
        ```{code}```
        \n\nDeveloper comment: ```{comment}```
        \n\nImproved code: ```{improved_code}```\n\n---\n\n
        """
        ret = ret + sample_prompt

    return ret


def create_prompt(row):
    # comment, code_diff = row["review"], row["old"]
    code_diff, comment = row["input"].split("<SEP_DATA>")
    # language_code = row["language"]

    remove_minus_code_diff = remove_minus(code_diff)
    # language = LANGUAGES.get(language_code, "Python")

    prompt_header = f"""
    You are given 3 examples. Each example begins with "##Example" and ends with "---".
    Each example contains the submitted code, the developer comment, and the improved code.
    Your task is to improve your submitted code based on the comment that another developer gave you.\n\n
    """
    sample_prompt = create_example_prompt(row)

    main_prompt = f"""
    Submitted code: ```{remove_minus_code_diff}```
    Developer comment: ```{comment}```
    \n\nImproved code: 
    """
    return prompt_header + sample_prompt + main_prompt


def make_instructions(system_prompt, user_prompt):
    instructions = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return instructions


def get_user_prompts(in_path):
    df = pd.read_csv(in_path)
    df["user_prompt"] = df.apply(lambda x: create_prompt(x), axis=1)
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
        instructions = make_instructions(cfg.system_prompt, user_prompt)
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
            lambda row: evaluate_code_diff(row["output"], row["deepseek_code"]), axis=1
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
