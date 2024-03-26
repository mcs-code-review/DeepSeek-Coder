import json
import os
import re

import pandas as pd
import fire
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


class Config:
    def __init__(self, conf_path):
        """
        conf_path: a json file storing configs
        """
        with open(conf_path, "r") as json_file:
            conf = json.load(json_file)

        for key, value in conf.items():
            setattr(self, key, value)


def create_prompt(comment, code_diff):
    user_prompt = f"""
    As a developer, imagine you've submitted a pull request and your team leader
    requests you to make a change to a piece of code. The old code being
    referred to in the hunk of code changes is:
    ```
    {code_diff}
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
    result = re.search(r"```(.*)```", text, re.DOTALL)

    if result:
        return result.group(1)
    return "NO CODE"


################################################# Main #################################################
def main(
    ckpt_dir: str,
    tokenizer_path: str,
    conf_path: str,
    temperature: float = 0.0,
    top_p: float = 0.95,
    max_new_tokens: int = 512,
    debug: bool = False,
):

    # set trust_remote_code=False to use local models
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        tokenizer_path, trust_remote_code=False, torch_dtype=torch.bfloat16
    ).cuda()

    cfg = Config(conf_path)

    df = get_user_prompts(cfg.in_path)

    if debug:
        df = df.head(5)

    outputs = []
    for user_prompt in tqdm(df.user_prompt, total=len(df.index), desc="Prompting"):
        instructions = [
            {"role": "system", "content": cfg.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        inputs = tokenizer.apply_chat_template(
            instructions, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        outputs_raw = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            top_k=50,
            top_p=top_p,
            temperature=temperature,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
        results = tokenizer.decode(
            outputs_raw[0][len(inputs[0]) :], skip_special_tokens=True
        )

        outputs.append(results)

        if debug:
            print("Instruction:\n", instructions)
            print("Response:\n", results)
            print("\n==================================\n")
    print("Prompting done")

    df["deepseek_answer"] = outputs
    df["deepseek_code"] = df["deepseek_answer"].apply(extract_code_diff)

    dataset_name = os.path.splitext(os.path.basename(cfg.in_path))[0]
    output_dir = f"{cfg.out_dir}/{cfg.model}"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = f"{cfg.out_dir}/{cfg.model}/{dataset_name}_prompt.jsonl"

    df.to_json(output_path, orient="records", lines=True)
    print(f"Output saved to {output_path}")


if __name__ == "__main__":
    fire.Fire(main)
