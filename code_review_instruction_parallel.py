import json
import os
import re

import pandas as pd
import fire
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
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
    tp_size: int = 1,
    debug: bool = False,
):
    cfg = Config(conf_path)

    if torch.cuda.is_available():
        print("CUDA is available")
    else:
        print("CUDA is not available")
        return

    # set trust_remote_code=False to use local models
    # tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=False)
    # model = AutoModelForCausalLM.from_pretrained(
    #     tokenizer_path, trust_remote_code=False,
    #     torch_dtype=torch.float16,
    # ).cuda()

    sampling_params = SamplingParams(
        temperature=temperature, top_p=top_p, max_tokens=max_new_tokens
    )
    model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = LLM(
        model=model_name,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=tp_size,
    )

    df = get_user_prompts(cfg.in_path)

    if debug:
        df = df.head(5)

    messages_list = [
        [{"role": "user", "content": "Who are you?"}],
        [{"role": "user", "content": "What can you do?"}],
        [{"role": "user", "content": "Explain Transformer briefly."}],
    ]
    prompts = [
        tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        for messages in messages_list
    ]

    sampling_params.stop = [tokenizer.eos_token]
    outputs = llm.generate(prompts, sampling_params)

    generated_text = [output.outputs[0].text for output in outputs]
    print(generated_text)

    # outputs = []
    # for user_prompt in tqdm(df.user_prompt, total=len(df.index), desc="Prompting"):
    #     instructions = [
    #         {"role": "system", "content": cfg.system_prompt},
    #         {"role": "user", "content": user_prompt},
    #     ]

    #     # inputs = tokenizer.apply_chat_template(
    #     #     instructions, add_generation_prompt=True, return_tensors="pt"
    #     # ).to(model.device)
    #     # outputs_raw = model.generate(
    #     #     inputs,
    #     #     max_new_tokens=max_new_tokens,
    #     #     do_sample=False,
    #     #     top_k=50,
    #     #     top_p=top_p,
    #     #     temperature=temperature,
    #     #     num_return_sequences=1,
    #     #     eos_token_id=tokenizer.eos_token_id,
    #     #     pad_token_id=tokenizer.eos_token_id,
    #     # )
    #     # results = tokenizer.decode(
    #     #     outputs_raw[0][len(inputs[0]) :], skip_special_tokens=True
    #     # )

    #     # outputs.append(results)

    #     if debug:
    #         print("Instruction:\n", instructions)
    #         print("Response:\n", results)
    #         print("\n==================================\n")
    # print("Prompting done")

    # df["deepseek_answer"] = outputs
    # df["deepseek_code"] = df["deepseek_answer"].apply(extract_code_diff)

    # dataset_name = os.path.splitext(os.path.basename(cfg.in_path))[0]
    # output_dir = f"{cfg.out_dir}/{cfg.model}"

    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    # output_path = f"{cfg.out_dir}/{cfg.model}/{dataset_name}_output.jsonl"

    # df.to_json(output_path, orient="records", lines=True)
    print(f"Output saved to {output_path}")


if __name__ == "__main__":
    fire.Fire(main)
