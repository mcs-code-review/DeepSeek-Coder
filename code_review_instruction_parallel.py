from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

tp_size = 4  # Tensor Parallelism
sampling_params = SamplingParams(temperature=0.0, top_p=0.9, max_tokens=512)
model_name = "deepseek-ai/deepseek-coder-6.7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = LLM(
    model=model_name,
    trust_remote_code=True,
    gpu_memory_utilization=0.9,
    tensor_parallel_size=tp_size,
)

messages_list = [
    [{"role": "user", "content": "Who are you?"}],
    [{"role": "user", "content": "What can you do?"}],
    [{"role": "user", "content": "Explain Transformer briefly."}],
]
prompts = [
    tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    for messages in messages_list
]

sampling_params.stop = [tokenizer.eos_token]
outputs = llm.generate(prompts, sampling_params)

generated_text = [output.outputs[0].text for output in outputs]
print(generated_text)
