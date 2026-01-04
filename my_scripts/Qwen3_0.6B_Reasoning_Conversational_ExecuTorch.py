from unsloth import FastLanguageModel  # noqa: I001
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from transformers import TextStreamer
from trl import SFTConfig, SFTTrainer
from unsloth.chat_templates import (
    get_chat_template,
    standardize_sharegpt,
)

MODEL_ID = "unsloth/Qwen3-0.6B"
QAT_SCHEME = "int8-int4"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_ID,
    max_seq_length=2048,
    dtype=torch.bfloat16,
    load_in_4bit=False,
    full_finetuning=True,
    qat_scheme=QAT_SCHEME,
)
tokenizer = get_chat_template(tokenizer, chat_template="qwen3")
reasoning_dataset = load_dataset("unsloth/OpenMathReasoning-mini", split="cot")
non_reasoning_dataset = load_dataset("mlabonne/FineTome-100k", split="train")
reasoning_dataset
non_reasoning_dataset


def generate_conversation(examples):
    problems = examples["problem"]
    solutions = examples["generated_solution"]
    conversations = []
    for problem, solution in zip(problems, solutions):
        conversations.append(
            [
                {"role": "user", "content": problem},
                {"role": "assistant", "content": solution},
            ]
        )
    return {
        "conversations": conversations,
    }


reasoning_conversations = tokenizer.apply_chat_template(
    list(reasoning_dataset.map(generate_conversation, batched=True)["conversations"]),
    tokenize=False,
)
reasoning_conversations[0]
dataset = standardize_sharegpt(non_reasoning_dataset)
non_reasoning_conversations = tokenizer.apply_chat_template(
    list(dataset["conversations"]),
    tokenize=False,
)
non_reasoning_conversations[0]
print(len(reasoning_conversations))
print(len(non_reasoning_conversations))
chat_percentage = 0.25
non_reasoning_subset = pd.Series(non_reasoning_conversations)
non_reasoning_subset = non_reasoning_subset.sample(
    int(len(reasoning_conversations) * (chat_percentage / (1 - chat_percentage))),
    random_state=2407,
)
print(len(reasoning_conversations))
print(len(non_reasoning_subset))
print(len(non_reasoning_subset) / (len(non_reasoning_subset) + len(reasoning_conversations)))

data = pd.concat([pd.Series(reasoning_conversations), pd.Series(non_reasoning_subset)])
data.name = "text"
combined_dataset = Dataset.from_pandas(pd.DataFrame(data))
combined_dataset = combined_dataset.shuffle(seed=3407)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=20,
        learning_rate=5e-5,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",
        packing=False,  # Explicitly set packing to False since we're using chat templates
    ),
    train_dataset=combined_dataset,
    eval_dataset=None,
)

trainer_stats = trainer.train()

messages = [{"role": "user", "content": "Solve (x + 2)^2 = 0."}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
_ = model.generate(
    **tokenizer(text, return_tensors="pt").to("cuda"),
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    streamer=TextStreamer(tokenizer, skip_prompt=True),
)
messages = [{"role": "user", "content": "Solve (x + 2)^2 = 0."}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,
)
_ = model.generate(
    **tokenizer(text, return_tensors="pt").to("cuda"),
    max_new_tokens=1024,
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    streamer=TextStreamer(tokenizer, skip_prompt=True),
)
model_name = MODEL_ID.split("/")[-1]
save_to = f"{model_name}-{QAT_SCHEME}-unsloth"
model.save_pretrained_torchao(save_to, tokenizer=tokenizer)
