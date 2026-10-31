import os
import json
import random
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from peft import LoraConfig, get_peft_model

MODEL_ID = "google/gemma-3-4b-it"
TRAIN_FILE = "korean_xlsum_TRAIN100_SWAP_AUGMENTED.jsonl"
OUT_DIR = "gemma3_4b_ko_xlsum_FINAL_DPO_train100"

BETA = 0.1
EPOCHS = 5
LR = 1e-5
MAX_LEN = 3072

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def build_input(tokenizer, prompt, answer, device):
    prompt_text = prompt.strip() + "\n"
    answer_text = answer.strip()

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]

    if len(answer_ids) == 0:
        raise ValueError("Empty answer tokenization")

    # Always preserve the answer; truncate the beginning of the prompt if needed.
    max_prompt_len = MAX_LEN - len(answer_ids)
    if max_prompt_len < 16:
        raise ValueError("Answer too long for MAX_LEN")

    prompt_ids = prompt_ids[-max_prompt_len:]
    input_ids_list = prompt_ids + answer_ids

    input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)

    prompt_len = len(prompt_ids)
    return input_ids, attention_mask, prompt_len

def logprob_answer(model, tokenizer, prompt, answer, device):
    input_ids, attention_mask, prompt_len = build_input(tokenizer, prompt, answer, device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :].float()
    labels = input_ids[:, 1:]

    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(
        dim=-1,
        index=labels.unsqueeze(-1)
    ).squeeze(-1)

    mask = torch.zeros_like(labels, dtype=torch.bool)
    mask[:, max(prompt_len - 1, 0):] = True

    val = token_log_probs[mask].sum()

    if not torch.isfinite(val):
        raise RuntimeError("Non-finite logprob detected")

    return val

SEED = 2026

random.seed(SEED)
torch.manual_seed(SEED)

print("seed =", SEED, flush=True)

print("Loading data...", flush=True)
train_rows = load_jsonl(TRAIN_FILE)

# Adapt frozen TRAIN100 identifiers to the unchanged V4 training loop.
for obj in train_rows:
    if "semantic_pair_id" not in obj or "orientation" not in obj:
        raise RuntimeError("TRAIN100 row missing semantic_pair_id/orientation")
    obj["id"] = obj["semantic_pair_id"] + "__" + obj["orientation"]
print("train n =", len(train_rows), flush=True)

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA IS NOT AVAILABLE. "
        "Run this training only through the GPU Slurm job."
    )

if not torch.cuda.is_bf16_supported():
    raise RuntimeError(
        "CUDA is available but bfloat16 is not supported."
    )

device = torch.device("cuda")
dtype = torch.bfloat16


print("device =", device, flush=True)
print("dtype =", dtype, flush=True)

print("Loading processor/tokenizer...", flush=True)
processor = AutoProcessor.from_pretrained(MODEL_ID)
tokenizer = processor.tokenizer

print("Loading base model...", flush=True)
base_model = Gemma3ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
)
base_model.to(device)
base_model.config.use_cache = False

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
)

print("Attaching LoRA...", flush=True)
model = get_peft_model(base_model, peft_config)
model.print_trainable_parameters()
model.train()

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LR
)

print("Starting stable manual DPO training...", flush=True)
global_step = 0

for epoch in range(1, EPOCHS + 1):
    total_loss = 0.0

    epoch_rows = list(train_rows)
    rng = random.Random(SEED + epoch)
    rng.shuffle(epoch_rows)

    print(
        "epoch order =",
        [x["id"] for x in epoch_rows],
        flush=True
    )

    for obj in epoch_rows:
        prompt = obj["prompt"]
        chosen = obj["chosen"]
        rejected = obj["rejected"]

        optimizer.zero_grad(set_to_none=True)

        pi_chosen = logprob_answer(model, tokenizer, prompt, chosen, device)
        pi_rejected = logprob_answer(model, tokenizer, prompt, rejected, device)

        with torch.no_grad():
            with model.disable_adapter():
                ref_chosen = logprob_answer(model, tokenizer, prompt, chosen, device)
                ref_rejected = logprob_answer(model, tokenizer, prompt, rejected, device)

        pi_margin = pi_chosen - pi_rejected
        ref_margin = ref_chosen - ref_rejected

        loss = -F.logsigmoid(BETA * (pi_margin - ref_margin))

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss at epoch={epoch}, id={obj['id']}, "
                f"pi_margin={pi_margin.item()}, ref_margin={ref_margin.item()}"
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=1.0
        )
        optimizer.step()

        global_step += 1
        total_loss += float(loss.detach().cpu())

        print(
            f"epoch={epoch} step={global_step} id={obj['id']} "
            f"loss={float(loss.detach().cpu()):.4f} "
            f"pi_margin={float(pi_margin.detach().cpu()):.2f} "
            f"ref_margin={float(ref_margin.detach().cpu()):.2f}",
            flush=True
        )

    print(f"Epoch {epoch} average loss = {total_loss / len(train_rows):.4f}", flush=True)

print("Saving LoRA adapter...", flush=True)
model.save_pretrained(OUT_DIR)
processor.save_pretrained(OUT_DIR)

print("DONE", flush=True)
print("Saved adapter to:", OUT_DIR, flush=True)
