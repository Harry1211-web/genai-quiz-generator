# %% [markdown]
# # LoRA fine-tuning of a small question-generation model (Google Colab, free T4 GPU)
# Experiment: does small model + LoRA catch up with small model + best prompt, or with the large model?
# 1. Upload data/sft/train.jsonl to Colab.  2. Runtime -> Change runtime type -> T4 GPU.
# Record the library versions you used in the README (the trl API changes quickly).

# %%
# !pip install -q "transformers>=4.44" "peft>=0.12" "trl>=0.10" "datasets>=2.20" "accelerate>=0.33"

# %%
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

BASE = "Qwen/Qwen2.5-1.5B-Instruct"   # small, supports Vietnamese, fits a 16 GB T4
ds = load_dataset("json", data_files="train.jsonl")["train"].train_test_split(test_size=0.1, seed=42)

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16, device_map="auto")

# %% [markdown]
# LoRA freezes the original weights W and only learns two small matrices A (r x k) and B (d x r):
# W' = W + (alpha/r) * B * A. With r = 16 only about 1% of the parameters are trained.

# %%
lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
cfg = SFTConfig(output_dir="lora-quizgen", num_train_epochs=3, per_device_train_batch_size=2,
                gradient_accumulation_steps=8, learning_rate=2e-4, logging_steps=10,
                eval_strategy="epoch", save_strategy="epoch", fp16=True, max_seq_length=2048,
                report_to="none")
trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds["train"], eval_dataset=ds["test"],
                     peft_config=lora, processing_class=tok)  # older trl versions: tokenizer=tok
trainer.print_trainable_parameters() if hasattr(trainer, "print_trainable_parameters") else None
trainer.train()
trainer.save_model("lora-quizgen")   # saves only the adapter (tens of MB)

# %% [markdown]
# ## Try it and add it to the experiment
# Simplest fair comparison: merge the adapter, export to GGUF, load it in Ollama, add a
# "small-lora" entry to eval/configs.json and re-run eval.run_eval on the EVALUATION SET
# (documents that were not used to build train.jsonl).

# %%
msgs = ds["test"][0]["messages"][:2]
inp = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
out = trainer.model.generate(inp, max_new_tokens=512, do_sample=False)
print(tok.decode(out[0][inp.shape[1]:], skip_special_tokens=True))
