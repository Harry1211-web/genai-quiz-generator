# %% [markdown]
# # LoRA fine-tune model nhỏ sinh câu hỏi (chạy trên Google Colab, GPU T4 miễn phí)
# Mục tiêu thí nghiệm: model nhỏ + LoRA có đuổi kịp model nhỏ + prompt tốt nhất / model lớn không?
# 1. Upload data/sft/train.jsonl lên Colab.  2. Runtime → Change runtime type → T4 GPU.
# Ghi lại phiên bản thư viện bạn dùng vào README (API của trl thay đổi khá nhanh).

# %%
# !pip install -q "transformers>=4.44" "peft>=0.12" "trl>=0.10" "datasets>=2.20" "accelerate>=0.33"

# %%
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

BASE = "Qwen/Qwen2.5-1.5B-Instruct"   # nhỏ, hỗ trợ tiếng Việt, vừa T4 16 GB
ds = load_dataset("json", data_files="train.jsonl")["train"].train_test_split(test_size=0.1, seed=42)

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16, device_map="auto")

# %% [markdown]
# LoRA: đóng băng toàn bộ trọng số gốc W, chỉ học hai ma trận nhỏ A (r×k) và B (d×r):
# W' = W + (alpha/r)·B·A. Với r = 16, số tham số học được chỉ khoảng 1% model.

# %%
lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
cfg = SFTConfig(output_dir="lora-quizgen", num_train_epochs=3, per_device_train_batch_size=2,
                gradient_accumulation_steps=8, learning_rate=2e-4, logging_steps=10,
                eval_strategy="epoch", save_strategy="epoch", fp16=True, max_seq_length=2048,
                report_to="none")
trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds["train"], eval_dataset=ds["test"],
                     peft_config=lora, processing_class=tok)  # trl cũ: tokenizer=tok
trainer.print_trainable_parameters() if hasattr(trainer, "print_trainable_parameters") else None
trainer.train()
trainer.save_model("lora-quizgen")   # chỉ lưu adapter (~vài chục MB)

# %% [markdown]
# ## Thử sinh và đưa vào thí nghiệm
# Cách đơn giản nhất để so sánh công bằng: merge adapter, xuất GGUF, nạp vào Ollama, rồi thêm
# một dòng "small-lora" vào eval/configs.json và chạy lại eval.run_eval trên TẬP ĐÁNH GIÁ
# (tài liệu không dùng để tạo train.jsonl).

# %%
msgs = ds["test"][0]["messages"][:2]
inp = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
out = trainer.model.generate(inp, max_new_tokens=512, do_sample=False)
print(tok.decode(out[0][inp.shape[1]:], skip_special_tokens=True))
