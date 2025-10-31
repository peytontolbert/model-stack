from specs.config import ModelConfig
from model.lm import TransformerLM
from data.loader import build_dataloader
from train.trainer import Trainer
from serve.generate import generate
from eval.metrics import perplexity

cfg = ModelConfig(d_model=512, n_heads=8, n_layers=8, d_ff=2048, vocab_size=32000, attn_impl="flash")

model = TransformerLM(cfg).cuda()
dl = build_dataloader("/corpus/shards", batch_size=8, seq_len=1024)
trainer = Trainer(model, cfg, dl)

for step, batch in enumerate(dl):
    loss = trainer.step(batch)
    if step % 100 == 0:
        print("ppl:", perplexity(loss))

# Inference
out = generate(model, batch.input_ids[:1].cuda(), max_new_tokens=64)