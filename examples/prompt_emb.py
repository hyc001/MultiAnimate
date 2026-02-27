from diffsynth import ModelManager, WanPrompter
import torch
import os
device = "cuda:0"

model_manager = ModelManager(torch_dtype=torch.bfloat16, device=device)
prompter = WanPrompter(tokenizer_path=None)
text_encoder = None

text_encoder_path = './Wan2.1-I2V-14B-720P/models_t5_umt5-xxl-enc-bf16.pth'
model_path = [text_encoder_path]
model_manager.load_models(model_path)

text_encoder_model_and_path = model_manager.fetch_model("wan_video_text_encoder", require_model_path=True)

if text_encoder_model_and_path is not None:
    print("load")
    text_encoder, tokenizer_path = text_encoder_model_and_path
    prompter.fetch_models(text_encoder)
    prompter.fetch_tokenizer(os.path.join(os.path.dirname(tokenizer_path), "google/umt5-xxl"))

pos_prompt = "two person are dancing"
# neg_prompt = ""
pos_prompt_emb = prompter.encode_prompt(pos_prompt, positive=True)
# neg_prompt_emb = prompter.encode_prompt(neg_prompt, positive=False)

pos_emb = {"context": pos_prompt_emb}
# neg_emb = {"context": neg_prompt_emb}

torch.save(pos_emb, os.path.join("/cache/yingcheng/UniAnimate-DiT", f"pos2_emb.pt"))
# torch.save(neg_emb, os.path.join("/temp/liusonghua/yingcheng/UniAnimate-DiT", f"neg_emb.pt"))
