import os
import json
from diffusers import DiffusionPipeline
import torch
from tqdm import tqdm

# 文件路径
NUM_LIMIT=256
metadata_path = "/home/ubuntu/lora/data/test/metadata.jsonl"

lora_path_dir = "/home/ubuntu/inference/output"
loras = os.listdir(lora_path_dir)
loras = ["lora-rank128-snr-bs6-encoder-sample500"]
print("loras: ", len(loras), loras)

for lora in loras:
    output_dir = f"/home/ubuntu/inference/output_test/{lora}"
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join("/home/ubuntu/lora/", lora)

    # 加载扩散模型
    print("Loading model...")
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16")
    if "lora" in lora:
        print("loading lora from", model_path)
        # pipe.unet.load_attn_procs(model_path)
        pipe.load_lora_weights(model_path)
    pipe.to("cuda")
    print("Model loaded.")

    # 批量生成图像
    def generate_images(batch_prompts, batch_file_names):
        try:
            generator = torch.Generator("cuda").manual_seed(42)
            images = pipe(batch_prompts, num_inference_steps=30, guidance_scale=7.5, generator=generator).images
            for image, file_name in zip(images, batch_file_names):
                output_path = os.path.join(output_dir, file_name)
                image.save(output_path)
        except Exception as e:
            print(f"Error generating images for batch: {e}")

    batch_prompts = []
    batch_file_names = []
    batch_size = 2

    with open(metadata_path, "r") as file:
        for line in tqdm(file.readlines()[:NUM_LIMIT], desc="Processing prompts"):
            try:
                data = json.loads(line.strip())
                prompt = data.get("text")
                file_name = data.get("file_name")
                
                if prompt and file_name:
                    batch_prompts.append(prompt)
                    batch_file_names.append(file_name)

                    if len(batch_prompts) == batch_size:
                        generate_images(batch_prompts, batch_file_names)
                        batch_prompts = []
                        batch_file_names = []
                else:
                    print(f"Invalid data: {data}")
            except json.JSONDecodeError as e:
                print(f"Error parsing line: {line.strip()} - {e}")

    # 处理剩余的未完成批次
    if batch_prompts:
        generate_images(batch_prompts, batch_file_names)

    print("Batch inference completed.")
