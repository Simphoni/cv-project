MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
DATASET_NAME="${HOME}/lora/data"

accelerate launch --mixed_precision="bf16" train.py \
  --pretrained_model_name_or_path=${MODEL_NAME} \
  --dataset_name=${DATASET_NAME} --caption_column="text" \
  --resolution=512 --random_flip \
  --train_batch_size=6 \
  --num_train_epochs=25 --checkpointing_steps=1000 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 --variant="fp16" --rank=16 \
  --resume_from_checkpoint latest
