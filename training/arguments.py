import os, argparse

def init_args(input_args=None):
    parser = argparse.ArgumentParser(description="Lora training script for medical image generation.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--variant", type=str, default="fp16")
    parser.add_argument("--dataset_name", type=str, default="/home/ubuntu/lora/data")
    parser.add_argument("--image_column", type=str, default="file_name")
    parser.add_argument("--caption_column", type=str, default="text")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="sd-model-finetuned-lora")
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=2333)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--random_flip", action="store_true")
    parser.add_argument("--train_text_encoder", action="store_true")
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--checkpointing_steps", type=int, default=1000)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--scale_lr", action="store_true", default=False)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--prediction_type", type=str, default=None)
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--debug_loss", action="store_true")
    parser.add_argument("--snr_gamma", type=float, default=None)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    
    return args
