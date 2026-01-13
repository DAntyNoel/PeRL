
import torch
import os

from typing import List, Optional
from datasets import load_dataset
from transformers import set_seed, AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from fire import Fire

from perl.utils.logging import init_logger, logger
from perl.data import load_dataset
from perl.config.config import TrainConfig
import inspect

def fuzzy_jobs(
    args: TrainConfig
):
    init_logger()
    args.training.output_dir = args.training.output_dir or "output"
    args.training.run_name = args.training.run_name or args.training.output_dir # training run name is the output_dir
    if not os.path.exists(args.training.output_dir): # check if output_dir exists
        os.makedirs(args.training.output_dir, exist_ok=True)
    else:
        logger.info(f"Output directory {args.training.output_dir} already exists, using it")
    set_seed(args.common.seed)

    if args.common.debug:
        args.training.report_to = []

    # only initialize for rank 0 when process group is available
    is_main_process = True
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        is_main_process = torch.distributed.get_rank() == 0

    if is_main_process:
        if "trackio" in args.training.report_to:
            import trackio
            trackio.init(
                project=args.logging.trackio_project,
                space_id=args.logging.trackio_space_id,
                config=vars(args.training)
            )
            logger.info(f"Trackio initialized successfully")
        elif "wandb" in args.training.report_to:
            import wandb
            wandb.init(
                name=args.training.run_name,
                config=vars(args.training),
            )
            logger.info(f"Wandb initialized successfully")

    return args

def train(
    config: TrainConfig = None
):
    # 0. parse args and prepare logger
    print(config)
    args = fuzzy_jobs(config)

    # 1. load tokenizer and dataset
    logger.info(f"Loading tokenizer from {args.model.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model.model_name_or_path)
    tokenizer.padding_side = "left"  # Configure for decoder-only architecture: use left padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else "<|endoftext|>"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    
    logger.info(f"Loading dataset from {args.dataset.dataset_name_or_path}")
    dataset = load_dataset(
        args.dataset.dataset_name_or_path,
        example_numbers=args.dataset.example_numbers,
        tokenizer=tokenizer
    )
    train_dataset = dataset["train_dataset"]
    test_dataset = dataset["test_dataset"]
    reward_functions = dataset["reward_functions"]

    if "reward_weights" in dataset:
        reward_weights = dataset["reward_weights"]
    else:
        reward_weights = [1.0] * len(reward_functions)
    args.training.reward_weights = reward_weights

    # 2. load and configure model
    logger.info(f"Loading model from {args.model.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model.model_name_or_path,
        torch_dtype= torch.bfloat16 if args.model.dtype == "bfloat16" else torch.float16,
        attn_implementation="flash_attention_2"
    )
    logger.info(f"Model loaded successfully")

    # 3. configure lora
    if args.peft.use_peft:
        from perl.lora.adapter import apply_peft, PEFT_TYPE_TO_FUNCTION_MAPPING
        peft_type = getattr(args.peft, "type", "lora")
        logger.info(
            "[PEFT] Detected configuration -> type=%s, r=%s, alpha=%s, dropout=%s, targets=%s",
            peft_type,
            getattr(args.peft, "r", None),
            getattr(args.peft, "lora_alpha", None),
            getattr(args.peft, "lora_dropout", None),
            getattr(args.peft, "target_modules", None),
        )

        # Log the loader function and its source snippet for traceability
        loader_fn = PEFT_TYPE_TO_FUNCTION_MAPPING.get(peft_type)
        if loader_fn is not None:
            try:
                fn_file = inspect.getsourcefile(loader_fn)
                fn_src = inspect.getsource(loader_fn)
                # Keep snippet reasonably short
                snippet_lines = fn_src.splitlines()
                preview = "\n".join(snippet_lines[:80])
                logger.info(
                    "[PEFT] Loader function: %s (%s)\n-----8<----- SOURCE BEGIN -----8<-----\n%s\n-----8<----- SOURCE END -----8<-----",
                    f"{loader_fn.__module__}.{loader_fn.__name__}",
                    fn_file,
                    preview,
                )
            except Exception as e:
                logger.warning("[PEFT] Failed to inspect loader function: %s", e)
        else:
            logger.warning("[PEFT] Unknown peft.type '%s' (no loader function found)", peft_type)

        logger.info("[PEFT] Applying adapters to model…")
        optimizer, model = apply_peft(model, args)

        # Summarize trainable parameters, especially LoRA params
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        trainable_count = sum(p.numel() for _, p in trainable_params)
        # Heuristic: LoRA parameters usually include "lora_" in their names
        lora_params = [(n, p) for n, p in trainable_params if ("lora_" in n.lower() or "adapter" in n.lower())]
        lora_count = sum(p.numel() for _, p in lora_params)

        def fmt(n: int) -> str:
            try:
                return f"{n:,}"
            except Exception:
                return str(n)

        logger.info(
            "[PEFT] Params -> total=%s, trainable=%s (%.4f%%), lora=%s (%.4f%% of total; %.4f%% of trainable)",
            fmt(total_params),
            fmt(trainable_count),
            100.0 * (trainable_count / max(total_params, 1)),
            fmt(lora_count),
            100.0 * (lora_count / max(total_params, 1)),
            100.0 * (lora_count / max(trainable_count, 1)),
        )

        # Also list a few representative LoRA parameter names for clarity
        preview_names = ", ".join([n for n, _ in lora_params[:10]])
        logger.info("[PEFT] LoRA trainable tensors (sample): %s", preview_names if preview_names else "<none>")
        logger.info("Lora configured successfully")

    # 4.Training configuration
    training_args = GRPOConfig(
        **vars(args.training),
    )

    # 5.Train
    logger.info(f"Training model with GRPO")
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_functions,
        args=training_args,
        train_dataset=train_dataset,
        optimizers=(optimizer, None) if optimizer is not None else (None, None)
    )
    
    # 支持从 checkpoint 恢复训练
    resume_checkpoint = args.training.resume_from_checkpoint
    if resume_checkpoint == "true":
        resume_checkpoint = True
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    logger.info(f"Training completed successfully")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")
