# copyright (c) 2025, mikastars39.org
# All rights reserved.
# This source code is licensed under the Apache-2.0 License.
# See the LICENSE file in the root directory for details.

from .slicefine import register_slicefine_method
register_slicefine_method() # register slicefine method to peft

import logging

logger = logging.getLogger('[Adapter]')

def apply_lora(model, args):
    logger.info(f"Applying standard LoRA with rank={args.peft.r}")
    from peft import LoraConfig, get_peft_model
    config = LoraConfig(
        peft_type="LORA",
        task_type=args.peft.task_type,
        r=args.peft.r,
        lora_alpha=args.peft.lora_alpha,
        target_modules=args.peft.target_modules,
        lora_dropout=args.peft.lora_dropout,
    )
    return None, get_peft_model(model, config)

def apply_dora(model, args):
    logger.info(f"Applying DoRA with rank={args.peft.r}")
    from peft import LoraConfig, get_peft_model
    config = LoraConfig(
        peft_type="LORA",
        use_dora=True,
        task_type=args.peft.task_type,
        r=args.peft.r,
        lora_alpha=args.peft.lora_alpha,
        target_modules=args.peft.target_modules,
        lora_dropout=args.peft.lora_dropout,
    )
    return None, get_peft_model(model, config)

def apply_vera(model, args):
    logger.info(f"Applying Vera with rank={args.peft.r}")
    from peft import VeraConfig, get_peft_model
    config = VeraConfig(r=args.peft.r)
    return None, get_peft_model(model, config)

def apply_miss(model, args):
    logger.info(f"Applying MiSS with rank={args.peft.r}")
    from peft import MissConfig, get_peft_model
    config = MissConfig(r=args.peft.r)
    return None, get_peft_model(model, config)

def apply_pissa(model, args):
    logger.info(f"Applying PiSSA with rank={args.peft.r}")
    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(
        # init_lora_weights="pissa", # Configure the initialization method to "pissa", which may take several minutes to execute SVD on the pre-trained model.
        init_lora_weights="pissa_niter_4", # Initialize the PiSSA with fast SVD, which completes in just a few seconds.
        r=args.peft.r,
        lora_alpha=args.peft.lora_alpha,
        lora_dropout=args.peft.lora_dropout, # Since the component of the PiSSA adapter are the principal singular values and vectors, dropout should be set to 0 to avoid random discarding.
        target_modules=args.peft.target_modules,
        task_type=args.peft.task_type,
    )
    return None, get_peft_model(model, lora_config)

def apply_milora(model, args):
    logger.info(f"Applying MiLoRA with rank={args.peft.r}")
    from .milora_dual import add_dual_svd_initialized_lora
    from .milora import add_svd_initialized_lora
    return None, add_svd_initialized_lora(
        model=model,
        rank=args.peft.r,
    )

def apply_layernorm(model, args):
    logger.info(f"Applying LayerNorm Tuning")
    from peft import get_peft_model, TaskType, LNTuningConfig
    peft_config = LNTuningConfig(
        task_type=TaskType.CAUSAL_LM,
    )
    return None, get_peft_model(model, peft_config)

def apply_adalora(model, args):
    logger.info(f"Applying AdaLoRA with rank={args.peft.r}")
    from peft import AdaLoraConfig, get_peft_model
    config = AdaLoraConfig(
        peft_type="ADALORA",
        task_type=args.peft.task_type,
        init_r=args.peft.r,
        lora_alpha=args.peft.lora_alpha,
        target_modules=args.peft.target_modules,
        lora_dropout=args.peft.lora_dropout,
        total_step=args.training.max_steps,
    )
    return None, get_peft_model(model, config)

def apply_IA3(model, args):
    logger.info(f"Applying IA3 Tuning with rank={args.peft.r}")
    from peft import IA3Config, get_peft_model, TaskType
    config = IA3Config(task_type=TaskType.CAUSAL_LM)
    return None, get_peft_model(model, config)

def apply_milora_plus(model, args):
    logger.info(f"Applying MiLoRA++ with rank={args.peft.r}")
    from .milora_plus import add_svd_initialized_lora
    return None, add_svd_initialized_lora(
        model=model,
        rank=args.peft.r,
    )

def apply_lorafa(model, args):
    logger.info(f"Applying LoRA-FA with rank={args.peft.r}")
    from peft import LoraConfig, get_peft_model
    from peft.optimizers import create_lorafa_optimizer

    config = LoraConfig(
        peft_type="LORA",
        task_type=args.peft.task_type,
        r=args.peft.r,
        lora_alpha=args.peft.lora_alpha,
        target_modules=args.peft.target_modules,
        lora_dropout=args.peft.lora_dropout,
    ) 

    model = get_peft_model(model, config)
        
    optimizer = create_lorafa_optimizer(
        model=model,
        r=args.peft.r,
        lora_alpha=args.peft.lora_alpha,
        lr=args.training.learning_rate,
    )
    return optimizer, model

def apply_lora_plus(model, args):
    logger.info(f"Applying LoRA+ with rank={args.peft.r}")
    from peft import LoraConfig, get_peft_model
    from torch.optim import AdamW
    from peft.optimizers import create_loraplus_optimizer
    
    # First create the LoRA model
    config = LoraConfig(
        peft_type="LORA",
        task_type=args.peft.task_type,
        r=args.peft.r,
        lora_alpha=args.peft.lora_alpha,
        target_modules=args.peft.target_modules,
        lora_dropout=args.peft.lora_dropout,
    )
    model = get_peft_model(model, config)
    
    # Then create the LoraPlus optimizer with different learning rates for A and B
    optimizer = create_loraplus_optimizer(
        model=model,
        optimizer_cls=AdamW,
        lr=args.training.learning_rate,
        loraplus_lr_ratio=2.0,
    )
    return optimizer, model

def apply_slicefine(model, args):
    logger.info(f"Applying SliceFine with rank={args.peft.r}")
    from .slicefine import SliceFineConfig
    from peft import get_peft_model
    config = SliceFineConfig(
        r=args.peft.r,
        slice_mode=getattr(args.peft, "slice_mode", "column"),
        slice_position=getattr(args.peft, "slice_position", 0),
        target_modules=args.peft.target_modules,
        bias="all" if getattr(args.peft, "bias", False) else "none"
    )
    print(f"[SliceFine] Applying SliceFine with rank={config.r}, modules={config.target_modules}")
    
    peft_model = get_peft_model(model, config)
    
    peft_model.print_trainable_parameters()
    
    trainable_params = [p for p in peft_model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError(
            "[SliceFine Error] No trainable parameters found! \n"
            "1. Check if 'target_modules' match the model architecture.\n"
            "2. Check if 'part_T' is correctly set to requires_grad=True."
        )
    return None, peft_model

def apply_hra(model, args):
    logger.info(f"Applying HRA with rank={args.peft.r}")
    from peft import HRAConfig, get_peft_model
    config = HRAConfig(
        r=args.peft.r,
        target_modules=args.peft.target_modules,
        init_weights="True",
    )
    return None, get_peft_model(model, config)

def apply_rslora(model, args):
    logger.info(f"Applying RS-LoRA with rank={args.peft.r}")
    from peft import LoraConfig, get_peft_model
    config = LoraConfig(
        peft_type="LORA",
        use_rslora=True,
        task_type=args.peft.task_type,
        r=args.peft.r,
        lora_alpha=args.peft.lora_alpha,
        target_modules=args.peft.target_modules,
        lora_dropout=args.peft.lora_dropout,
    )
    return None, get_peft_model(model, config)


def apply_milora_dual(model, args):
    logger.info(f"Applying MiLoRA Dual Mode with rank={args.peft.r}")
    # 导入我们新建的文件
    from .milora_dual import add_dual_svd_initialized_lora
    from torch.optim import AdamW

    # 1. 应用双模初始化
    # 注意：这里会创建 "milora_max" 和 "milora_min" 两个 adapter
    model = add_dual_svd_initialized_lora(
        model=model,
        rank=args.peft.r
    )
    
    # 2. 从 Config 获取特定学习率 (假设你在 Config 中添加了这些字段)
    # 如果 Config 没设，提供默认 fallback
    default_lr = args.training.learning_rate
    lr_max = getattr(args.peft, "lr_max", default_lr)      # Max 模式通常 LR 小一点
    lr_min = getattr(args.peft, "lr_min", default_lr * 2)  # Min 模式通常 LR 大一点
    
    logger.info(f"[Dual Optimizer] Setting LR: Max={lr_max}, Min={lr_min}, Base={default_lr}")

    # 3. 参数分组与过滤
    # 我们需要把参数分为三组：Max组, Min组, 其他组
    params_max = []
    params_min = []
    params_other = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if "milora_max" in name:
            params_max.append(param)
        elif "milora_min" in name:
            params_min.append(param)
        else:
            params_other.append(param)
    
    # 4. 构建优化器参数组
    optimizer_grouped_parameters = [
        {
            "params": params_max,
            "lr": lr_max,
            "name": "milora_max_group"
        },
        {
            "params": params_min,
            "lr": lr_min,
            "name": "milora_min_group"
        },
        {
            "params": params_other,
            "lr": default_lr,
            "name": "other_params_group"
        }
    ]
    
    # 5. 实例化优化器
    # 可以在这里根据 args.training 设置 weight_decay 等
    optimizer = AdamW(optimizer_grouped_parameters, weight_decay=0.01)
    
    return optimizer, model

# ------------------------------ mapping function to peft type ------------------------------

PEFT_TYPE_TO_FUNCTION_MAPPING = {
    "lora": apply_lora,
    "dora": apply_dora,
    "slicefine": apply_slicefine,
    "milora": apply_milora,
    "milora_plus": apply_milora_plus,
    "lorafa": apply_lorafa,
    "lora_plus": apply_lora_plus,
    "adalora": apply_adalora,
    "IA3": apply_IA3,
    "layernorm": apply_layernorm,
    "vera": apply_vera,
    "miss": apply_miss,
    "pissa": apply_pissa,
    "hra": apply_hra,
    "milora_dual": apply_milora_dual, 
}

# ------------------------------ dispatch function to peft type ------------------------------

def apply_peft(model, args):
    """Dispatch function that routes to the appropriate PEFT method based on args.peft.type"""
    peft_type = args.peft.type
    if peft_type in PEFT_TYPE_TO_FUNCTION_MAPPING:
        return PEFT_TYPE_TO_FUNCTION_MAPPING[peft_type](model, args)
    else:
        raise ValueError(f"Unsupported PEFT type: {peft_type}")