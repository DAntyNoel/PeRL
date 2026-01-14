import os
import torch
import time
import logging
import torch.distributed as dist
from transformers import AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model
# 尝试导入 LoraLinear，如果版本不同可能路径不同，做个兼容
try:
    from peft.tuners.lora import Linear as LoraLinear
    from peft.tuners.lora import LoraLayer
except ImportError:
    from peft.tuners.lora.layer import Linear as LoraLinear
    from peft.tuners.lora.layer import LoraLayer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def enable_dual_stream_mode(model, adapter_names=["milora_max", "milora_min"]):
    """
    遍历模型中的所有 LoRA 层，强制激活双流模式 (Dual Stream Mode)。
    
    功能：
    1. 设置自定义标记 module.dual_stream_mode = True (兼容你提到的修改版 PEFT)
    2. 强制设置 module.active_adapters 为列表 (解决标准 PEFT 的 active_adapters 激活问题)
    """
    count = 0
    skip = 0
    
    logger.info(f"Traversing model to activate dual stream for adapters: {adapter_names}...")
    
    for name, module in model.named_modules():
        # 检查是否是 LoRA 层 (通常是 peft.tuners.lora.Linear)
        # 也可以通过检查是否有 lora_A 属性来判定，这样更通用
        if (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
            # 1. 设置自定义 Flag (参考你的代码)
            module.dual_stream_mode = True
            
            # 2. [关键修复] 在 Layer 层级强制写入 active_adapters
            # PeftModel (外层) 不让写，但 LoraLinear (内层) 是允许写的
            # 这行代码让标准 PEFT 在 forward 时会循环计算这两个 adapter
            try:
                module.set_adapter(adapter_names)
            except Exception as e:
                logger.warning(f"[Dual-Stream] set_adapter failed at {name}: {e}")
            
            # 兼容某些旧版本 PEFT，可能使用的是 active_adapter (单数)
            if hasattr(module, "active_adapter") and isinstance(module.active_adapter, str):
                 # 如果它原本存的是字符串，我们把它改成列表
                 module.active_adapter = adapter_names

            count += 1
        else:
            skip += 1
            
    # 只在主进程打印，避免多卡训练刷屏
    if not dist.is_initialized() or dist.get_rank() == 0:
        logger.info(f"[Dual-Stream] Activated dual_stream_mode on {count} LoRA layers. (Target: {adapter_names})")

def verify_dual_stream(model, adapter_names=["milora_max", "milora_min"], verbose: bool = None):
    """
    验证所有 LoRA 层的双流激活状态，并打印统计：
    - 层数总计、具备双 adapter 的层数
    - active_adapters / active_adapter 的类型与内容
    - 返回统计字典，便于上层决定是否告警

    通过环境变量 MILORA_DUAL_DEBUG=1 控制逐层详细日志；也可通过 verbose 参数显式指定。
    """
    if verbose is None:
        verbose = os.environ.get("MILORA_DUAL_DEBUG", "0") == "1"

    total_lora = 0
    with_both_adapters = 0
    active_list_count = 0
    active_str_count = 0
    active_other_count = 0
    problems = []

    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            total_lora += 1
            have_both = all(a in getattr(module, "lora_A", {}) for a in adapter_names)
            if have_both:
                with_both_adapters += 1

            active_repr = None
            if hasattr(module, "active_adapters"):
                act = module.active_adapters
                if isinstance(act, (list, tuple)):
                    active_list_count += 1
                    active_repr = list(act)
                    # 检查是否包含需要的两个适配器
                    if not all(a in act for a in adapter_names):
                        problems.append((name, f"active_adapters missing {adapter_names}, got {act}"))
                else:
                    active_other_count += 1
                    active_repr = act
                    problems.append((name, f"active_adapters type {type(act)} not list/tuple"))
            elif hasattr(module, "active_adapter"):
                act = module.active_adapter
                if isinstance(act, (list, tuple)):
                    active_list_count += 1
                    active_repr = list(act)
                    if not all(a in act for a in adapter_names):
                        problems.append((name, f"active_adapter (list-like) missing {adapter_names}, got {act}"))
                elif isinstance(act, str):
                    active_str_count += 1
                    active_repr = act
                    # 字符串表示时通常只激活单个 adapter
                    problems.append((name, f"active_adapter is str='{act}', likely single-adapter forward"))
                else:
                    active_other_count += 1
                    active_repr = act
                    problems.append((name, f"active_adapter type {type(act)} not supported"))
            else:
                problems.append((name, "No active_adapters/active_adapter attribute found"))

            if verbose and (not dist.is_initialized() or dist.get_rank() == 0):
                logger.info(f"[Dual-Verify] {name}: both={have_both}, active={active_repr}")

    if not dist.is_initialized() or dist.get_rank() == 0:
        logger.info(
            f"[Dual-Verify] LoRA layers: total={total_lora}, with_both={with_both_adapters}, "
            f"active_list={active_list_count}, active_str={active_str_count}, active_other={active_other_count}"
        )
        if problems:
            logger.warning(f"[Dual-Verify] Potential issues found: {len(problems)}")
            if verbose:
                for n, msg in problems:
                    logger.warning(f" - {n}: {msg}")

    return {
        "total_lora": total_lora,
        "with_both_adapters": with_both_adapters,
        "active_list_count": active_list_count,
        "active_str_count": active_str_count,
        "active_other_count": active_other_count,
        "problems": problems,
    }

def add_dual_svd_initialized_lora(model, 
                                rank=64, 
                                hyper_param_type="LLM-Adapters"):
    """
    MiLoRA++ Dual Mode 入口函数
    """
    
    # 1. 配置基础参数
    if hyper_param_type == "LLM-Adapters":
        lora_alpha = rank  
        lora_dropout = 0.05
        target_modules = ["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]
    else:
        lora_alpha = rank
        lora_dropout = 0.1
        target_modules = ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']

    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=rank,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )

    logger.info(f"Initializing Dual MiLoRA++ Adapters (Rank: {rank})...")

    # 2. 注入 Adapter 结构
    # 创建第一个 adapter "milora_max"
    model = get_peft_model(model, peft_config, adapter_name="milora_max")
    
    # 添加第二个 adapter "milora_min"
    model.add_adapter("milora_min", peft_config)
    
    # 3. [改进] 使用遍历法激活双模式
    # 这替代了之前会报错的 model.set_adapter(...) 或 model.active_adapters = ...
    enable_dual_stream_mode(model, adapter_names=["milora_max", "milora_min"])
    # 额外：启用后做一次验收，便于日志确认是否真正是双流
    verify_dual_stream(model, adapter_names=["milora_max", "milora_min"], verbose=None)
    print(model)

    # 4. 执行 SVD 并初始化
    start_time = time.time()
    
    with torch.no_grad():
        for name, module in model.named_modules():
            # 检查是否是包含了我们 adapter 的层
            # 兼容性检查：确保它有 base_layer 和 lora_A 字典
            if hasattr(module, 'base_layer') and hasattr(module, 'lora_A'):
                if not any(t in name for t in target_modules):
                    continue

                # 确保两个 adapter 都在这个模块里
                if "milora_max" not in module.lora_A or "milora_min" not in module.lora_A:
                    continue

                try:
                    # 获取原始权重
                    base_weight = module.base_layer.weight.data
                    
                    # --- SVD 分解 ---
                    U, S, Vh = torch.linalg.svd(base_weight.float(), full_matrices=False)
                    
                    # --- Max 模式 (主成分) ---
                    # 取前 r 个方向
                    V_max = Vh[:rank, :]
                    A_max_init = V_max.contiguous()
                    B_max_init = torch.zeros((base_weight.shape[0], rank), device=base_weight.device).contiguous()
                    
                    # --- Min 模式 (非主成分) ---
                    # 取后 r 个方向
                    V_min = Vh[-rank:, :]
                    A_min_init = V_min.contiguous()
                    B_min_init = torch.zeros((base_weight.shape[0], rank), device=base_weight.device).contiguous()

                    # --- 赋值 ---
                    module.lora_A['milora_max'].weight.data = A_max_init.to(base_weight.device).to(base_weight.dtype)
                    module.lora_B['milora_max'].weight.data = B_max_init.to(base_weight.device).to(base_weight.dtype)

                    module.lora_A['milora_min'].weight.data = A_min_init.to(base_weight.device).to(base_weight.dtype)
                    module.lora_B['milora_min'].weight.data = B_min_init.to(base_weight.device).to(base_weight.dtype)

                except Exception as e:
                    logger.warning(f"Failed to initialize dual adapters for {name}: {e}")

    end_time = time.time()
    logger.info(f"Dual SVD initialization completed in {end_time - start_time:.2f} seconds.")
    
    return model