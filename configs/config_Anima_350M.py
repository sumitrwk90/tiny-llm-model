from dataclasses import dataclass


@dataclass
class ModelConfig:
    n_layers: int = 24
    emb_dim: int = 1024
    n_heads: int = 16
    ff_dim: int = 4096
    rotary_dim: int = 32
    rope_theta: float = 10000.0
    seq_len: int = 2048
    norm_type: str = "rmsnorm"
    activation_function: str = "swiglu"
    attention_type: str = "multi-head"
    use_flash_attention: bool = True
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    vocab_size: int = 32000
    tie_word_embeddings: bool = True
    initializer_range: float = 0.02


@dataclass
class TrainingConfig:
    tokens: int = 1000000000
    seq_len: int = 2048
    micro_batch_size: int = 8
    global_batch_size: int = 1024
    gradient_accumulation: int = 128
    learning_rate: float = 3e-4
    lr_warmup_steps: int = 2000
    weight_decay: float = 0.1
    optimizer: str = "adamw"
    precision: str = "bf16"
    checkpointing: str = "full"
    eval_interval: int = 2000
    save_interval: int = 20000