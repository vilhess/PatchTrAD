from dataclasses import dataclass

from omegaconf import DictConfig, OmegaConf


@dataclass
class DatasetConfig:
    name: str
    bs: int
    lr: float
    ws: int
    in_dim: int
    epochs: int
    stride: int
    patch_len: int
    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> "DatasetConfig":
        return cls(**OmegaConf.to_container(cfg, resolve=True))
