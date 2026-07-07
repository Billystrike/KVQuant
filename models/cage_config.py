from dataclasses import dataclass, field, fields
from typing import Any


_DEFAULT_K_GROUP_SIZES = [32, 64, 128]
_DEFAULT_K_CLIP_PERCENTILES = [0.999, 0.995, 0.99]
_DEFAULT_V_GROUP_SIZES = [32, 64, 128]
_DEFAULT_V_CLIP_PERCENTILES = [0.999, 0.995, 0.99]
_SUPPORTED_MODES = {"fake"}


@dataclass
class CageConfig:
    cage_enable: bool = False
    cage_mode: str = "fake"
    cage_k_enable: bool = True
    cage_v_enable: bool = True
    cage_k_importance: str = "q2_var"
    cage_k_group_sizes: list[int] = field(default_factory=lambda: list(_DEFAULT_K_GROUP_SIZES))
    cage_k_clip_percentiles: list[float] = field(default_factory=lambda: list(_DEFAULT_K_CLIP_PERCENTILES))
    cage_k_num_buckets: int = 3
    cage_k_flush_length: int = 128
    cage_v_importance: str = "wo_var"
    cage_v_group_sizes: list[int] = field(default_factory=lambda: list(_DEFAULT_V_GROUP_SIZES))
    cage_v_clip_percentiles: list[float] = field(default_factory=lambda: list(_DEFAULT_V_CLIP_PERCENTILES))
    cage_v_num_buckets: int = 3
    cage_collect_metrics: bool = False
    cage_dump_dir: str | None = None
    cage_memory_summary: bool = False

    def __post_init__(self):
        if self.cage_k_group_sizes is None:
            self.cage_k_group_sizes = list(_DEFAULT_K_GROUP_SIZES)
        if self.cage_k_clip_percentiles is None:
            self.cage_k_clip_percentiles = list(_DEFAULT_K_CLIP_PERCENTILES)
        if self.cage_v_group_sizes is None:
            self.cage_v_group_sizes = list(_DEFAULT_V_GROUP_SIZES)
        if self.cage_v_clip_percentiles is None:
            self.cage_v_clip_percentiles = list(_DEFAULT_V_CLIP_PERCENTILES)


def get_cage_config(config: Any) -> CageConfig:
    cage_config = CageConfig(
        cage_enable=getattr(config, "cage_enable", False),
        cage_mode=getattr(config, "cage_mode", "fake"),
        cage_k_enable=getattr(config, "cage_k_enable", True),
        cage_v_enable=getattr(config, "cage_v_enable", True),
        cage_k_importance=getattr(config, "cage_k_importance", "q2_var"),
        cage_k_group_sizes=_list_or_default(
            getattr(config, "cage_k_group_sizes", None),
            _DEFAULT_K_GROUP_SIZES,
        ),
        cage_k_clip_percentiles=_list_or_default(
            getattr(config, "cage_k_clip_percentiles", None),
            _DEFAULT_K_CLIP_PERCENTILES,
        ),
        cage_k_num_buckets=getattr(config, "cage_k_num_buckets", 3),
        cage_k_flush_length=getattr(config, "cage_k_flush_length", 128),
        cage_v_importance=getattr(config, "cage_v_importance", "wo_var"),
        cage_v_group_sizes=_list_or_default(
            getattr(config, "cage_v_group_sizes", None),
            _DEFAULT_V_GROUP_SIZES,
        ),
        cage_v_clip_percentiles=_list_or_default(
            getattr(config, "cage_v_clip_percentiles", None),
            _DEFAULT_V_CLIP_PERCENTILES,
        ),
        cage_v_num_buckets=getattr(config, "cage_v_num_buckets", 3),
        cage_collect_metrics=getattr(config, "cage_collect_metrics", False),
        cage_dump_dir=getattr(config, "cage_dump_dir", None),
        cage_memory_summary=getattr(config, "cage_memory_summary", False),
    )

    if cage_config.cage_enable:
        _validate_enabled_config(cage_config)

    _write_back(config, cage_config)
    return cage_config


def _list_or_default(value: Any, default: list[Any]) -> Any:
    if value is None:
        return list(default)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return list(value)
    return value


def _validate_enabled_config(config: CageConfig) -> None:
    if config.cage_mode not in _SUPPORTED_MODES:
        supported = ", ".join(sorted(_SUPPORTED_MODES))
        raise ValueError(f"Unsupported CAGE mode {config.cage_mode!r}; supported modes: {supported}")

    _validate_bucket_policy(
        "cage_k",
        config.cage_k_num_buckets,
        config.cage_k_group_sizes,
        config.cage_k_clip_percentiles,
    )
    _validate_bucket_policy(
        "cage_v",
        config.cage_v_num_buckets,
        config.cage_v_group_sizes,
        config.cage_v_clip_percentiles,
    )


def _validate_bucket_policy(
    prefix: str,
    num_buckets: int,
    group_sizes: Any,
    clip_percentiles: Any,
) -> None:
    if not isinstance(num_buckets, int) or num_buckets <= 0:
        raise ValueError(f"{prefix}_num_buckets must be a positive integer, got {num_buckets!r}")

    _validate_list_length(f"{prefix}_group_sizes", group_sizes, num_buckets)
    _validate_list_length(f"{prefix}_clip_percentiles", clip_percentiles, num_buckets)


def _validate_list_length(name: str, value: Any, expected_length: int) -> None:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list with {expected_length} entries, got {type(value).__name__}")
    if len(value) != expected_length:
        raise ValueError(f"{name} must have {expected_length} entries, got {len(value)}")


def _write_back(config: Any, cage_config: CageConfig) -> None:
    for field in fields(cage_config):
        setattr(config, field.name, getattr(cage_config, field.name))
