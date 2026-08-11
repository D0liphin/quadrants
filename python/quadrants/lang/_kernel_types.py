from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, TypeAlias

from quadrants.types.enums import AutodiffMode


class KernelBatchedArgType(IntEnum):
    FLOAT = 0
    INT = 1
    UINT = 2
    QD_ARRAY = 3
    QD_ARRAY_WITH_GRAD = 4


@dataclass
class SrcLlCacheObservations:
    cache_key_generated: bool = False
    cache_validated: bool = False
    cache_loaded: bool = False
    cache_stored: bool = False


@dataclass
class FeLlCacheObservations:
    cache_hit: bool = False


@dataclass
class PerOffloadCacheObservations:
    """Per-offloaded-task and per-construct compilation cache stats for one kernel compile.

    ``constructs_*`` count the per-*task* codegen cache (LLVM module reuse per offloaded task; always active on the
    LLVM backends). ``frontend_constructs_*`` count the per-*construct* FRONTEND cache (reuse of the
    simplify/mgp/offload output per top-level construct) and are ``-1`` when the per-construct frontend split did not
    run for this compile. On a warm compile where only one construct changed, the expected result is
    ``recompiled == 1`` and ``cache_hit == total - 1`` at whichever layer ran. Counts (not wall time) so the behavior
    can be asserted deterministically in tests.
    """

    constructs_total: int = 0
    constructs_cache_hit: int = 0
    constructs_recompiled: int = 0
    frontend_constructs_total: int = -1
    frontend_constructs_cache_hit: int = -1
    frontend_constructs_recompiled: int = -1


@dataclass
class LaunchObservations:
    found_kernel_in_materialize_cache: bool = False


@dataclass
class LaunchStats:
    kernel_args_count_by_type: dict[KernelBatchedArgType, int]


CompiledKernelKeyType = tuple[Callable, int, AutodiffMode]
ArgsHash: TypeAlias = tuple[int, ...]
