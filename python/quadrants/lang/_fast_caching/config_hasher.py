from quadrants.lang import impl

from .hash_utils import hash_iterable_strings

EXCLUDE_PREFIXES = ["_", "offline_cache", "print_", "verbose_"]

# Exact key names that do not feed codegen. Matched by full name rather than by prefix, so that config fields added
# later under the same prefix (say a new cuda_* compiler option) are not silently dropped from the cache key.
EXCLUDE_KEYS = {
    # Destination directory for IR dumps only. Whether anything is dumped is gated on the QD_DUMP_IR env var, which is
    # not part of the cache key either, so the path cannot influence the generated kernel.
    "debug_dump_path",
    # Applied once at runtime via context_set_limit(CU_LIMIT_STACK_SIZE) in llvm_runtime_executor, never read by
    # codegen.
    "cuda_stack_limit",
}


def hash_compile_config() -> str:
    """
    Calculates a hash string for the current compiler config.

    If any compilation-relevant value in the compiler config changes, the hash string changes too. Keys that only
    affect debug output or runtime device limits are excluded, via EXCLUDE_PREFIXES and EXCLUDE_KEYS, so that they do
    not cause spurious cache misses. The C++ offline cache key does not include them either, see
    get_offline_cache_key_of_compile_config in quadrants/analysis/offline_cache_util.cpp.
    """
    config = impl.get_runtime().prog.config()
    config_l = []
    for k in dir(config):
        if k in EXCLUDE_KEYS:
            continue
        skip = False
        for prefix in EXCLUDE_PREFIXES:
            if k.startswith(prefix) or k in [""]:
                skip = True
        if skip:
            continue
        v = getattr(config, k)
        config_l.append(f"{k}={v}")
    config_hash = hash_iterable_strings(config_l, separator="\n")
    return config_hash
