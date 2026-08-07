"""QD_LOAD_IR / QUADRANTS_LOAD_PTX read replacement IR / PTX from ``debug_dump_path``, which codegen only does for a
kernel it actually compiles. A cached kernel skips codegen, so the edited files would be ignored without any diagnostic.
``qd.init`` rejects the combination instead."""

import os
from contextlib import contextmanager

import pytest

import quadrants as qd


@contextmanager
def env_vars(**overrides):
    """Set the given env vars for the duration of the block, restoring them afterwards. A value of None unsets."""
    previous = {name: os.environ.get(name) for name in overrides}

    def apply(values):
        for name, value in values.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    apply(overrides)
    try:
        yield
    finally:
        apply(previous)


def test_error_load_ir_with_offline_cache():
    with env_vars(QD_LOAD_IR="1", QUADRANTS_LOAD_PTX=None):
        with pytest.raises(ValueError, match="QD_LOAD_IR"):
            qd.init(log_level="warn", offline_cache=True, src_ll_cache=False)


def test_error_load_ir_with_fastcache():
    with env_vars(QD_LOAD_IR="1", QUADRANTS_LOAD_PTX=None):
        with pytest.raises(ValueError, match="QD_LOAD_IR"):
            qd.init(log_level="warn", offline_cache=False, src_ll_cache=True)


def test_error_load_ptx_with_offline_cache():
    # jit_cuda only checks that QUADRANTS_LOAD_PTX is present, so even "0" enables the PTX load path.
    with env_vars(QD_LOAD_IR=None, QUADRANTS_LOAD_PTX="0"):
        with pytest.raises(ValueError, match="QUADRANTS_LOAD_PTX"):
            qd.init(log_level="warn", offline_cache=True, src_ll_cache=False)


def test_no_error_load_ir_with_caching_disabled():
    with env_vars(QD_LOAD_IR="1", QUADRANTS_LOAD_PTX=None):
        qd.init(log_level="warn", offline_cache=False, src_ll_cache=False)


def test_no_error_when_load_ir_is_zero():
    # QD_LOAD_IR goes through get_environ_config on the C++ side, which parses it as an int, so "0" leaves it off.
    with env_vars(QD_LOAD_IR="0", QUADRANTS_LOAD_PTX=None):
        qd.init(log_level="warn", offline_cache=True, src_ll_cache=True)


def test_no_error_without_load_env_vars():
    with env_vars(QD_LOAD_IR=None, QUADRANTS_LOAD_PTX=None):
        qd.init(log_level="warn", offline_cache=True, src_ll_cache=True)
