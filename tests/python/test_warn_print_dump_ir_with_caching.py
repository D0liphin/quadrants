import warnings

import pytest

import quadrants as qd
from quadrants.lang import impl, misc

WARNING_CODE = r"\[warning_code=DUMP_IR_CACHE_MISMATCH\]"


def test_warn_caching_with_print_ir():
    with pytest.warns(
        UserWarning,
        match=WARNING_CODE,
    ):
        qd.init(print_ir=True, log_level="warn", offline_cache=True)


def test_warn_caching_with_qd_dump(monkeypatch):
    monkeypatch.setenv("QD_DUMP_IR", "1")

    with pytest.warns(
        UserWarning,
        match=WARNING_CODE,
    ):
        qd.init(log_level="warn", offline_cache=True)


def test_no_warn_caching_ir():
    warnings.filterwarnings("error")
    try:
        qd.init(log_level="warn", offline_cache=True)
    except UserWarning as user_warning:
        assert WARNING_CODE not in user_warning.args[0]

    warnings.resetwarnings()


def test_both_caches_disabled_over_an_explicit_request():
    # Caching is turned off even though it was asked for explicitly: the request for IR output is the more specific
    # intent, and honouring both would drop the output for precisely the kernels that are already cached.
    with pytest.warns(UserWarning, match=WARNING_CODE):
        qd.init(print_ir=True, log_level="warn", offline_cache=True, src_ll_cache=True)

    assert not impl.default_cfg().offline_cache
    assert not impl.get_runtime().src_ll_cache


@pytest.mark.parametrize("env_var", ["QD_DUMP_IR", "QD_DUMP_CFG", "QD_DUMP_SIMPLIFY"])
def test_caching_disabled_by_each_dump_env_var(monkeypatch, env_var):
    monkeypatch.setenv(env_var, "1")

    with pytest.warns(UserWarning, match=WARNING_CODE):
        qd.init(log_level="warn", offline_cache=True, src_ll_cache=True)

    assert not impl.default_cfg().offline_cache
    assert not impl.get_runtime().src_ll_cache


def test_caching_disabled_by_a_print_option_other_than_print_ir():
    with pytest.warns(UserWarning, match=WARNING_CODE):
        qd.init(print_kernel_asm=True, log_level="warn", offline_cache=True, src_ll_cache=True)

    assert not impl.default_cfg().offline_cache
    assert not impl.get_runtime().src_ll_cache


def test_caching_kept_when_dump_env_var_is_zero(monkeypatch):
    # The C++ side reads these through get_environ_config, which parses the value as an int, so 0 leaves the dump off.
    monkeypatch.setenv("QD_DUMP_IR", "0")

    qd.init(log_level="warn", offline_cache=True, src_ll_cache=True)

    assert impl.default_cfg().offline_cache
    assert impl.get_runtime().src_ll_cache


def test_caching_kept_without_any_request_for_codegen_output():
    qd.init(log_level="warn", offline_cache=True, src_ll_cache=True)

    assert impl.default_cfg().offline_cache
    assert impl.get_runtime().src_ll_cache


def test_every_watched_config_key_exists():
    # Renaming one of these should fail here, rather than as an AttributeError out of qd.init.
    cfg = impl.default_cfg()
    for key in misc._CODEGEN_OUTPUT_CONFIG_KEYS:
        assert isinstance(getattr(cfg, key), bool)
