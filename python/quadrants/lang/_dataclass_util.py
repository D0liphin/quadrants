import typing
from typing import Any


def is_final_annotation(annotation: Any) -> bool:
    """Return True if ``annotation`` is a ``typing.Final[T]`` special form.

    The POC uses ``typing.Final[T]`` on frozen-dataclass fields as the signal that the field's value should be baked
    into the compiled kernel as a compile-time constant (like a ``qd.template()``-annotated arg): its value is folded
    into the template spec key (so distinct values compile distinct kernels), it does not appear as a runtime kernel
    scalar arg, and ``qd.static(config.field)`` inside the kernel body resolves at compile time.

    Bare ``typing.Final`` (with no ``T``) is not accepted - Quadrants needs the wrapped type to know how to treat the
    baked value.
    """
    return typing.get_origin(annotation) is typing.Final


def unwrap_final(annotation: Any) -> Any:
    """Strip a ``typing.Final[T]`` wrapper and return ``T``. Returns ``annotation`` unchanged when it isn't ``Final``.

    Callers that already know the annotation is ``Final`` should still use this helper (rather than
    ``typing.get_args(annotation)[0]`` directly) so the failure mode for a malformed ``Final`` is uniform.
    """
    if not is_final_annotation(annotation):
        return annotation
    args = typing.get_args(annotation)
    if not args:
        raise TypeError(f"typing.Final without a wrapped type is not supported: {annotation!r}")
    return args[0]


def create_flat_name(basename: str, child_name: str) -> str:
    """
    Appends child_name to basename, separated by __qd_.
    If basename does not start with __qd_ then prefix the resulting string
    with __qd_.

    Note that we want to avoid adding prefix __qd_ if already included in `basename`,
    to avoid duplicating said delimiter.

    We'll use this when expanding py dataclass members, e.g.

    @dataclasses.dataclass
    def Foo:
        a: int
        b: int

    foo = Foo(a=5, b=3)

    When we expand out foo, we'll replace foo with the following names instead:
    - __qd_foo__qd_a
    - __qd_foo__qd_b

    We use the __qd_ to ensure that it's easy to ensure no collision with existing user-defined
    names. We require the user to not create any fields or variables which themselves are prefixed
    with __qd_, and given this constraint, the names we create will not conflict with user-generated
    names.
    """
    if basename.startswith("__qd_"):
        return f"{basename}__qd_{child_name}"
    return f"__qd_{basename}__qd_{child_name}"
