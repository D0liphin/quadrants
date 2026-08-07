"""Helpers for the ``@dataclasses.dataclass`` kernel-arg path, including ``typing.Final[T]`` compile-time template
fields.

PERF NOTE: everything about ``Final`` resolution is computed **once per dataclass type** and cached in
``_final_plan_cache``. Callers on the per-launch hot path (``_extract_arg``, ``args_hasher.dataclass_to_repr``) do a
single ``dict.get`` keyed on the dataclass type and then, in the overwhelmingly common no-Final-field case, take a
branch that is byte-for-byte the pre-existing code path. No ``isinstance`` / ``typing.get_origin`` /
``dataclasses.fields`` call happens per launch. See the module docstring of ``_template_mapper_hotpath.py`` for why
that matters (``isinstance`` is a ~100-200ns MRO walk vs a ~10ns pointer comparison for ``type(x) is Y``).
"""

import dataclasses
import enum
import typing
from typing import Any

# ``T`` values permitted inside ``Final[T]``. A Final field's value is baked into the compiled kernel as a literal and
# folded into both the in-process template spec key and the cross-process fastcache key, so ``T`` must be something
# that (a) is meaningful as a compile-time literal in generated code and (b) hashes and ``repr``s by value, stably
# across processes. ``bool`` precedes ``int`` only for readability - membership is by exact type so ordering is
# irrelevant.
#
# ``enum.Enum`` subclasses are permitted too (resolved separately, since membership here is by exact type): Genesis
# declares ``integrator: int`` / ``ccd_algorithm: int`` etc. but stores ``IntEnum`` members in them, and an IntEnum
# member is both a valid literal and stably repr'able.
_FINAL_SCALAR_TYPES = frozenset({bool, int, float, str})

# Types that are specifically worth a tailored error, because ``Final[T]`` on them is a plausible user mistake with a
# clear better alternative. Resolved lazily to dodge import cycles (``_ndarray`` imports back into ``lang``).
_FINAL_REJECT_HINTS: "dict[str, str]" = {
    "NdarrayType": "arrays are runtime data, not compile-time constants - drop the ``Final`` and annotate the field "
    "with ``qd.types.NDArray[dtype, ndim]`` as usual",
    "MatrixType": "matrices are runtime data, not compile-time constants - drop the ``Final``",
    "StructType": "``qd.dataclass`` structs are runtime data, not compile-time constants - drop the ``Final``",
    "Tensor": "``qd.Tensor`` is runtime data, not a compile-time constant - drop the ``Final``",
    # ``qd.Template`` and ``qd.types.annotations.template`` are both named ``Template``.
    "Template": "``Final[qd.Template]`` is redundant - a ``qd.Template`` field is already compile-time; use "
    "``Final[<python type>]`` (e.g. ``Final[int]``) instead",
}


def is_final_annotation(annotation: Any) -> bool:
    """Return True if ``annotation`` is a ``typing.Final[T]`` special form.

    ``typing.Final[T]`` on a field of a frozen ``@dataclasses.dataclass`` kernel argument marks the field's value as a
    compile-time constant: the value is baked into the compiled kernel (so ``qd.static(config.field)`` is legal), it is
    folded into the template spec key and the fastcache key (so distinct values compile distinct kernels), and it is
    NOT declared as a runtime scalar kernel arg.

    Bare ``typing.Final`` (with no ``[T]``) returns False and is treated as an ordinary field - ``typing.get_origin``
    yields ``None`` for it, and Quadrants needs the wrapped type anyway.

    ``typing_extensions.Final`` is the same object as ``typing.Final`` on every Python version Quadrants supports
    (>=3.10), so it is accepted transparently; ``validate_final_fields`` raises a clear error if a future divergence
    ever makes that untrue.
    """
    return typing.get_origin(annotation) is typing.Final


def _describe_annotation(annotation: Any) -> str:
    return getattr(annotation, "__name__", None) or repr(annotation)


def _reject_hint_for(inner: Any) -> str | None:
    """Return a tailored remediation hint when ``Final[inner]`` names a type we specifically want to reject well.

    Matched on the *type name* rather than by importing the classes themselves, both to avoid import cycles and
    because the check runs once per dataclass type at compile time, where the cost of a string compare is irrelevant.
    """
    for name in (type(inner).__name__, _describe_annotation(inner)):
        hint = _FINAL_REJECT_HINTS.get(name)
        if hint is not None:
            return hint
    return None


def _validate_final_inner_type(dc_type: type, field_name: str, annotation: Any) -> None:
    """Raise a clear error unless ``Final[annotation]`` names a type we can bake as a compile-time literal."""
    inner = typing.get_args(annotation)
    if not inner:
        raise TypeError(
            f"{dc_type.__name__}.{field_name}: bare ``typing.Final`` is not supported as a Quadrants compile-time "
            f"template field. Write ``Final[T]`` with a concrete type, e.g. ``{field_name}: Final[int]``."
        )
    if len(inner) != 1:
        raise TypeError(
            f"{dc_type.__name__}.{field_name}: ``typing.Final`` takes exactly one type argument, got "
            f"``Final[{', '.join(_describe_annotation(a) for a in inner)}]``."
        )
    inner_type = inner[0]

    if inner_type in _FINAL_SCALAR_TYPES:
        return
    # ``issubclass`` here is fine: this runs once per dataclass type at compile time, never per launch.
    if isinstance(inner_type, type) and issubclass(inner_type, enum.Enum):
        return

    hint = _reject_hint_for(inner_type)
    if hint is None:
        if isinstance(inner_type, type) and dataclasses.is_dataclass(inner_type):
            hint = (
                "nested dataclasses are walked structurally - drop the ``Final`` from this field and mark the "
                "leaf fields inside it as ``Final[...]`` instead"
            )
        else:
            allowed = ", ".join(sorted(t.__name__ for t in _FINAL_SCALAR_TYPES))
            hint = f"``Final[T]`` supports T in {{{allowed}}} or an ``enum.Enum`` subclass"
    raise TypeError(
        f"{dc_type.__name__}.{field_name}: ``Final[{_describe_annotation(inner_type)}]`` cannot be baked as a "
        f"Quadrants compile-time constant - {hint}."
    )


def _build_final_plan(dc_type: type) -> "frozenset[str]":
    """Validate every ``Final`` field on ``dc_type`` and return the set of Final-annotated field names.

    Called once per dataclass type (memoised in ``_final_plan_cache``), so all of the reflection here - including
    ``dataclasses.fields``, ``typing.get_origin`` and ``issubclass`` - stays entirely off the per-launch hot path.
    """
    final_names = []
    for field in dataclasses.fields(dc_type):
        annotation = field.type
        if isinstance(annotation, str):
            # ``from __future__ import annotations`` (or an explicit string annotation) leaves ``field.type`` as an
            # unresolved string. The pre-existing dataclass kernel-arg path already assumes resolved types, so rather
            # than half-supporting it, flag the one case where silently ignoring it would be a correctness trap: a
            # field the user believes is a compile-time constant but which we would lower as a runtime arg.
            if "Final" in annotation:
                raise TypeError(
                    f"{dc_type.__name__}.{field.name}: annotation is the unresolved string {annotation!r}. Quadrants "
                    f"cannot see ``Final`` through a string annotation, so this field would silently become a runtime "
                    f"kernel argument. Remove ``from __future__ import annotations`` from the module defining "
                    f"{dc_type.__name__}, or annotate with the real type object."
                )
            continue
        if not is_final_annotation(annotation):
            # Catch a ``Final``-like special form that is not ``typing.Final`` - e.g. if a future ``typing_extensions``
            # release stops aliasing the stdlib object. Silently treating such a field as a runtime one would be a
            # correctness trap of exactly the kind described just above.
            origin = typing.get_origin(annotation)
            if origin is not None and "Final" in _describe_annotation(origin):
                raise TypeError(
                    f"{dc_type.__name__}.{field.name}: annotation {annotation!r} looks like a ``Final`` special form "
                    f"but is not ``typing.Final`` (got origin {origin!r}). Use ``typing.Final`` from the standard "
                    f"library."
                )
            continue
        _validate_final_inner_type(dc_type, field.name, annotation)
        final_names.append(field.name)

    if final_names and dc_type.__hash__ is None:
        # ``@dataclasses.dataclass`` (non-frozen, eq=True) sets ``__hash__ = None``. ``Final`` asserts the value never
        # changes, and Quadrants bakes it into compiled code accordingly - so a mutable carrier is a contradiction we
        # reject rather than silently tolerate. ``unsafe_hash=True`` is accepted for the same reason the rest of the
        # dataclass path accepts it: the user has explicitly asserted value-stability.
        raise TypeError(
            f"{dc_type.__name__} has ``Final`` field(s) {sorted(final_names)} but is not frozen. A ``Final`` field's "
            f"value is baked into the compiled kernel, so it must not be reassignable. Declare the class as "
            f"``@dataclasses.dataclass(frozen=True)`` (or ``unsafe_hash=True`` if you must keep it mutable and accept "
            f"responsibility for never reassigning these fields)."
        )
    return frozenset(final_names)


# Memo of ``dataclass type -> frozenset of Final field names``. Keyed on the type object, so it is bounded by the
# number of distinct dataclass types the process ever passes to a kernel. An empty frozenset (the common case) is a
# meaningful cached result, so callers must distinguish it from a cache miss via ``.get(...) is None``.
_final_plan_cache: "dict[type, frozenset[str]]" = {}


def final_field_names(dc_type: Any) -> "frozenset[str]":
    """Return the cached set of ``Final``-annotated field names on ``dc_type``, validating on first sighting.

    Hot-path contract: one ``dict.get``. Callers should short-circuit on the empty result so that dataclasses with no
    ``Final`` fields (the overwhelmingly common case) run the pre-existing code path untouched.

    ``dc_type`` is typed ``Any`` rather than ``type`` because ``_extract_arg`` calls this with its loosely-typed
    ``annotation`` parameter (a union covering every kernel-arg annotation shape), having already established that it
    is a dataclass type via the ``__dataclass_fields__`` probe. Narrowing at that call site would need a
    ``typing.cast``, which is a real function call on a per-launch path.
    """
    names = _final_plan_cache.get(dc_type)
    if names is None:
        names = _build_final_plan(dc_type)
        _final_plan_cache[dc_type] = names
    return names


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
