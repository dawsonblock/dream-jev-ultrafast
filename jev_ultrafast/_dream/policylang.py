"""_dream.policylang — the verified policy-program AST.

A bounded expression language for policy *structure*. An
``ExplorationPolicy`` may carry, alongside its scalar knobs, a small
program made of the only two computations a policy performs:

- ``rules`` — score adjustments over a fixed candidate/run feature
  vocabulary. Each rule is ``{"when": <bool expr>, "add": <num expr>}``;
  the adjustment is the sum of every rule whose guard holds.
- ``stop_when`` — an optional extra stopping predicate over run
  features. It can only *add* stopping authority (logical OR with the
  base ``no_progress_window`` check), never weaken it.

This is what makes mutation structural rather than parametric: a
candidate can add a rule, remove one, introduce an interaction term, or
change the stopping expression — moves no scalar perturbation of the
knob vector expresses. What it may never do is write arbitrary
computation: the vocabulary below is the entire language, the
interpreter is total and deterministic, and validation fails closed on
anything outside it. The interpreter, the causal evaluator, and the
authority kernel stay fixed; only the validated AST is mutable.

Security properties, by construction:

- total: every node evaluates on every feature assignment — no
  exceptions, no division, no recursion outside the fixed grammar, no
  unbounded growth (each evaluation clamps to ``_RESULT_CAP``);
- bounded: ``MAX_RULES`` rules, ``MAX_NODES`` AST nodes, ``MAX_DEPTH``
  nesting — a program is a data artifact, never a payload;
- pure: features are numbers supplied by the caller; evaluation has no
  I/O, no environment access, no side effects;
- typed: ``cmp``/``and``/``or``/``not``/``if``(cond) are boolean,
  everything else numeric — a type error is a validation error, not a
  runtime coercion.
"""

from __future__ import annotations

import json
import math

__all__ = [
    "BOOL_FEATURES",
    "MAX_DEPTH",
    "MAX_NODES",
    "MAX_RULES",
    "NUMERIC_FEATURES",
    "normalize_expr",
    "normalize_rules",
    "program_adjustment",
    "program_stop",
    "validate_expr",
]

MAX_RULES = 16
MAX_NODES = 128
MAX_DEPTH = 8
_MAX_CONST = 1.0e6
_RESULT_CAP = 1.0e12
_POW_EXPONENT_RANGE = (0.25, 2.0)

# The complete feature vocabulary. Candidate features describe the offered
# action; run features describe the trajectory so far and are uniform
# across candidates at a step. Numeric and boolean features are distinct
# types — a flag is a guard, not a silent operand; the ``if`` node is the
# only bool→number bridge (``{"if": [flag, then, else]}``).
NUMERIC_FEATURES = (
    # candidate features
    "overlap",        # goal-overlap token count of the candidate (>= 0)
    "offered_rank",   # position in the observed catalogue (0-based)
    # run features (uniform across candidates at a step)
    "no_progress",    # consecutive steps without a page change
    "steps_taken",    # actions executed so far this run
)

BOOL_FEATURES = (
    "is_click",       # candidate kind == "click"
    "is_fill",        # candidate kind == "fill"
    "is_select",      # candidate kind == "select"
    "is_upload",      # candidate kind == "upload"
    "is_control",     # candidate kind is none of the above
    "has_node",       # candidate carries node metadata
)

_COMPARE_OPS = ("<", "<=", "==", ">=", ">")
_ARITH_2 = ("add", "sub", "mul", "min", "max")
_ARITH_1 = ("neg", "abs")
_BOOL_2 = ("and", "or")

# Every legal node key — one key per node dict. Anything else is rejected
# at validation; the interpreter has no else-branch for unknown ops.
_EXPR_KEYS = frozenset(
    {"const", "feature", "clamp", "pow", "cmp", "not", "if", "bool"}
    | set(_ARITH_2)
    | set(_ARITH_1)
    | set(_BOOL_2)
)


def _is_num(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_expr(node, *, depth: int = 0, budget: list | None = None) -> str:
    """Validate one AST node; return its static type (``"num"``/``"bool"``).

    Fails closed on every deviation: unknown keys, wrong arity, unknown
    features, non-finite or oversized constants, type mismatches, and
    depth/size overruns. ``budget`` is a one-element node counter shared
    across the whole program so no single subtree may spend the cap.
    """
    if budget is None:
        budget = [0]
    budget[0] += 1
    if budget[0] > MAX_NODES:
        raise ValueError(f"policy program exceeds {MAX_NODES} nodes")
    if depth > MAX_DEPTH:
        raise ValueError(f"policy program exceeds depth {MAX_DEPTH}")
    if not isinstance(node, dict) or len(node) != 1:
        raise ValueError("policy program nodes are single-key objects")
    (op, arg), = node.items()
    if op not in _EXPR_KEYS:
        raise ValueError(f"unknown policy program node {op!r}")

    def _num(child):
        if validate_expr(child, depth=depth + 1, budget=budget) != "num":
            raise ValueError(f"{op} operands must be numeric")

    def _bool(child):
        if validate_expr(child, depth=depth + 1, budget=budget) != "bool":
            raise ValueError(f"{op} operands must be boolean")

    if op == "const":
        if not _is_num(arg) or not math.isfinite(arg):
            raise ValueError("const must be a finite number")
        if abs(float(arg)) > _MAX_CONST:
            raise ValueError(f"const exceeds {_MAX_CONST}")
        return "num"
    if op == "feature":
        if arg in NUMERIC_FEATURES:
            return "num"
        if arg in BOOL_FEATURES:
            return "bool"
        raise ValueError(f"unknown policy feature {arg!r}")
    if op == "bool":
        if not isinstance(arg, bool):
            raise ValueError("bool literal must be true or false")
        return "bool"
    if op in _ARITH_2 or op in _BOOL_2:
        if not isinstance(arg, list) or len(arg) != 2:
            raise ValueError(f"{op} takes exactly two operands")
        check = _num if op in _ARITH_2 else _bool
        check(arg[0])
        check(arg[1])
        return "num" if op in _ARITH_2 else "bool"
    if op in _ARITH_1:
        _num(arg)
        return "num"
    if op == "clamp":
        if (
            not isinstance(arg, list)
            or len(arg) != 3
            or not all(
                _is_num(a) and math.isfinite(a) and abs(float(a)) <= _MAX_CONST
                for a in arg[1:]
            )
        ):
            raise ValueError(
                f"clamp takes [expr, const_lo, const_hi] bounded to {_MAX_CONST}"
            )
        _num(arg[0])
        return "num"
    if op == "pow":
        if not isinstance(arg, list) or len(arg) != 2:
            raise ValueError("pow takes [expr, const] with a const exponent")
        _num(arg[0])
        # The exponent must pass the ordinary validator — an exact single-key
        # const node inside the shared node budget — before the narrower
        # exponent range applies. Anything extra in that object is smuggled
        # payload, not an exponent.
        if (
            validate_expr(arg[1], depth=depth + 1, budget=budget) != "num"
            or "const" not in arg[1]
            or not _POW_EXPONENT_RANGE[0]
            <= float(arg[1]["const"])
            <= _POW_EXPONENT_RANGE[1]
        ):
            raise ValueError(
                "pow takes [expr, const] with the exponent bounded to "
                f"{_POW_EXPONENT_RANGE}"
            )
        return "num"
    if op == "cmp":
        if (
            not isinstance(arg, list)
            or len(arg) != 3
            or arg[0] not in _COMPARE_OPS
        ):
            raise ValueError(f"cmp takes one of {_COMPARE_OPS} and two operands")
        _num(arg[1])
        _num(arg[2])
        return "bool"
    if op == "not":
        _bool(arg)
        return "bool"
    if op == "if":
        if not isinstance(arg, list) or len(arg) != 3:
            raise ValueError("if takes [cond, then, else]")
        _bool(arg[0])
        _num(arg[1])
        _num(arg[2])
        return "num"
    raise ValueError(f"unhandled policy program node {op!r}")


def normalize_expr(node) -> dict:
    """A canonical (sorted-key, primitive-only) copy of a validated expr."""
    validate_expr(node)
    return json.loads(json.dumps(node, sort_keys=True))


def normalize_rules(rules, stop_when=None) -> tuple[tuple, object]:
    """Validate and canonicalize a whole program.

    Returns ``(rules_tuple, stop_when_or_None)`` with every node validated
    and normalized. Shared node budget: one program, one cap — a program
    that sprinkles nodes across many rules cannot smuggle size past it.
    """
    budget = [0]
    out = []
    for rule in rules or ():
        if (
            not isinstance(rule, dict)
            or set(rule) != {"when", "add"}
        ):
            raise ValueError('policy rules are {"when": expr, "add": expr}')
        if validate_expr(rule["when"], budget=budget) != "bool":
            raise ValueError("rule guards must be boolean")
        if validate_expr(rule["add"], budget=budget) != "num":
            raise ValueError("rule adjustments must be numeric")
        out.append({"when": normalize_expr(rule["when"]),
                    "add": normalize_expr(rule["add"])})
    if len(out) > MAX_RULES:
        raise ValueError(f"policy program exceeds {MAX_RULES} rules")
    stop = None
    if stop_when is not None:
        if validate_expr(stop_when, budget=budget) != "bool":
            raise ValueError("stop_when must be a boolean expression")
        stop = normalize_expr(stop_when)
    return tuple(out), stop


def _clamp(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(-_RESULT_CAP, min(_RESULT_CAP, value))


def _eval(node, features) -> float:
    """Total evaluation: every node yields a finite float on any input.

    Truth is ``!= 0.0``; arithmetic clamps each intermediate to
    ``_RESULT_CAP`` so no expression can produce infinities or NaN.
    """
    (op, arg), = node.items()
    if op == "const":
        return _clamp(float(arg))
    if op == "feature":
        return _clamp(float(features.get(arg, 0.0) or 0.0))
    if op == "bool":
        return 1.0 if arg else 0.0
    if op == "add":
        return _clamp(_eval(arg[0], features) + _eval(arg[1], features))
    if op == "sub":
        return _clamp(_eval(arg[0], features) - _eval(arg[1], features))
    if op == "mul":
        return _clamp(_eval(arg[0], features) * _eval(arg[1], features))
    if op == "min":
        return min(_eval(arg[0], features), _eval(arg[1], features))
    if op == "max":
        return max(_eval(arg[0], features), _eval(arg[1], features))
    if op == "neg":
        return _clamp(-_eval(arg, features))
    if op == "abs":
        return abs(_eval(arg, features))
    if op == "clamp":
        lo, hi = float(arg[1]), float(arg[2])
        if lo > hi:
            lo, hi = hi, lo
        return _clamp(max(lo, min(hi, _eval(arg[0], features))))
    if op == "pow":
        # abs(base)**e: total for every finite base and exponent.
        return _clamp(abs(_eval(arg[0], features)) ** float(arg[1]["const"]))
    if op == "cmp":
        a, b = _eval(arg[1], features), _eval(arg[2], features)
        relation = arg[0]
        result = (
            a < b if relation == "<"
            else a <= b if relation == "<="
            else a == b if relation == "=="
            else a >= b if relation == ">="
            else a > b
        )
        return 1.0 if result else 0.0
    if op == "and":
        return 1.0 if _eval(arg[0], features) != 0.0 and _eval(arg[1], features) != 0.0 else 0.0
    if op == "or":
        return 1.0 if _eval(arg[0], features) != 0.0 or _eval(arg[1], features) != 0.0 else 0.0
    if op == "not":
        return 0.0 if _eval(arg, features) != 0.0 else 1.0
    if op == "if":
        return _eval(arg[1] if _eval(arg[0], features) != 0.0 else arg[2], features)
    # Unreachable: every constructor path validates first.
    raise ValueError(f"unhandled policy program node {op!r}")


def program_adjustment(rules, features: dict) -> float:
    """The score adjustment a validated rule set adds to a candidate."""
    total = 0.0
    for rule in rules or ():
        if _eval(rule["when"], features) != 0.0:
            total += _eval(rule["add"], features)
    return _clamp(total)


def program_stop(stop_when, features: dict) -> bool:
    """Whether the optional extra stopping predicate fires."""
    if stop_when is None:
        return False
    return _eval(stop_when, features) != 0.0
