"""Decidable predicate evaluation for Op Spec ``constraints``.

Each constraint is a string predicate over bound shape/dtype symbols, e.g.
``"K % 8 == 0"``, ``"dtype(x) == 'bf16'"``, ``"M >= 16 and N % 32 == 0"``.
Applicability must be decidable from metadata alone — never by running code
(design principle §1.3.2: "reject before you compile").

We evaluate a deliberately tiny subset of Python syntax (constants, names,
arithmetic, comparisons, boolean ops, and the ``dtype(...)`` builtin) against a
binding dict. Anything else is rejected as "undecidable" so that an over-clever
constraint can never silently pass.
"""
from __future__ import annotations

import ast
from typing import Any

# Node types we permit in a constraint expression. Anything outside this set
# means the constraint is not statically decidable and must be rejected at ingest.
_ALLOWED_NODE_TYPES: tuple[type, ...] = (
    ast.Expression,
    ast.BoolOp,        # and / or
    ast.UnaryOp,       # not  (only `not`)
    ast.BinOp,         # + - * / % //
    ast.Compare,       # == != < <= > >=
    ast.Constant,      # numbers, strings, bools, None
    ast.Name,          # bound symbols (K, M, ...) + the `dtype` builtin
    ast.Call,          # only the dtype(...) builtin
    ast.And,
    ast.Or,
    ast.Not,
    ast.Add, ast.Sub, ast.Mult, ast.Mod, ast.FloorDiv, ast.Div,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.USub,          # unary minus (for negative numbers)
)

_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Mod: lambda a, b: a % b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Div: lambda a, b: a / b,
}
_CMPOPS = {
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
}


class UndecidableConstraintError(ValueError):
    """Raised when a constraint uses syntax outside the decidable subset."""


class _Evaluator(ast.NodeVisitor):
    def __init__(self, bindings: dict[str, Any]) -> None:
        self._b = bindings

    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        vals = [self.visit(v) for v in node.values]
        if isinstance(node.op, ast.And):
            result = True
            for v in vals:
                result = result and v
            return result
        result = False
        for v in vals:
            result = result or v
        return result

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.Not):
            return not operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise UndecidableConstraintError(f"unsupported unary op {type(node.op).__name__}")

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        op = _BINOPS.get(type(node.op))
        if op is None:
            raise UndecidableConstraintError(f"unsupported binop {type(node.op).__name__}")
        return op(self.visit(node.left), self.visit(node.right))

    def visit_Compare(self, node: ast.Compare) -> Any:
        left = self.visit(node.left)
        for op_node, right_node in zip(node.ops, node.comparators, strict=True):
            op = _CMPOPS.get(type(op_node))
            if op is None:
                raise UndecidableConstraintError(f"unsupported compare {type(op_node).__name__}")
            right = self.visit(right_node)
            if not op(left, right):
                return False
            left = right
        return True

    def visit_Constant(self, node: ast.Constant) -> Any:
        return node.value

    def visit_Name(self, node: ast.Name) -> Any:
        # `dtype` is handled in Call; any other bare name must be a bound symbol.
        if node.id == "dtype":
            raise UndecidableConstraintError("use dtype(<arg>) not bare 'dtype'")
        if node.id not in self._b:
            raise UndecidableConstraintError(f"unbound symbol '{node.id}' in constraint")
        return self._b[node.id]

    def visit_Call(self, node: ast.Call) -> Any:
        if not (isinstance(node.func, ast.Name) and node.func.id == "dtype"
                and len(node.args) == 1 and not node.keywords):
            raise UndecidableConstraintError(
                "only the dtype(<arg>) builtin call is allowed in constraints"
            )
        arg_name = node.args[0]
        if not isinstance(arg_name, ast.Name):
            raise UndecidableConstraintError("dtype() expects an argument name")
        key = f"dtype:{arg_name.id}"
        if key not in self._b:
            raise UndecidableConstraintError(f"no dtype binding for '{arg_name.id}'")
        return self._b[key]

    def generic_visit(self, node: ast.AST) -> Any:  # pragma: no cover - defensive
        raise UndecidableConstraintError(f"unsupported syntax: {type(node).__name__}")


def validate_decidable(expr: str) -> None:
    """Reject (at ingest) any constraint that isn't in the decidable subset.

    Implements invariant §2.4: "An Op Spec with unsatisfiable/non-decidable
    constraints is rejected at ingest."
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise UndecidableConstraintError(f"invalid syntax: {e}") from e
    _check_grammar(tree, expr)
    _lenient_visit(tree)  # dry-run: surfaces structural/type errors while
    # treating unbound symbols as placeholders (a bare name that isn't bound yet
    # at validation time is fine; it will be bound at query time).
    # syntax/type errors still surface. We swallow only Undecidable for *shape*:
    # bare-name unbinding is a structural problem worth catching, but a name that
    # simply isn't bound yet is fine for validation. So we re-run leniently:
    _lenient_visit(tree)


class _LenientEvaluator(_Evaluator):
    """Same grammar, but treats any unbound name/dtype as a wildcard placeholder."""

    def visit_Name(self, node: ast.Name) -> Any:
        return self._b.get(node.id, 0)

    def visit_Call(self, node: ast.Call) -> Any:
        if isinstance(node.func, ast.Name) and node.func.id == "dtype" and len(node.args) == 1:
            arg = node.args[0]
            name = arg.id if isinstance(arg, ast.Name) else ""
            return self._b.get(f"dtype:{name}", "")
        raise UndecidableConstraintError("unsupported call in constraint")


def _check_grammar(tree: ast.AST, expr: str) -> None:
    for n in ast.walk(tree):
        # expr_context (Load/Store/Del) tags ride on Name/etc.; ignore them.
        if isinstance(n, ast.expr_context):
            continue
        if not isinstance(n, _ALLOWED_NODE_TYPES):
            raise UndecidableConstraintError(
                f"unsupported syntax {type(n).__name__!r} in constraint: {expr!r}"
            )


def _lenient_visit(tree: ast.AST) -> None:
    _LenientEvaluator({}).visit(tree)


def evaluate(expr: str, bindings: dict[str, Any]) -> bool:
    """Evaluate a decidable constraint against ``bindings``.

    ``bindings`` maps shape symbols (e.g. ``"K"``) to ints and
    ``"dtype:<arg>"`` keys to dtype short strings (``"bf16"``). Returns the
    boolean truth of the predicate.
    """
    tree = ast.parse(expr, mode="eval")
    # Enforce grammar on every evaluation too (defense in depth).
    _check_grammar(tree, expr)
    return bool(_Evaluator(bindings).visit(tree))
