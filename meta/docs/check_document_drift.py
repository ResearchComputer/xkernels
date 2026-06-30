#!/usr/bin/env python3
"""
Document drift detection script.

This script compares documented features in meta/docs/ with the actual codebase in
src/xkernels/ to detect mismatches between documentation and implementation.

Usage:
    python meta/docs/check_document_drift.py

Output:
    - Summary of documented vs implemented public APIs
    - List of any discrepancies found
    - Overall drift status
"""

from __future__ import annotations

import ast
import keyword
import re
from pathlib import Path

# Python keywords and builtins that are not project APIs.
_PY_RESERVED: set[str] = set(keyword.kwlist) | set(dir(__builtins__)) | {
    "self",
    "cls",
    "args",
    "kwargs",
    "true",
    "false",
    "none",
    "returns",
    "return",
    "import",
    "from",
    "as",
    "pass",
    "raise",
    "assert",
    "break",
    "continue",
    "del",
    "global",
    "nonlocal",
    "lambda",
    "yield",
    "await",
    "async",
    # Common English words / PyTorch methods matched inside code blocks.
    "main",
    "sweep",
    "sweep_dense",
    "supports",
    "float",
    "int",
    "str",
    "bool",
    "list",
    "dict",
    "tuple",
    "ceil_div",
    "dot",
    "item",
    "store",
    "to",
    "view",
    "get_device_name",
    "preferred_blas_library",
}


class DocumentDriftChecker:
    """Check for document drift between meta/docs/ and src/xkernels/."""

    def __init__(self, docs_dir: str = "meta/docs", src_dir: str = "src"):
        self.docs_dir = Path(docs_dir)
        self.src_dir = Path(src_dir)

    @staticmethod
    def _is_api_name(name: str) -> bool:
        """Return True if ``name`` looks like a public project API."""
        if not name or name.startswith("_"):
            return False
        if name in _PY_RESERVED:
            return False
        # Must be snake_case.
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            return False
        return True

    def find_documented_apis(
        self, top_level_exports: dict[str, list[str]] | None = None
    ) -> dict[str, list[str]]:
        """Find API names mentioned in docs (including README.md at repo root).

        Only explicit function calls (``func_name(...)``) are counted from issue
        docs; bare backtick names are only counted when they correspond to a
        known top-level export (this catches README performance-table entries
        without flooding the list with parameter names).
        """
        apis: dict[str, list[str]] = {}
        md_files = list(self.docs_dir.rglob("*.md"))
        root_readme = Path("README.md")
        if root_readme.exists():
            md_files.append(root_readme)

        # Match ``func(...)`` or ``xkernels.func(...)`` or ``module.func(...)``.
        call_pattern = re.compile(
            r"`(?:[a-z_][a-z0-9_]*\.)*([a-z_][a-z0-9_]*)\s*\([^)]*\)`"
        )
        name_pattern = re.compile(r"`([a-z_][a-z0-9_]*)`")
        top_names = set(top_level_exports or {})

        for md_file in md_files:
            content = md_file.read_text()
            for match in call_pattern.finditer(content):
                name = match.group(1)
                if self._is_api_name(name):
                    apis.setdefault(name, []).append(str(md_file))
            for match in name_pattern.finditer(content):
                name = match.group(1)
                if name in top_names and self._is_api_name(name):
                    apis.setdefault(name, []).append(str(md_file))
        return apis

    def find_exported_apis(self) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        """Find APIs exported via ``__all__``.

        Returns two mappings:
        - ``top_level``: names exported from ``src/xkernels/__init__.py`` (the
          public user-facing surface).
        - ``all``: names exported from any ``__all__`` in src/xkernels (includes
          submodule helpers that tests/benchmarks import directly).
        """
        top_level: dict[str, list[str]] = {}
        all_exports: dict[str, list[str]] = {}
        top_init = self.src_dir / "xkernels" / "__init__.py"

        for py_file in self.src_dir.rglob("*.py"):
            try:
                tree = ast.parse(py_file.read_text())
            except SyntaxError:
                continue
            is_top_init = top_init.exists() and py_file.samefile(top_init)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Assign):
                    continue
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        value = node.value
                        if isinstance(value, (ast.List, ast.Tuple)):
                            for elt in value.elts:
                                name = None
                                if isinstance(elt, ast.Constant) and isinstance(
                                    elt.value, str
                                ):
                                    name = elt.value
                                elif isinstance(elt, ast.Str):  # py<3.8 compat
                                    name = elt.s
                                if name and self._is_api_name(name):
                                    all_exports.setdefault(name, []).append(
                                        str(py_file)
                                    )
                                    if is_top_init:
                                        top_level.setdefault(name, []).append(
                                            str(py_file)
                                        )
        return top_level, all_exports

    def find_documented_issues(self) -> dict[str, list[str]]:
        """Find issue numbers mentioned in documentation filenames and bodies."""
        issues: dict[str, list[str]] = {}
        issue_pattern = re.compile(r"#(\d+)")

        for md_file in self.docs_dir.rglob("*.md"):
            content = md_file.read_text()
            for match in issue_pattern.finditer(content):
                issue_num = match.group(1)
                issues.setdefault(issue_num, []).append(str(md_file))
        return issues

    def _find_files_mentioning(
        self, directory: Path, pattern: str, apis: set[str]
    ) -> dict[str, list[str]]:
        """Return, for each API, files under ``directory`` whose content mentions it."""
        result: dict[str, list[str]] = {}
        if not directory.exists():
            return result
        regex = re.compile(r"\b(" + "|".join(re.escape(a) for a in apis) + r")\b")
        for file in directory.rglob(pattern):
            content = file.read_text()
            for match in regex.finditer(content):
                result.setdefault(match.group(1), []).append(str(file))
        return result

    def find_test_coverage(self, apis: set[str]) -> dict[str, list[str]]:
        """Find test files that mention each public API."""
        return self._find_files_mentioning(Path("tests"), "test_*.py", apis)

    def find_benchmark_coverage(self, apis: set[str]) -> dict[str, list[str]]:
        """Find benchmark files that mention each public API."""
        return self._find_files_mentioning(Path("meta/benchmarks"), "bench_*.py", apis)

    def find_slurm_jobs(self) -> dict[str, list[str]]:
        """Find SLURM job files and index by filename and by issue number."""
        slurm_jobs: dict[str, list[str]] = {}
        slurm_dir = Path("scripts/slurm")
        if not slurm_dir.exists():
            return slurm_jobs
        for sbatch_file in slurm_dir.glob("*.sbatch"):
            stem = sbatch_file.stem
            slurm_jobs.setdefault(stem, []).append(str(sbatch_file))
            # Index by issue numbers found in the filename or file body.
            text = stem + "\n" + sbatch_file.read_text()
            for match in re.finditer(
                r"issue[_-]?(\d+)|#(\d+)", text, re.IGNORECASE
            ):
                issue_num = match.group(1) or match.group(2)
                slurm_jobs.setdefault(issue_num, []).append(str(sbatch_file))
        return slurm_jobs

    def check_consistency(self) -> dict:
        """Run all checks and return consistency report."""
        # Need exports first so documented-API detection can use top-level names.
        top_level_exports, all_exports = self.find_exported_apis()
        documented_apis = self.find_documented_apis(top_level_exports)
        documented_issues = self.find_documented_issues()

        api_names = set(documented_apis.keys()) | set(all_exports.keys())
        test_coverage = self.find_test_coverage(api_names)
        benchmark_coverage = self.find_benchmark_coverage(api_names)
        slurm_jobs = self.find_slurm_jobs()

        return {
            "documented_apis": documented_apis,
            "top_level_exports": top_level_exports,
            "all_exports": all_exports,
            "documented_issues": documented_issues,
            "test_coverage": test_coverage,
            "benchmark_coverage": benchmark_coverage,
            "slurm_jobs": slurm_jobs,
        }

    @staticmethod
    def _dedupe(paths: list[str]) -> list[str]:
        return sorted(set(paths))

    def generate_summary(self, report: dict) -> str:
        """Generate a human-readable summary of the drift check."""
        documented_apis = report["documented_apis"]
        top_level_exports = report["top_level_exports"]
        all_exports = report["all_exports"]
        documented_issues = report["documented_issues"]
        test_coverage = report["test_coverage"]
        benchmark_coverage = report["benchmark_coverage"]
        slurm_jobs = report["slurm_jobs"]

        lines = [
            "=" * 70,
            "DOCUMENT DRIFT DETECTION SUMMARY",
            "=" * 70,
            "",
            f"Total documented APIs: {len(documented_apis)}",
            f"Total top-level public APIs: {len(top_level_exports)}",
            f"Total submodule-exported APIs: {len(all_exports) - len(top_level_exports)}",
            f"Total documented issues: {len(documented_issues)}",
            f"Total APIs with test coverage: {len(test_coverage)}",
            f"Total APIs with benchmark coverage: {len(benchmark_coverage)}",
            f"Total SLURM jobs found: {len(slurm_jobs)}",
            "",
        ]

        # Documented APIs without a corresponding exported symbol. These are
        # often internal framework helpers (register, dispatch, detect_vendor,
        # benchmark) that are documented for contributors but not exported as
        # user APIs.
        missing_exports = set(documented_apis.keys()) - set(all_exports.keys())
        if missing_exports:
            lines.append(
                "ℹ️  Documented APIs NOT exported in any src/xkernels/__all__ "
                "(internal helpers):"
            )
            lines.append("-" * 70)
            for api in sorted(missing_exports):
                lines.append(f"  {api}:")
                for doc in self._dedupe(documented_apis[api]):
                    lines.append(f"    - {doc}")
            lines.append("")

        # Top-level public APIs without any documentation mention.
        undocumented_public = set(top_level_exports.keys()) - set(documented_apis.keys())
        if undocumented_public:
            lines.append("⚠️  Top-level public APIs WITHOUT documentation mention:")
            lines.append("-" * 70)
            for api in sorted(undocumented_public):
                lines.append(f"  {api}:")
                for src in self._dedupe(top_level_exports[api]):
                    lines.append(f"    - {src}")
            lines.append("")

        # Submodule helpers without docs — informational only.
        undocumented_submodule = (
            set(all_exports.keys()) - set(top_level_exports.keys()) - set(documented_apis.keys())
        )
        if undocumented_submodule:
            lines.append(
                "ℹ️  Submodule-exported helpers WITHOUT documentation mention "
                "(informational):"
            )
            lines.append("-" * 70)
            for api in sorted(undocumented_submodule)[:20]:
                lines.append(f"  {api}")
            if len(undocumented_submodule) > 20:
                lines.append(f"  ... and {len(undocumented_submodule) - 20} more")
            lines.append("")

        # Test coverage.
        lines.append("Test Coverage (documented APIs):")
        lines.append("-" * 70)
        documented_with_tests = set(documented_apis.keys()) & set(test_coverage.keys())
        lines.append(
            f"  Documented APIs with tests: "
            f"{len(documented_with_tests)}/{len(documented_apis)}"
        )
        for api in sorted(documented_with_tests):
            for test in self._dedupe(test_coverage[api]):
                lines.append(f"    {api}: {test}")
        lines.append("")

        # Benchmark coverage.
        lines.append("Benchmark Coverage (documented APIs):")
        lines.append("-" * 70)
        documented_with_benches = set(documented_apis.keys()) & set(
            benchmark_coverage.keys()
        )
        lines.append(
            f"  Documented APIs with benchmarks: "
            f"{len(documented_with_benches)}/{len(documented_apis)}"
        )
        for api in sorted(documented_with_benches):
            for bench in self._dedupe(benchmark_coverage[api]):
                lines.append(f"    {api}: {bench}")
        lines.append("")

        # SLURM job coverage by issue.
        lines.append("On-Device Test Coverage (documented issues):")
        lines.append("-" * 70)
        documented_with_jobs = set(documented_issues.keys()) & set(slurm_jobs.keys())
        lines.append(
            f"  Documented issues with SLURM jobs: "
            f"{len(documented_with_jobs)}/{len(documented_issues)}"
        )
        for issue in sorted(documented_with_jobs, key=int):
            for job in self._dedupe(slurm_jobs[issue]):
                lines.append(f"    #{issue}: {job}")
        lines.append("")

        # Overall status.
        if not undocumented_public:
            lines.append("✅ OVERALL STATUS: CLEAN")
            lines.append(
                "All top-level public APIs are documented. "
                "(Internal helper symbols listed above are informational only.)"
            )
        else:
            lines.append("⚠️  OVERALL STATUS: SOME DRIFT DETECTED")
            lines.append(
                "Some top-level public APIs are not mentioned in the documentation."
            )
            lines.append(
                f"  - {len(undocumented_public)} top-level public APIs not documented"
            )

        lines.append("=" * 70)
        return "\n".join(lines)

    def run(self) -> int:
        """Run the drift check and return exit code."""
        report = self.check_consistency()
        summary = self.generate_summary(report)
        print(summary)

        output_file = self.docs_dir / "DRIFT_CHECK_REPORT.txt"
        output_file.write_text(summary)
        print(f"\nDetailed report saved to: {output_file}")

        # Only treat undocumented top-level public APIs as a hard failure.
        # Documented-but-not-exported symbols are typically internal framework
        # helpers described for contributors (e.g. dispatch, register).
        undocumented_public = set(report["top_level_exports"].keys()) - set(
            report["documented_apis"].keys()
        )
        return 0 if not undocumented_public else 1


def main() -> None:
    import sys

    checker = DocumentDriftChecker()
    sys.exit(checker.run())


if __name__ == "__main__":
    main()
