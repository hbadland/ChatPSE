"""
Deterministic benchmark mode for experiments (Item 10).

Provides a context manager that enforces strict evaluation discipline:
  - Clears all thermodynamic and state caches between runs
  - Optionally disables LLM calls (full deterministic mode)
  - Logs per-run metrics: iterations, final IR errors, sim convergence, magnitude

Usage:
    mode = BenchmarkMode(disable_llm=True)

    with mode.run("case_ethanol_water"):
        graph, changes = repair_agent.repair(graph, errors, ...)
        report = validate(graph)
        mode.record(
            case_id="case_ethanol_water",
            iterations=3,
            final_ir_errors=len(report.errors()),
            final_ir_warnings=len(report.warnings()),
            sim_converged=True,
            param_magnitude=2.3,
            changes=changes,
        )

    print(mode.report())
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

from ir.thermo_estimation import clear_cache as _clear_thermo_cache


@dataclass
class RunMetrics:
    case_id:            str
    iterations:         int
    final_ir_errors:    int
    final_ir_warnings:  int
    sim_converged:      bool
    param_magnitude:    float   # total normalised magnitude of parameter changes
    elapsed_s:          float
    llm_disabled:       bool
    changes:            list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.final_ir_errors == 0


class BenchmarkMode:
    """
    Evaluation context for reproducible benchmark runs.

    disable_llm=True: LLM candidates return None in RepairAgent._llm_candidate.
    State caches registered via register_cache() are cleared at the start of
    each run context.
    """

    def __init__(
        self,
        disable_llm:         bool = False,
        clear_cache_per_run: bool = True,
    ) -> None:
        self.disable_llm         = disable_llm
        self.clear_cache_per_run = clear_cache_per_run
        self._runs:          list[RunMetrics] = []
        self._state_caches:  list             = []
        self._start_time:    float            = 0.0
        self._current_case:  Optional[str]    = None

    def register_cache(self, cache: object) -> None:
        """Register a StateCache instance to be cleared between runs."""
        self._state_caches.append(cache)

    @contextmanager
    def run(self, case_id: str):
        """Context manager for a single benchmark case."""
        if self.clear_cache_per_run:
            _clear_thermo_cache()
            for c in self._state_caches:
                if hasattr(c, "clear"):
                    c.clear()

        self._current_case = case_id
        self._start_time   = time.monotonic()
        try:
            yield self
        finally:
            self._current_case = None

    def record(
        self,
        case_id:           str,
        iterations:        int,
        final_ir_errors:   int,
        final_ir_warnings: int   = 0,
        sim_converged:     bool  = False,
        param_magnitude:   float = 0.0,
        changes:           Optional[list[str]] = None,
    ) -> RunMetrics:
        elapsed = time.monotonic() - self._start_time
        m = RunMetrics(
            case_id           = case_id,
            iterations        = iterations,
            final_ir_errors   = final_ir_errors,
            final_ir_warnings = final_ir_warnings,
            sim_converged     = sim_converged,
            param_magnitude   = param_magnitude,
            elapsed_s         = elapsed,
            llm_disabled      = self.disable_llm,
            changes           = list(changes or []),
        )
        self._runs.append(m)
        return m

    def report(self) -> str:
        if not self._runs:
            return "BenchmarkMode: no runs recorded."

        n         = len(self._runs)
        n_pass    = sum(1 for r in self._runs if r.passed)
        n_conv    = sum(1 for r in self._runs if r.sim_converged)
        avg_iter  = sum(r.iterations     for r in self._runs) / n
        avg_mag   = sum(r.param_magnitude for r in self._runs) / n
        avg_t     = sum(r.elapsed_s       for r in self._runs) / n
        llm_str   = "disabled" if self._runs[0].llm_disabled else "enabled"

        lines = [
            f"── BenchmarkMode Report (n={n}, llm={llm_str}) ──",
            f"  IR-clean:      {n_pass}/{n}  ({100*n_pass/n:.0f}%)",
            f"  Sim-converged: {n_conv}/{n}  ({100*n_conv/n:.0f}%)",
            f"  Avg iterations: {avg_iter:.1f}",
            f"  Avg magnitude:  {avg_mag:.3f}",
            f"  Avg elapsed:    {avg_t:.2f} s",
            "",
            "  Per-run summary:",
        ]
        for r in self._runs:
            status = "PASS" if r.passed else f"FAIL({r.final_ir_errors}e/{r.final_ir_warnings}w)"
            conv   = "SIM-OK" if r.sim_converged else "SIM-FAIL"
            lines.append(
                f"    {r.case_id:32s} {status:18s} {conv:8s} "
                f"iter={r.iterations:2d} mag={r.param_magnitude:.3f} "
                f"t={r.elapsed_s:.1f}s"
            )
        return "\n".join(lines)

    def runs(self) -> list[RunMetrics]:
        return list(self._runs)

    def pass_rate(self) -> float:
        if not self._runs:
            return 0.0
        return sum(1 for r in self._runs if r.passed) / len(self._runs)
