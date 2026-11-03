from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class SearchStats:
    generated: int = 0
    expanded: int = 0
    pruned: int = 0
    backtracks: int = 0
    max_frontier: int = 0
    max_depth: int = 0
    _started: float = 0.0
    _finished: float = 0.0

    def start(self) -> None:
        self._started = perf_counter()
        self._finished = 0.0

    def finish(self) -> None:
        self._finished = perf_counter()

    def observe_frontier(self, size: int) -> None:
        self.max_frontier = max(self.max_frontier, size)

    def observe_depth(self, depth: int) -> None:
        self.max_depth = max(self.max_depth, depth)

    @property
    def elapsed_seconds(self) -> float:
        end = self._finished or perf_counter()
        return 0.0 if not self._started else end - self._started

    def __str__(self) -> str:
        return (
            f"generated={self.generated}, expanded={self.expanded}, "
            f"pruned={self.pruned}, backtracks={self.backtracks}, "
            f"maxFrontier={self.max_frontier}, maxDepth={self.max_depth}, "
            f"elapsedSeconds={self.elapsed_seconds:.6f}"
        )


@dataclass(frozen=True)
class SolverResult(Generic[T]):
    solution: T | None
    solved: bool
    status: str
    stats: SearchStats


class SearchLimitExceeded(RuntimeError):
    def __init__(self, limit: int) -> None:
        super().__init__(f"Search node limit exceeded: {limit}")


class DotGraph:
    def __init__(self, name: str) -> None:
        self.name = name
        self.nodes: list[str] = []
        self.edges: list[str] = []

    @staticmethod
    def _escape(value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

    def add_node(self, node_id: int, label: str, attributes: str = "") -> None:
        suffix = f", {attributes}" if attributes else ""
        self.nodes.append(f'  n{node_id} [label="{self._escape(label)}"{suffix}];')

    def add_edge(self, source: int, target: int, action: str, attributes: str = "") -> None:
        suffix = f", {attributes}" if attributes else ""
        self.edges.append(
            f'  n{source} -> n{target} [label="{self._escape(action)}"{suffix}];'
        )

    def render(self) -> str:
        lines = [
            f"digraph {self.name} {{",
            "  rankdir=TB;",
            '  node [shape=box, fontname="monospace"];',
            *self.nodes,
            *self.edges,
            "}",
        ]
        return "\n".join(lines) + "\n"

    def write(self, output: Path) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.render(), encoding="utf-8")
