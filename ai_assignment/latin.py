from __future__ import annotations

from ai_assignment.common import SearchLimitExceeded, SearchStats, SolverResult

Grid = tuple[tuple[int, ...], ...]


def solve_iddfs(size: int, node_limit: int = 5_000_000) -> SolverResult[Grid]:
    if not 1 <= size <= 9:
        raise ValueError("Latin Square size must be between 1 and 9")
    initial = [[0] * size for _ in range(size)]
    for index in range(size):
        initial[0][index] = index + 1
        initial[index][0] = index + 1
    initially_filled = 2 * size - 1
    stats = SearchStats()
    stats.start()

    def depth_limited(grid: list[list[int]], filled: int, limit: int) -> tuple[str, Grid | None]:
        if node_limit > 0 and stats.expanded >= node_limit:
            raise SearchLimitExceeded(node_limit)
        stats.generated += 1
        stats.expanded += 1
        stats.observe_depth(filled)
        stats.observe_frontier(filled + 1)
        cell = _next_empty(grid)
        if cell is None:
            return "found", tuple(tuple(row) for row in grid)
        if filled == limit:
            return "cutoff", None
        cutoff = False
        row, column = cell
        for value in range(1, size + 1):
            if not _legal(grid, row, column, value):
                stats.pruned += 1
                continue
            grid[row][column] = value
            outcome, solution = depth_limited(grid, filled + 1, limit)
            if outcome == "found":
                return outcome, solution
            cutoff |= outcome == "cutoff"
            grid[row][column] = 0
            stats.backtracks += 1
        return ("cutoff" if cutoff else "failure"), None

    try:
        for limit in range(initially_filled, size * size + 1):
            candidate = [row.copy() for row in initial]
            outcome, solution = depth_limited(candidate, initially_filled, limit)
            if outcome == "found":
                stats.finish()
                return SolverResult(solution, True, f"Solved with IDDFS at depth limit {limit}", stats)
            if outcome == "failure":
                break
        stats.finish()
        return SolverResult(None, False, "No Latin Square found", stats)
    except SearchLimitExceeded as error:
        stats.finish()
        return SolverResult(None, False, str(error), stats)


def _next_empty(grid: list[list[int]]) -> tuple[int, int] | None:
    for row, values in enumerate(grid):
        for column, value in enumerate(values):
            if value == 0:
                return row, column
    return None


def _legal(grid: list[list[int]], row: int, column: int, value: int) -> bool:
    return value not in grid[row] and all(other[column] != value for other in grid)


def is_valid(grid: Grid | None) -> bool:
    if not grid or any(len(row) != len(grid) for row in grid):
        return False
    expected = set(range(1, len(grid) + 1))
    return all(set(row) == expected for row in grid) and all(
        {grid[row][column] for row in range(len(grid))} == expected
        for column in range(len(grid))
    )


def objective(grid: Grid) -> int:
    size = len(grid)
    blanks = sum(value == 0 for row in grid for value in row)
    duplicates = 0
    for index in range(size):
        row = [value for value in grid[index] if value]
        column = [grid[other][index] for other in range(size) if grid[other][index]]
        duplicates += len(row) - len(set(row)) + len(column) - len(set(column))
    return blanks + 100 * duplicates


def cyclic_construction(size: int) -> Grid:
    return tuple(tuple((row + column) % size + 1 for column in range(size)) for row in range(size))


def format_grid(grid: Grid) -> str:
    return "\n".join(str(list(row)) for row in grid)
