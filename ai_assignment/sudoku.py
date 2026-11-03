from __future__ import annotations

from collections import deque
from pathlib import Path

from ai_assignment.common import SearchLimitExceeded, SearchStats, SolverResult

ALL_DIGITS = sum(1 << value for value in range(1, 10))


def read(path: Path) -> tuple[int, ...]:
    text = "".join(char for char in path.read_text(encoding="utf-8") if char.isdigit() or char == ".")
    if len(text) != 81:
        raise ValueError("A Sudoku input must contain exactly 81 digits/dots")
    board = tuple(0 if char == "." else int(char) for char in text)
    require_valid_partial(board)
    return board


def candidate_mask(board: list[int] | tuple[int, ...], cell: int) -> int:
    if board[cell]:
        return 0
    row, column = divmod(cell, 9)
    used = 0
    for index in range(9):
        used |= 1 << board[row * 9 + index]
        used |= 1 << board[index * 9 + column]
    box_row, box_column = row // 3 * 3, column // 3 * 3
    for dr in range(3):
        for dc in range(3):
            used |= 1 << board[(box_row + dr) * 9 + box_column + dc]
    return ALL_DIGITS & ~used


def _propagate_singles(board: list[int], stats: SearchStats) -> bool:
    changed = True
    while changed:
        changed = False
        for cell, current in enumerate(board):
            if current:
                continue
            mask = candidate_mask(board, cell)
            if not mask:
                return False
            if mask.bit_count() == 1:
                board[cell] = mask.bit_length() - 1
                stats.generated += 1
                changed = True
    return True


def solve_bfs(input_board: tuple[int, ...] | list[int], node_limit: int = 1_000_000) -> SolverResult[tuple[int, ...]]:
    require_valid_partial(input_board)
    stats = SearchStats()
    stats.start()
    root = list(input_board)
    if not _propagate_singles(root, stats):
        stats.finish()
        return SolverResult(None, False, "Initial puzzle is contradictory", stats)
    frontier: deque[list[int]] = deque([root])
    stats.generated += 1
    stats.observe_frontier(1)
    try:
        while frontier:
            _enforce_limit(stats, node_limit)
            board = frontier.popleft()
            stats.expanded += 1
            stats.observe_depth(sum(value != 0 for value in board))
            try:
                cell = board.index(0)
            except ValueError:
                stats.finish()
                return SolverResult(tuple(board), True, "Solved with breadth-first search", stats)
            mask = candidate_mask(board, cell)
            for value in range(1, 10):
                if not mask & (1 << value):
                    continue
                child = board.copy()
                child[cell] = value
                stats.generated += 1
                if _propagate_singles(child, stats):
                    frontier.append(child)
                else:
                    stats.pruned += 1
            stats.observe_frontier(len(frontier))
        stats.finish()
        return SolverResult(None, False, "No solution exists", stats)
    except SearchLimitExceeded as error:
        stats.finish()
        return SolverResult(None, False, str(error), stats)


def solve_backtracking(input_board: tuple[int, ...] | list[int], node_limit: int = 1_000_000) -> SolverResult[tuple[int, ...]]:
    require_valid_partial(input_board)
    stats = SearchStats()
    stats.start()

    def backtrack(board: list[int]) -> tuple[int, ...] | None:
        _enforce_limit(stats, node_limit)
        stats.generated += 1
        stats.expanded += 1
        if not _propagate_singles(board, stats):
            stats.pruned += 1
            return None
        stats.observe_depth(sum(value != 0 for value in board))
        empty = [cell for cell, value in enumerate(board) if value == 0]
        if not empty:
            return tuple(board)
        cell = min(empty, key=lambda index: candidate_mask(board, index).bit_count())
        mask = candidate_mask(board, cell)
        for value in range(1, 10):
            if mask & (1 << value):
                child = board.copy()
                child[cell] = value
                solution = backtrack(child)
                if solution is not None:
                    return solution
                stats.backtracks += 1
        return None

    try:
        solution = backtrack(list(input_board))
        stats.finish()
        return SolverResult(
            solution,
            solution is not None,
            "Solved with DFS backtracking" if solution else "No solution exists",
            stats,
        )
    except SearchLimitExceeded as error:
        stats.finish()
        return SolverResult(None, False, str(error), stats)


def is_valid_solution(solution: tuple[int, ...] | None, givens: tuple[int, ...] | list[int]) -> bool:
    if solution is None or len(solution) != 81 or len(givens) != 81:
        return False
    if any(not 1 <= value <= 9 for value in solution):
        return False
    if any(given and given != solution[cell] for cell, given in enumerate(givens)):
        return False
    try:
        require_valid_partial(solution)
        return True
    except ValueError:
        return False


def objective(board: tuple[int, ...] | list[int]) -> int:
    conflicts = 0
    for index in range(9):
        conflicts += _duplicates(_row(board, index))
        conflicts += _duplicates(_column(board, index))
        conflicts += _duplicates(_box(board, index))
    return board.count(0) + 100 * conflicts


def format_board(board: tuple[int, ...] | list[int]) -> str:
    lines: list[str] = []
    for row in range(9):
        if row and row % 3 == 0:
            lines.append("------+-------+------")
        cells: list[str] = []
        for column in range(9):
            if column and column % 3 == 0:
                cells.append("|")
            value = board[row * 9 + column]
            cells.append(str(value) if value else ".")
        lines.append(" ".join(cells))
    return "\n".join(lines)


def require_valid_partial(board: tuple[int, ...] | list[int]) -> None:
    if len(board) != 81 or any(not 0 <= value <= 9 for value in board):
        raise ValueError("A Sudoku board must contain 81 values between 0 and 9")
    for index in range(9):
        if any(_duplicates(values) for values in (_row(board, index), _column(board, index), _box(board, index))):
            raise ValueError("Sudoku givens contain a duplicate")


def _row(board: tuple[int, ...] | list[int], row: int) -> list[int]:
    return list(board[row * 9 : row * 9 + 9])


def _column(board: tuple[int, ...] | list[int], column: int) -> list[int]:
    return [board[row * 9 + column] for row in range(9)]


def _box(board: tuple[int, ...] | list[int], box: int) -> list[int]:
    start_row, start_column = box // 3 * 3, box % 3 * 3
    return [board[(start_row + dr) * 9 + start_column + dc] for dr in range(3) for dc in range(3)]


def _duplicates(values: list[int]) -> int:
    populated = [value for value in values if value]
    return len(populated) - len(set(populated))


def _enforce_limit(stats: SearchStats, limit: int) -> None:
    if limit > 0 and stats.expanded >= limit:
        raise SearchLimitExceeded(limit)
