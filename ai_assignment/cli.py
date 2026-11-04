from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from ai_assignment import golfers, latin, sudoku, trees


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Artificial Intelligence Search Assignment")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sudoku_parser = subparsers.add_parser("sudoku", help="solve a Sudoku")
    sudoku_parser.add_argument("file", type=Path)
    sudoku_parser.add_argument("algorithm", choices=("bfs", "backtracking", "dfs"), nargs="?", default="backtracking")
    sudoku_parser.add_argument("node_limit", type=int, nargs="?", default=1_000_000)

    latin_parser = subparsers.add_parser("latin", help="solve a Latin Square with IDDFS")
    latin_parser.add_argument("size", type=int, nargs="?", default=3)
    latin_parser.add_argument("node_limit", type=int, nargs="?", default=5_000_000)

    golf_parser = subparsers.add_parser("golfers", help="solve Social Golfers with DFS backtracking")
    golf_parser.add_argument("groups", type=int, nargs="?", default=2)
    golf_parser.add_argument("group_size", type=int, nargs="?", default=2)
    golf_parser.add_argument("weeks", type=int, nargs="?", default=3)
    golf_parser.add_argument("node_limit", type=int, nargs="?", default=5_000_000)

    golf_max = subparsers.add_parser("golfers-max", help="find and prove the maximum week count")
    golf_max.add_argument("groups", type=int, nargs="?", default=8)
    golf_max.add_argument("group_size", type=int, nargs="?", default=4)
    golf_max.add_argument("node_limit", type=int, nargs="?", default=5_000_000)
    golf_validate = subparsers.add_parser("golfers-validate", help="validate a saved Social Golfers schedule")
    golf_validate.add_argument("file", type=Path, nargs="?", default=Path("data/social-golfers/8-4-10-aguado.txt"))
    golf_validate.add_argument("groups", type=int, nargs="?", default=8)
    golf_validate.add_argument("group_size", type=int, nargs="?", default=4)

    puzzle = subparsers.add_parser("tree-puzzle8", help="export a bounded 8-Puzzle tree")
    puzzle.add_argument("state", nargs="?", default="123456708")
    puzzle.add_argument("depth", type=int, nargs="?", default=4)
    puzzle.add_argument("output", type=Path, nargs="?", default=Path("generated-trees/puzzle8.dot"))

    missionaries = subparsers.add_parser("tree-missionaries", help="export Missionaries and Cannibals tree")
    missionaries.add_argument("output", type=Path, nargs="?", default=Path("generated-trees/missionaries.dot"))

    workforce = subparsers.add_parser("tree-workforce", help="export a workforce scheduling tree")
    workforce.add_argument("config", type=Path, nargs="?", default=Path("data/workforce/example.properties"))
    workforce.add_argument("depth", type=int, nargs="?", default=None)
    workforce.add_argument("output", type=Path, nargs="?", default=Path("generated-trees/workforce.dot"))

    subparsers.add_parser("demo", help="run every module")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "sudoku":
            _run_sudoku(args.file, args.algorithm, args.node_limit)
        elif args.command == "latin":
            _run_latin(args.size, args.node_limit)
        elif args.command == "golfers":
            _run_golfers(args.groups, args.group_size, args.weeks, args.node_limit)
        elif args.command == "golfers-max":
            _run_golfers_max(args.groups, args.group_size, args.node_limit)
        elif args.command == "golfers-validate":
            _run_golfers_validation(args.file, args.groups, args.group_size)
        elif args.command == "tree-puzzle8":
            _run_puzzle(args.state, args.depth, args.output)
        elif args.command == "tree-missionaries":
            _run_missionaries(args.output)
        elif args.command == "tree-workforce":
            _run_workforce(args.config, args.depth, args.output)
        elif args.command == "demo":
            _run_demo()
        return 0
    except (OSError, ValueError) as error:
        print(f"Error: {error}")
        return 1


def _run_sudoku(path: Path, algorithm: str, node_limit: int) -> None:
    givens = sudoku.read(path)
    result = (
        sudoku.solve_bfs(givens, node_limit)
        if algorithm == "bfs"
        else sudoku.solve_backtracking(givens, node_limit)
    )
    print(result.status)
    if result.solution:
        print(sudoku.format_board(result.solution))
        print(f"valid={str(sudoku.is_valid_solution(result.solution, givens)).lower()}")
        print(f"objective={sudoku.objective(result.solution)}")
    print(result.stats)


def _run_latin(size: int, node_limit: int) -> None:
    result = latin.solve_iddfs(size, node_limit)
    print(result.status)
    if result.solution:
        print(latin.format_grid(result.solution))
        print(f"valid={str(latin.is_valid(result.solution)).lower()}")
        print(f"objective={latin.objective(result.solution)}")
    print(result.stats)


def _run_golfers(groups: int, group_size: int, weeks: int, node_limit: int) -> None:
    result = golfers.solve(groups, group_size, weeks, node_limit)
    print(result.status)
    if result.solution:
        print(golfers.format_schedule(result.solution))
        print(f"valid={str(golfers.is_valid(result.solution, groups, group_size)).lower()}")
        print(f"objective={golfers.objective(result.solution, groups * group_size)}")
    print(result.stats)


def _run_golfers_max(groups: int, group_size: int, node_limit: int) -> None:
    result = golfers.maximize(groups, group_size, node_limit)
    print(result.status)
    print(f"maximumWeeks={result.maximum_weeks}")
    print(f"upperBound={result.upper_bound}")
    print(f"proven={str(result.proven).lower()}")
    if result.schedule:
        print(golfers.format_schedule(result.schedule))
        print(f"valid={str(golfers.is_valid(result.schedule, groups, group_size)).lower()}")
        print(f"objective={golfers.objective(result.schedule, groups * group_size)}")
    print(result.stats)


def _run_golfers_validation(path: Path, groups: int, group_size: int) -> None:
    schedule = golfers.read_schedule(path, groups, group_size)
    valid = golfers.is_valid(schedule, groups, group_size)
    print(f"weeks={len(schedule)}")
    print(f"valid={str(valid).lower()}")
    print(f"objective={golfers.objective(schedule, groups * group_size)}")
    if not valid:
        raise ValueError("The supplied Social Golfers schedule is invalid")


def _run_puzzle(state: str, depth: int, output: Path) -> None:
    result = trees.generate_puzzle_tree(trees.parse_puzzle(state), depth, output)
    print(f"nodes={result.nodes}")
    print(f"goalFound={str(result.goal_found).lower()}")
    print(f"solutionActions={list(result.solution_actions)}")
    print(f"output={result.output.resolve()}")


def _run_missionaries(output: Path) -> None:
    result = trees.generate_missionaries_tree(output)
    print(f"nodes={result.nodes}")
    print(f"goalFound={str(result.goal_found).lower()}")
    print(f"crossings={len(result.solution_actions)}")
    print(f"solutionActions={list(result.solution_actions)}")
    print(f"output={result.output.resolve()}")


def _run_workforce(config_path: Path, depth: int | None, output: Path) -> None:
    config = trees.WorkforceConfig.read(config_path)
    result = trees.generate_workforce_tree(config, config.days if depth is None else depth, output)
    print(f"nodes={result.nodes}")
    print(f"completeScheduleFound={str(result.complete_schedule_found).lower()}")
    for employee, shifts in enumerate(result.schedule, 1):
        print(f"employee{employee}={list(shifts)}")
    print(f"output={result.output.resolve()}")


def _run_demo() -> None:
    print("=== Sudoku Easy / BFS ===")
    _run_sudoku(Path("data/sudoku/easy.txt"), "bfs", 1_000_000)
    print("\n=== Latin Square n=3 / IDDFS ===")
    _run_latin(3, 1_000_000)
    print("\n=== Social Golfers 2-2-3 / DFS backtracking ===")
    _run_golfers(2, 2, 3, 1_000_000)
    print("\n=== Search trees ===")
    _run_puzzle("123456708", 4, Path("generated-trees/puzzle8.dot"))
    _run_missionaries(Path("generated-trees/missionaries.dot"))
    _run_workforce(Path("data/workforce/example.properties"), None, Path("generated-trees/workforce.dot"))
