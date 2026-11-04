from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_assignment import golfers, latin, sudoku, trees


def board(text: str) -> tuple[int, ...]:
    return tuple(int(value) for value in "".join(text.split()))


EASY = board(
    """
    530070000
    600195000
    098000060
    800060003
    400803001
    700020006
    060000280
    000419005
    000080079
    """
)


class AssignmentTests(unittest.TestCase):
    def test_sudoku_backtracking(self) -> None:
        result = sudoku.solve_backtracking(EASY)
        self.assertTrue(result.solved)
        self.assertTrue(sudoku.is_valid_solution(result.solution, EASY))
        self.assertEqual(sudoku.objective(result.solution), 0)  # type: ignore[arg-type]

    def test_sudoku_bfs(self) -> None:
        result = sudoku.solve_bfs(EASY)
        self.assertTrue(result.solved)
        self.assertTrue(sudoku.is_valid_solution(result.solution, EASY))

    def test_invalid_sudoku(self) -> None:
        invalid = list(EASY)
        invalid[1] = 5
        with self.assertRaises(ValueError):
            sudoku.solve_backtracking(invalid)

    def test_latin_square(self) -> None:
        result = latin.solve_iddfs(3)
        self.assertTrue(result.solved)
        self.assertTrue(latin.is_valid(result.solution))
        self.assertEqual(latin.objective(result.solution), 0)  # type: ignore[arg-type]

    def test_social_golfers(self) -> None:
        result = golfers.solve(2, 2, 3)
        self.assertTrue(result.solved)
        self.assertTrue(golfers.is_valid(result.solution, 2, 2))
        self.assertEqual(golfers.objective(result.solution, 4), 0)  # type: ignore[arg-type]

    def test_impossible_social_golfers_bound(self) -> None:
        self.assertFalse(golfers.solve(2, 2, 4).solved)

    def test_social_golfers_maximum_search(self) -> None:
        result = golfers.maximize(2, 2)
        self.assertTrue(result.proven)
        self.assertEqual(result.maximum_weeks, 3)
        self.assertEqual(result.upper_bound, 3)
        self.assertTrue(golfers.is_valid(result.schedule, 2, 2))

    def test_published_optimal_schedule(self) -> None:
        schedule = golfers.read_schedule(Path("data/social-golfers/8-4-10-aguado.txt"), 8, 4)
        self.assertEqual(len(schedule), 10)
        self.assertTrue(golfers.is_valid(schedule, 8, 4))
        self.assertEqual(golfers.objective(schedule, 32), 0)

    def test_optimal_schedule_is_generated_without_fixture(self) -> None:
        with patch.object(
            golfers,
            "read_schedule",
            side_effect=AssertionError("The solver must not read a saved schedule"),
        ):
            result = golfers.maximize(8, 4, node_limit=1_000)
        self.assertTrue(result.proven)
        self.assertEqual(result.maximum_weeks, 10)
        self.assertEqual(result.upper_bound, 10)
        self.assertTrue(golfers.is_valid(result.schedule, 8, 4))
        self.assertEqual(golfers.objective(result.schedule, 32), 0)
        self.assertGreater(result.stats.expanded, 0)

    def test_puzzle_tree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = trees.generate_puzzle_tree(
                trees.parse_puzzle("123456708"), 2, Path(directory) / "puzzle.dot"
            )
            self.assertTrue(result.goal_found)
            self.assertEqual(result.solution_actions, ("Right",))

    def test_missionaries_tree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = trees.generate_missionaries_tree(Path(directory) / "missionaries.dot")
            self.assertTrue(result.goal_found)
            self.assertEqual(len(result.solution_actions), 11)

    def test_workforce_tree(self) -> None:
        config = trees.WorkforceConfig(
            days=3,
            shifts=("D", "N", "O"),
            max_consecutive_work=2,
            minimum_days_off=1,
            forbidden_transitions=frozenset({"N>D"}),
            employees=3,
            coverage=(("D", 1), ("N", 1), ("O", 1)),
        )
        with tempfile.TemporaryDirectory() as directory:
            result = trees.generate_workforce_tree(config, 3, Path(directory) / "workforce.dot")
            self.assertTrue(result.complete_schedule_found)
            self.assertTrue(trees.workforce_valid_complete(result.schedule, config))
            self.assertEqual(len(result.schedule), 3)
            for day in range(3):
                self.assertEqual(
                    sorted(result.schedule[employee][day] for employee in range(3)),
                    ["D", "N", "O"],
                )


if __name__ == "__main__":
    unittest.main()
