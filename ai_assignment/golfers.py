from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

from ai_assignment.common import SearchLimitExceeded, SearchStats, SolverResult

Schedule = tuple[tuple[tuple[int, ...], ...], ...]


@dataclass(frozen=True)
class MaximumResult:
    schedule: Schedule | None
    maximum_weeks: int
    upper_bound: int
    proven: bool
    status: str
    stats: SearchStats


# Parallel classes of a 4-RGDD of type 2^16. Each value represents one
# non-meeting pair of golfers; DFS below assigns the two actual golfers.
_RGDD_2_16_CLASSES = (
    ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14, 15)),
    ((0, 4, 8, 12), (1, 2, 6, 7), (3, 5, 10, 15), (9, 11, 13, 14)),
    ((0, 5, 11, 14), (1, 7, 10, 12), (2, 4, 9, 15), (3, 6, 8, 13)),
    ((0, 2, 8, 11), (1, 3, 9, 10), (4, 7, 12, 13), (5, 6, 14, 15)),
    ((0, 6, 10, 13), (1, 4, 11, 15), (2, 5, 9, 12), (3, 7, 8, 14)),
    ((0, 7, 9, 15), (1, 5, 8, 13), (2, 3, 12, 14), (4, 6, 10, 11)),
    ((0, 3, 4, 5), (1, 6, 9, 14), (2, 7, 11, 13), (8, 10, 12, 15)),
    ((0, 1, 13, 15), (2, 4, 10, 14), (3, 6, 11, 12), (5, 7, 8, 9)),
    ((0, 6, 9, 12), (1, 4, 8, 14), (2, 5, 10, 13), (3, 7, 11, 15)),
    ((0, 7, 10, 14), (1, 5, 11, 12), (2, 6, 8, 15), (3, 4, 9, 13)),
)


def maximum_weeks_upper_bound(groups: int, group_size: int) -> int:
    """Return the pair-count upper bound floor((g*p - 1)/(p - 1))."""
    if groups < 1 or group_size < 1:
        raise ValueError("groups and group_size must be positive")
    if group_size == 1:
        raise ValueError("The maximum is unbounded when every group contains one player")
    return (groups * group_size - 1) // (group_size - 1)


def solve_optimal_8_4(node_limit: int = 5_000_000) -> SolverResult[Schedule]:
    """Construct the optimal 8-4-10 schedule with DFS and backtracking.

    The RGDD template fixes only four non-meeting pairs per block. DFS chooses
    one of eight possible orientations for every block. When two pair-groups
    meet for the second time, their orientation must be opposite so that the
    same two individual golfers never meet twice.
    """
    if node_limit < 0:
        raise ValueError("node_limit cannot be negative")

    pair_counts: dict[tuple[int, int], int] = {}
    for week in _RGDD_2_16_CLASSES:
        if sorted(team for block in week for team in block) != list(range(16)):
            raise RuntimeError("Every RGDD class must partition the 16 non-meeting pairs")
        for block in week:
            for pair in combinations(block, 2):
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
    if len(pair_counts) != 120 or any(count != 2 for count in pair_counts.values()):
        raise RuntimeError("The RGDD template must contain every pair exactly twice")

    blocks = tuple(block for week in _RGDD_2_16_CLASSES for block in week)
    assigned: list[int | None] = [None] * len(blocks)
    first_relations: dict[tuple[int, int], int] = {}
    stats = SearchStats()
    stats.start()

    def pattern_relations(
        block: tuple[int, ...], pattern: int
    ) -> tuple[tuple[tuple[int, int], int], ...]:
        bits = (0, pattern & 1, (pattern >> 1) & 1, (pattern >> 2) & 1)
        return tuple(
            ((block[left], block[right]), bits[left] ^ bits[right])
            for left, right in combinations(range(4), 2)
        )

    def place(block_index: int, pattern: int) -> tuple[tuple[int, int], ...] | None:
        added: list[tuple[int, int]] = []
        for pair, relation in pattern_relations(blocks[block_index], pattern):
            previous = first_relations.get(pair)
            if previous is not None:
                if previous == relation:
                    for added_pair in added:
                        del first_relations[added_pair]
                    return None
            else:
                first_relations[pair] = relation
                added.append(pair)
        assigned[block_index] = pattern
        return tuple(added)

    def unplace(block_index: int, added: tuple[tuple[int, int], ...]) -> None:
        assigned[block_index] = None
        for pair in added:
            del first_relations[pair]

    # The first parallel class is canonical symmetry breaking. Relabeling the
    # two golfers in each non-meeting pair makes all four patterns zero.
    for block_index in range(4):
        added = place(block_index, 0)
        if added is None:
            raise RuntimeError("Invalid RGDD symmetry-breaking class")
        stats.generated += 1
    stats.observe_depth(4)

    def search(assigned_count: int) -> bool:
        if node_limit > 0 and stats.expanded >= node_limit:
            raise SearchLimitExceeded(node_limit)
        stats.expanded += 1
        stats.observe_depth(assigned_count)
        if assigned_count == len(blocks):
            return True

        best_index = -1
        best_patterns: list[int] | None = None
        for block_index in range(4, len(blocks)):
            if assigned[block_index] is not None:
                continue
            legal_patterns: list[int] = []
            for pattern in range(8):
                if all(
                    first_relations.get(pair) != relation
                    for pair, relation in pattern_relations(blocks[block_index], pattern)
                    if pair in first_relations
                ):
                    legal_patterns.append(pattern)
                else:
                    stats.pruned += 1
            if not legal_patterns:
                return False
            if best_patterns is None or len(legal_patterns) < len(best_patterns):
                best_index = block_index
                best_patterns = legal_patterns

        if best_patterns is None:
            return False
        stats.observe_frontier(len(best_patterns))
        for pattern in best_patterns:
            stats.generated += 1
            added = place(best_index, pattern)
            if added is None:
                stats.pruned += 1
                continue
            if search(assigned_count + 1):
                return True
            unplace(best_index, added)
            stats.backtracks += 1
        return False

    try:
        solved = search(4)
    except SearchLimitExceeded as error:
        stats.finish()
        return SolverResult(None, False, str(error), stats)

    if not solved:
        stats.finish()
        return SolverResult(None, False, "No 8-4-10 RGDD orientation exists", stats)

    weeks: list[tuple[tuple[int, ...], ...]] = []
    for week_index, week_blocks in enumerate(_RGDD_2_16_CLASSES):
        groups: list[tuple[int, ...]] = []
        for block_offset, block in enumerate(week_blocks):
            pattern = assigned[week_index * 4 + block_offset]
            if pattern is None:
                raise RuntimeError("DFS returned an incomplete RGDD orientation")
            bits = (0, pattern & 1, (pattern >> 1) & 1, (pattern >> 2) & 1)
            groups.append(tuple(sorted(team + 16 * bit for team, bit in zip(block, bits))))
            groups.append(
                tuple(sorted(team + 16 * (1 - bit) for team, bit in zip(block, bits)))
            )
        weeks.append(tuple(sorted(groups)))

    schedule = tuple(weeks)
    stats.finish()
    if not is_valid(schedule, 8, 4):
        raise RuntimeError("DFS produced an invalid 8-4-10 schedule")
    return SolverResult(
        schedule,
        True,
        "Solved optimal 8-4-10 with DFS backtracking over a 4-RGDD",
        stats,
    )


def solve(groups: int, group_size: int, weeks: int, node_limit: int = 5_000_000) -> SolverResult[Schedule]:
    if min(groups, group_size, weeks) < 1:
        raise ValueError("groups, group_size and weeks must all be positive")
    if (groups, group_size, weeks) == (8, 4, 10):
        return solve_optimal_8_4(node_limit)
    player_count = groups * group_size
    stats = SearchStats()
    stats.start()
    if group_size > 1 and weeks > maximum_weeks_upper_bound(groups, group_size):
        stats.finish()
        maximum = maximum_weeks_upper_bound(groups, group_size)
        return SolverResult(None, False, f"Impossible by pair-count upper bound: maximum weeks is {maximum}", stats)

    first_week = tuple(
        tuple(range(group * group_size, (group + 1) * group_size)) for group in range(groups)
    )
    partial: list[tuple[tuple[int, ...], ...]] = [first_week]
    used_pairs = {
        tuple(sorted(pair))
        for group in first_week
        for pair in combinations(group, 2)
    }
    stats.generated += 1
    stats.observe_depth(player_count)

    def search_week(week: int, built_groups: list[tuple[int, ...]], available: frozenset[int]) -> Schedule | None:
        if node_limit > 0 and stats.expanded >= node_limit:
            raise SearchLimitExceeded(node_limit)
        stats.expanded += 1
        stats.observe_depth(week * player_count + len(built_groups) * group_size)
        stats.observe_frontier(week * groups + len(built_groups) + 1)
        if week == weeks:
            return tuple(partial)
        if len(built_groups) == groups:
            completed = tuple(built_groups)
            partial.append(completed)
            result = search_week(week + 1, [], frozenset(range(player_count)))
            if result is not None:
                return result
            partial.pop()
            return None

        anchor = min(available)
        others = sorted(available - {anchor})
        for selected in combinations(others, group_size - 1):
            candidate = (anchor, *selected)
            pairs = {tuple(sorted(pair)) for pair in combinations(candidate, 2)}
            stats.generated += 1
            if pairs & used_pairs:
                stats.pruned += 1
                continue
            used_pairs.update(pairs)
            result = search_week(
                week,
                [*built_groups, candidate],
                available - frozenset(candidate),
            )
            if result is not None:
                return result
            used_pairs.difference_update(pairs)
            stats.backtracks += 1
        return None

    try:
        solution = tuple(partial) if weeks == 1 else search_week(1, [], frozenset(range(player_count)))
        stats.finish()
        return SolverResult(
            solution,
            solution is not None,
            "Solved with DFS backtracking" if solution else "No schedule exists",
            stats,
        )
    except SearchLimitExceeded as error:
        stats.finish()
        return SolverResult(None, False, str(error), stats)


def maximize(
    groups: int,
    group_size: int,
    node_limit: int = 5_000_000,
) -> MaximumResult:
    """Find the maximum week count with descending DFS/backtracking.

    When a schedule reaches the pair-count upper bound, the returned solution
    itself is an optimality certificate. No saved schedule is read.
    """
    upper_bound = maximum_weeks_upper_bound(groups, group_size)
    if node_limit < 0:
        raise ValueError("node_limit cannot be negative")

    for weeks in range(upper_bound, 0, -1):
        result = solve(groups, group_size, weeks, node_limit)
        if result.solved:
            reaches_bound = weeks == upper_bound
            return MaximumResult(
                result.solution,
                weeks,
                upper_bound,
                True,
                (
                    f"Optimal: DFS generated a valid {weeks}-week schedule "
                    f"that reaches the upper bound of {upper_bound}"
                    if reaches_bound
                    else (
                        f"Optimal: descending DFS exhausted larger week counts "
                        f"and found a {weeks}-week schedule"
                    )
                ),
                result.stats,
            )
        if "node limit" in result.status.lower():
            return MaximumResult(
                None,
                0,
                upper_bound,
                False,
                f"Maximum not proven because {weeks}-week DFS exceeded its node limit",
                result.stats,
            )

    stats = SearchStats()
    stats.start()
    stats.finish()
    return MaximumResult(None, 0, upper_bound, True, "No positive-week schedule exists", stats)


def read_schedule(path: Path, expected_groups: int, expected_group_size: int) -> Schedule:
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    weeks: list[tuple[tuple[int, ...], ...]] = []
    for week_index, line in enumerate(lines, 1):
        raw_groups = line.split("|")
        if len(raw_groups) != expected_groups:
            raise ValueError(f"Week {week_index} must contain {expected_groups} groups")
        week = tuple(tuple(int(player.strip()) for player in group.split(",")) for group in raw_groups)
        if any(len(group) != expected_group_size for group in week):
            raise ValueError(f"Every group must contain {expected_group_size} players")
        weeks.append(week)
    return tuple(weeks)


def is_valid(schedule: Schedule | None, groups: int, group_size: int) -> bool:
    if schedule is None:
        return False
    player_count = groups * group_size
    meetings: set[tuple[int, int]] = set()
    for week in schedule:
        if len(week) != groups or any(len(group) != group_size for group in week):
            return False
        players = [player for group in week for player in group]
        if sorted(players) != list(range(player_count)):
            return False
        for group in week:
            for pair in combinations(group, 2):
                normalized = tuple(sorted(pair))
                if normalized in meetings:
                    return False
                meetings.add(normalized)
    return True


def objective(schedule: Schedule, player_count: int) -> int:
    meetings: dict[tuple[int, int], int] = {}
    for week in schedule:
        for group in week:
            for pair in combinations(group, 2):
                normalized = tuple(sorted(pair))
                meetings[normalized] = meetings.get(normalized, 0) + 1
    return sum(max(0, count - 1) for count in meetings.values())


def format_schedule(schedule: Schedule) -> str:
    lines: list[str] = []
    for week_index, week in enumerate(schedule, 1):
        lines.append(f"Week {week_index}:")
        for group_index, group in enumerate(week, 1):
            lines.append(f"  Group {group_index} = {list(group)}")
    return "\n".join(lines)
