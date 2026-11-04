from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

from ai_assignment.common import DotGraph


@dataclass(frozen=True)
class TreeResult:
    nodes: int
    goal_found: bool
    solution_actions: tuple[str, ...]
    output: Path


@dataclass(frozen=True)
class _TreeNode:
    node_id: int
    state: tuple[int, ...]
    parent: int
    action: str
    depth: int


PUZZLE_GOAL = (1, 2, 3, 4, 5, 6, 7, 8, 0)
PUZZLE_MOVES = ((-1, 0, "Up"), (1, 0, "Down"), (0, -1, "Left"), (0, 1, "Right"))


def parse_puzzle(value: str) -> tuple[int, ...]:
    digits = tuple(int(char) for char in value if char.isdigit())
    if len(digits) != 9 or set(digits) != set(range(9)):
        raise ValueError("8-Puzzle state must contain every value 0..8 exactly once")
    return digits


def puzzle_is_solvable(state: tuple[int, ...]) -> bool:
    inversions = sum(
        left > right
        for index, left in enumerate(state)
        if left
        for right in state[index + 1 :]
        if right
    )
    return inversions % 2 == 0


def generate_puzzle_tree(start: tuple[int, ...], max_depth: int, output: Path) -> TreeResult:
    if set(start) != set(range(9)) or len(start) != 9:
        raise ValueError("8-Puzzle state must contain every value 0..8 exactly once")
    if max_depth < 0:
        raise ValueError("max_depth cannot be negative")
    if not puzzle_is_solvable(start):
        raise ValueError("This 8-Puzzle state is unsolvable")
    nodes = [_TreeNode(0, start, -1, "Start", 0)]
    edges: list[tuple[int, int, str]] = []
    frontier = deque([nodes[0]])
    visited = {start}
    goal_id = 0 if start == PUZZLE_GOAL else None
    while frontier:
        current = frontier.popleft()
        if current.depth == max_depth:
            continue
        blank = current.state.index(0)
        row, column = divmod(blank, 3)
        for dr, dc, action in PUZZLE_MOVES:
            next_row, next_column = row + dr, column + dc
            if not (0 <= next_row < 3 and 0 <= next_column < 3):
                continue
            target = next_row * 3 + next_column
            child = list(current.state)
            child[blank], child[target] = child[target], child[blank]
            state = tuple(child)
            if state in visited:
                continue
            visited.add(state)
            node = _TreeNode(len(nodes), state, current.node_id, action, current.depth + 1)
            nodes.append(node)
            edges.append((current.node_id, node.node_id, action))
            frontier.append(node)
            if goal_id is None and state == PUZZLE_GOAL:
                goal_id = node.node_id
    path = _path_ids(goal_id, nodes)
    graph = DotGraph("Puzzle8")
    for node in nodes:
        label = "\n".join(
            " ".join("_" if value == 0 else str(value) for value in node.state[row : row + 3])
            for row in (0, 3, 6)
        )
        graph.add_node(
            node.node_id,
            label,
            'style=filled, fillcolor="palegreen"' if node.node_id in path else "",
        )
    for source, target, action in edges:
        selected = source in path and target in path
        graph.add_edge(source, target, action, 'color="darkgreen", penwidth=2' if selected else "")
    graph.write(output)
    return TreeResult(len(nodes), goal_id is not None, _path_actions(goal_id, nodes), output)


MCState = tuple[int, int, bool]


def generate_missionaries_tree(output: Path) -> TreeResult:
    start: MCState = (3, 3, True)
    goal: MCState = (0, 0, False)
    loads = ((1, 0), (2, 0), (0, 1), (0, 2), (1, 1))
    nodes = [_TreeNode(0, start, -1, "Start", 0)]
    edges: list[tuple[int, int, str]] = []
    frontier = deque([nodes[0]])
    visited = {start}
    goal_id: int | None = None
    while frontier:
        current = frontier.popleft()
        missionaries, cannibals, boat_left = current.state
        if current.state == goal and goal_id is None:
            goal_id = current.node_id
        for moved_m, moved_c in loads:
            direction = -1 if boat_left else 1
            state: MCState = (
                missionaries + direction * moved_m,
                cannibals + direction * moved_c,
                not boat_left,
            )
            if not _mc_safe(state) or state in visited:
                continue
            visited.add(state)
            action = f"{'L→R' if boat_left else 'R→L'} {moved_m}M,{moved_c}C"
            node = _TreeNode(len(nodes), state, current.node_id, action, current.depth + 1)
            nodes.append(node)
            edges.append((current.node_id, node.node_id, action))
            frontier.append(node)
    path = _path_ids(goal_id, nodes)
    graph = DotGraph("MissionariesCannibals")
    for node in nodes:
        missionaries, cannibals, boat_left = node.state
        label = (
            f"Left: {missionaries}M,{cannibals}C\n"
            f"Boat: {'L' if boat_left else 'R'}\n"
            f"Right: {3 - missionaries}M,{3 - cannibals}C"
        )
        graph.add_node(
            node.node_id,
            label,
            'style=filled, fillcolor="palegreen"' if node.node_id in path else "",
        )
    for source, target, action in edges:
        graph.add_edge(
            source,
            target,
            action,
            'color="darkgreen", penwidth=2' if source in path and target in path else "",
        )
    graph.write(output)
    return TreeResult(len(nodes), goal_id is not None, _path_actions(goal_id, nodes), output)


def _mc_safe(state: MCState) -> bool:
    missionaries, cannibals, _ = state
    if not (0 <= missionaries <= 3 and 0 <= cannibals <= 3):
        return False
    right_m, right_c = 3 - missionaries, 3 - cannibals
    return (missionaries == 0 or missionaries >= cannibals) and (right_m == 0 or right_m >= right_c)


@dataclass(frozen=True)
class WorkforceConfig:
    days: int
    shifts: tuple[str, ...]
    max_consecutive_work: int
    minimum_days_off: int
    forbidden_transitions: frozenset[str]
    employees: int = 1
    coverage: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        if self.days < 1 or self.employees < 1:
            raise ValueError("Workforce days and employees must be positive")
        if not self.shifts or len(set(self.shifts)) != len(self.shifts):
            raise ValueError("Workforce shifts must be unique and non-empty")
        if "O" not in self.shifts:
            raise ValueError("Shifts must contain O (off)")
        if self.max_consecutive_work < 1 or not 0 <= self.minimum_days_off <= self.days:
            raise ValueError("Invalid workforce limits")
        coverage = dict(self.coverage)
        if len(coverage) != len(self.coverage):
            raise ValueError("Coverage contains a duplicate shift")
        if coverage:
            if set(coverage) != set(self.shifts):
                raise ValueError("Coverage must specify every shift, including O")
            if any(amount < 0 for amount in coverage.values()):
                raise ValueError("Coverage values cannot be negative")
            if sum(coverage.values()) != self.employees:
                raise ValueError("Daily coverage must add up to the employee count")
        for transition in self.forbidden_transitions:
            source, separator, target = transition.partition(">")
            if not separator or source not in self.shifts or target not in self.shifts:
                raise ValueError(f"Invalid forbidden transition: {transition}")

    @classmethod
    def read(cls, path: Path) -> WorkforceConfig:
        properties: dict[str, str] = {}
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            key, separator, value = line.partition("=")
            if not separator:
                raise ValueError(f"Invalid workforce property: {raw_line}")
            properties[key.strip()] = value.strip()
        required = {
            "days",
            "employees",
            "shifts",
            "coverage",
            "maxConsecutiveWork",
            "minimumDaysOff",
        }
        missing = required - properties.keys()
        if missing:
            raise ValueError(f"Missing workforce properties: {', '.join(sorted(missing))}")

        coverage: list[tuple[str, int]] = []
        for entry in properties["coverage"].split(","):
            shift, separator, amount = entry.strip().partition(":")
            if not separator:
                raise ValueError(f"Invalid coverage entry: {entry}")
            coverage.append((shift.strip(), int(amount.strip())))

        return cls(
            days=int(properties["days"]),
            shifts=tuple(value.strip() for value in properties["shifts"].split(",") if value.strip()),
            max_consecutive_work=int(properties["maxConsecutiveWork"]),
            minimum_days_off=int(properties["minimumDaysOff"]),
            forbidden_transitions=frozenset(
                value.strip()
                for value in properties.get("forbiddenTransitions", "").split(",")
                if value.strip()
            ),
            employees=int(properties["employees"]),
            coverage=tuple(coverage),
        )

    def transition_allowed(self, source: str, target: str) -> bool:
        return f"{source}>{target}" not in self.forbidden_transitions

    def required(self, shift: str) -> int:
        if not self.coverage:
            return self.employees if shift == "O" else 0
        return dict(self.coverage)[shift]


WorkforceSchedule = tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class WorkforceResult:
    nodes: int
    complete_schedule_found: bool
    schedule: WorkforceSchedule
    output: Path


@dataclass(frozen=True)
class _WorkNode:
    node_id: int
    assignments: tuple[str, ...]
    parent: int


def generate_workforce_tree(config: WorkforceConfig, requested_depth: int, output: Path) -> WorkforceResult:
    """Generate a BFS tree by assigning one employee/day variable per edge."""
    max_days = min(requested_depth, config.days)
    if max_days < 0:
        raise ValueError("Depth cannot be negative")
    assignment_limit = max_days * config.employees
    nodes = [_WorkNode(0, (), -1)]
    edges: list[tuple[int, int, str]] = []
    frontier = deque([nodes[0]])
    goal_id: int | None = None

    while frontier:
        current = frontier.popleft()
        if len(current.assignments) == config.days * config.employees:
            schedule = _assignments_to_schedule(current.assignments, config)
            if workforce_valid_complete(schedule, config) and goal_id is None:
                goal_id = current.node_id
            continue
        if len(current.assignments) == assignment_limit:
            continue

        variable = len(current.assignments)
        day, employee = divmod(variable, config.employees)
        for shift in config.shifts:
            assignments = (*current.assignments, shift)
            if not workforce_valid_partial(assignments, config):
                continue
            node = _WorkNode(len(nodes), assignments, current.node_id)
            nodes.append(node)
            edges.append(
                (
                    current.node_id,
                    node.node_id,
                    f"Day {day + 1}, Employee {employee + 1} = {shift}",
                )
            )
            frontier.append(node)

    path = _work_path_ids(goal_id, nodes)
    graph = DotGraph("RotatingWorkforce")
    for node in nodes:
        graph.add_node(
            node.node_id,
            _workforce_node_label(node.assignments, config),
            'style=filled, fillcolor="palegreen"' if node.node_id in path else "",
        )
    for source, target, action in edges:
        graph.add_edge(
            source,
            target,
            action,
            'color="darkgreen", penwidth=2' if source in path and target in path else "",
        )
    graph.write(output)
    schedule = (
        _assignments_to_schedule(nodes[goal_id].assignments, config)
        if goal_id is not None
        else ()
    )
    return WorkforceResult(len(nodes), goal_id is not None, schedule, output)


def workforce_valid_partial(assignments: tuple[str, ...], config: WorkforceConfig) -> bool:
    """Validate a day-major prefix of employee shift assignments."""
    total_variables = config.days * config.employees
    if len(assignments) > total_variables or any(shift not in config.shifts for shift in assignments):
        return False

    coverage = dict(config.coverage)
    if coverage:
        for day in range(config.days):
            start = day * config.employees
            assigned = assignments[start : min(start + config.employees, len(assignments))]
            if not assigned:
                break
            remaining_slots = config.employees - len(assigned)
            for shift in config.shifts:
                count = assigned.count(shift)
                if count > coverage[shift] or count + remaining_slots < coverage[shift]:
                    return False
            if len(assigned) == config.employees and any(
                assigned.count(shift) != coverage[shift] for shift in config.shifts
            ):
                return False

    for employee in range(config.employees):
        sequence = tuple(
            assignments[day * config.employees + employee]
            for day in range(config.days)
            if day * config.employees + employee < len(assignments)
        )
        consecutive = 0
        for index, shift in enumerate(sequence):
            if index and not config.transition_allowed(sequence[index - 1], shift):
                return False
            if shift == "O":
                consecutive = 0
            else:
                consecutive += 1
                if consecutive > config.max_consecutive_work:
                    return False
        remaining_days = config.days - len(sequence)
        if sequence.count("O") + remaining_days < config.minimum_days_off:
            return False
    return True


def workforce_valid_complete(schedule: WorkforceSchedule, config: WorkforceConfig) -> bool:
    if len(schedule) != config.employees or any(len(row) != config.days for row in schedule):
        return False
    assignments = tuple(
        schedule[employee][day]
        for day in range(config.days)
        for employee in range(config.employees)
    )
    if len(assignments) != config.days * config.employees:
        return False
    if not workforce_valid_partial(assignments, config):
        return False

    coverage = dict(config.coverage)
    if coverage:
        for day in range(config.days):
            if any(
                sum(schedule[employee][day] == shift for employee in range(config.employees))
                != coverage[shift]
                for shift in config.shifts
            ):
                return False

    for employee_schedule in schedule:
        if employee_schedule.count("O") < config.minimum_days_off:
            return False
        if not config.transition_allowed(employee_schedule[-1], employee_schedule[0]):
            return False
        if not _cyclic_run_valid(employee_schedule, config.max_consecutive_work):
            return False
    return True


def _assignments_to_schedule(
    assignments: tuple[str, ...], config: WorkforceConfig
) -> WorkforceSchedule:
    if len(assignments) != config.days * config.employees:
        raise ValueError("A complete assignment is required")
    return tuple(
        tuple(assignments[day * config.employees + employee] for day in range(config.days))
        for employee in range(config.employees)
    )


def _workforce_node_label(assignments: tuple[str, ...], config: WorkforceConfig) -> str:
    if not assignments:
        return "Empty schedule"
    rows: list[str] = []
    for employee in range(config.employees):
        shifts = [
            assignments[day * config.employees + employee]
            for day in range(config.days)
            if day * config.employees + employee < len(assignments)
        ]
        rows.append(f"E{employee + 1}: {' '.join(shifts)}")
    return "\n".join(rows)


def _cyclic_run_valid(shifts: tuple[str, ...], maximum: int) -> bool:
    if "O" not in shifts:
        return len(shifts) <= maximum
    off = shifts.index("O")
    run = 0
    for offset in range(1, len(shifts) + 1):
        if shifts[(off + offset) % len(shifts)] == "O":
            run = 0
        else:
            run += 1
            if run > maximum:
                return False
    return True

def _path_ids(goal_id: int | None, nodes: list[_TreeNode]) -> set[int]:
    path: set[int] = set()
    current = goal_id
    while current is not None and current >= 0:
        path.add(current)
        parent = nodes[current].parent
        current = parent if parent >= 0 else None
    return path


def _path_actions(goal_id: int | None, nodes: list[_TreeNode]) -> tuple[str, ...]:
    actions: list[str] = []
    current = goal_id
    while current is not None and nodes[current].parent >= 0:
        actions.append(nodes[current].action)
        current = nodes[current].parent
    return tuple(reversed(actions))


def _work_path_ids(goal_id: int | None, nodes: list[_WorkNode]) -> set[int]:
    path: set[int] = set()
    current = goal_id
    while current is not None and current >= 0:
        path.add(current)
        parent = nodes[current].parent
        current = parent if parent >= 0 else None
    return path
