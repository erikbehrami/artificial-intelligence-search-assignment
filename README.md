# Artificial Intelligence Search Assignment

Dependency-free Python 3 implementation of the first uninformed-search assignment.

Assignment source: [Artificial-Intelligence-2026 / Detyra1.md](https://github.com/adrianymeri/Artificial-Intelligence-2026/blob/main/Detyra1.md)

## 1. Social Golfers

### Problem model and representation

For `g` groups, `p` players per group, and `n = g × p` players, the decision problem asks whether a schedule for `w` weeks exists.

- Variable: `X[w, i]`, the group assigned to player `i` in week `w`.
- Domain: `{0, ..., g - 1}`.
- Code representation: an immutable sequence of weeks, groups, and player numbers.
- Hard constraints: every player appears exactly once each week; every group has exactly `p` players; every unordered player pair meets at most once.
- Symmetry breaking: the first week is fixed to consecutive canonical groups, and the smallest unassigned player anchors each new group.

### Objective function

The primary objective is to maximize the number of weeks `W` while satisfying all hard constraints. For a candidate schedule, the violation penalty is

`P = Σ(i<j) max(0, meetings(i,j) - 1)`.

A valid schedule has `P = 0`. Among valid schedules, the one with the largest `W` is optimal.

### Landscape calculation

Each player meets `p - 1` new players per week and can meet at most `n - 1` other players. Therefore:

`W × (p - 1) ≤ n - 1`, so `W ≤ floor((n - 1)/(p - 1))`.

For `g=8`, `p=4`, and `n=32`, the upper bound is `floor(31/3) = 10` weeks. The DFS constructor generates and validates a 10-week schedule, so it reaches the bound and proves that the maximum is exactly 10. The saved Aguado schedule is retained only as an independent regression fixture; it is not read by the solver.

The number of unordered ways to partition one week is

`Q = n! / ((p!)^g × g!)`.

For `8-4`, `Q = 59,287,247,761,257,140,625 ≈ 5.93 × 10^19`. With the first week fixed, a naive 10-week landscape still contains `Q^9 ≈ 9.05 × 10^177` schedules before pair constraints. The DFS solver reduces this with canonical first-week/group symmetry and immediate repeated-pair pruning.

### Search algorithm

`golfers.solve` performs depth-first search with backtracking for a requested week count, and `golfers.maximize` searches week counts from the mathematical upper bound downward.

The required `8-4-10` instance uses a `4-RGDD` of type `2^16` as symmetry breaking. Its 16 values represent pairs of golfers who never meet. For each of 40 blocks, DFS chooses one of eight player orientations. MRV selects the next block, repeated individual pairs are pruned immediately, and failed choices are undone by backtracking. The search generates the complete player schedule itself and never reads the saved reference schedule.

Design references: [MathWorld — Social Golfer Problem](https://mathworld.wolfram.com/SocialGolferProblem.html) and [Aguado — A 10 Days Solution](https://citeseerx.ist.psu.edu/document?doi=3e5e8faf3bdb5a5e797f11ac7a519f004a1ab16d&repid=rep1&type=pdf).

## 2. Sudoku

### Problem model and representation

- Variables: one variable `X[r,c]` for each of the 81 cells.
- Domain: `{1, ..., 9}`; a file value of `0` or `.` denotes an unassigned cell.
- Hard constraints: all values in each row, column, and `3 × 3` box are distinct; original clues cannot change.
- Code representation: an immutable row-major tuple of 81 integers.

Easy, Medium, and Hard input instances are included in `data/sudoku/`.

### Objective function

`F = number_of_blank_cells + 100 × number_of_duplicate_conflicts`.

A solved Sudoku has `F = 0`. The conflict weight makes every hard-constraint violation more expensive than filling another cell.

### Landscape calculation

If a puzzle has `b` blank cells, the unfiltered landscape has `9^b` assignments. The absolute full-board upper bound is `9^81 ≈ 1.97 × 10^77`. At search depth `d`, the branching factor is at most 9 and normally much smaller after row, column, and box filtering.

- BFS worst-case time and memory: `O(9^b)`.
- DFS backtracking worst-case time: `O(9^b)`; memory: `O(b)`.

Constraint propagation fills forced single-value cells before branching. DFS also uses the minimum-remaining-values rule. These do not change the worst-case bound, but greatly reduce the measured search tree.

### Search algorithms

`solve_bfs` uses a FIFO frontier and constraint propagation. `solve_backtracking` uses recursive DFS, constraint propagation, MRV variable selection, and backtracking. Both independently validate the returned board.

## 3. Latin Square

### Problem model and representation

For size `n`:

- Variables: `X[r,c]` for every cell in an `n × n` grid.
- Domain: `{1, ..., n}`.
- Hard constraints: every value occurs exactly once in each row and exactly once in each column.
- Code representation: an immutable tuple of rows.
- Symmetry breaking: the first row and first column are fixed to `1, ..., n`. Every Latin square can be relabeled into this reduced form, so a solution is not lost.

### Objective function

`F = number_of_blank_cells + 100 × (row_duplicates + column_duplicates)`.

A complete valid Latin square has `F = 0`.

### Landscape calculation

Without constraints, there are `n^(n²)` complete grids. After fixing the first row and column, `(n - 1)²` variables remain, giving the reduced upper bound

`n^((n - 1)²)`.

For `n=5`, this is `5^16 = 152,587,890,625` raw assignments. Legal-value filtering reduces the branching factor from at most `n` as rows and columns fill.

IDDFS repeats depth-limited DFS for limits from `2n-1` fixed cells through `n²`. Its worst-case time is exponential, while its active recursion memory is `O(n²)`.

### Search algorithm

`solve_iddfs` implements iterative deepening depth-first search. Each depth-limited iteration assigns legal values and restores cells during backtracking. The required `n=5` example is solved and independently validated.

## 4. Search Trees

### 8-Puzzle

A state is a tuple containing tiles `0..8`, where `0` is the blank. Edges move the blank up, down, left, or right. BFS expands states up to the requested depth, rejects repeated states, marks the shortest discovered goal path, and exports the tree in Graphviz DOT format.

### Missionaries and Cannibals

A state is `(missionaries_left, cannibals_left, boat_on_left)`. Edges carry one or two people. States are rejected whenever cannibals outnumber missionaries on a bank that contains missionaries. BFS exports the complete reachable-state tree and marks the 11-crossing solution.

### Rotating Workforce Scheduling

This is a genuine multi-employee coverage model:

- Variables: `X[e,d]`, the shift assigned to employee `e` on day `d`.
- Domain: configured shifts such as Day (`D`), Night (`N`), and Off (`O`).
- Hard constraints: one shift per employee/day; exact daily staffing coverage for every shift; maximum consecutive working days; minimum days off per employee; forbidden consecutive shift transitions; and cyclic constraints between the final and first day.
- Search representation: day-major assignment prefixes. Every tree edge assigns one employee/day variable.
- Search: BFS with immediate coverage and employee-rule pruning.

The example contains 3 employees, 3 days, and exact daily coverage `D:1, N:1, O:1`. Its raw landscape is `3^9 = 19,683` assignments. Exact daily coverage permits only `3! = 6` day patterns, so at most `6^3 = 216` complete schedules remain before transition and cyclic constraints.

## Search statistics

Every solver records generated nodes, expanded nodes, pruned states, backtracks, maximum frontier, maximum depth, and elapsed time. These are empirical measurements of a particular run; the formulas above describe the theoretical landscape.

## Run

From the repository directory:

```bash
python3 -m ai_assignment --help

python3 -m ai_assignment golfers-max
python3 -m ai_assignment golfers 3 2 5 5000000
python3 -m ai_assignment golfers-validate

python3 -m ai_assignment sudoku data/sudoku/easy.txt bfs
python3 -m ai_assignment sudoku data/sudoku/hard.txt backtracking
python3 -m ai_assignment latin 5 5000000

python3 -m ai_assignment tree-puzzle8 123456708 4 generated-trees/puzzle8.dot
python3 -m ai_assignment tree-missionaries generated-trees/missionaries.dot
python3 -m ai_assignment tree-workforce data/workforce/example.properties 3 generated-trees/workforce.dot

python3 -m ai_assignment demo
```

The tree commands create DOT files. Graphviz can render one with, for example, `dot -Tpng generated-trees/workforce.dot -o workforce.png`.

## Test

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

## Structure

```text
ai_assignment/
├── __main__.py
├── cli.py
├── common.py
├── golfers.py
├── latin.py
├── sudoku.py
└── trees.py

data/
tests/
README.md
```

Python 3.10 or newer is required. No external packages are needed.
