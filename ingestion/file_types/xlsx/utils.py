"""Sheet -> raw grid -> table-island detection.

A workbook can hold multiple sheets, and a single sheet can hold multiple separate tables
(stacked with a blank row between them, or side-by-side with a blank column between them) -
common in exported finance/ops reports. detect_tables() finds those islands with two blank-run
passes (rows, then columns within each row-block) rather than true 2D connected-components -
simpler, and good enough for the common "clean rectangular blocks separated by blank
rows/cols" case. Known limitation: two side-by-side tables of very different row heights can
confuse the row pass (a blank-looking row for the shorter table isn't a real separator if the
taller table still has data in it) - _table_from_block() drops any resulting blank data rows
as a cheap mitigation, at the cost of also dropping genuine blank rows inside a real table.

Merged cells are forward-filled onto every cell in their range before island detection - Excel
(and openpyxl) only keeps a value on the merge's top-left cell and leaves the rest genuinely
empty, so a merged row-label column ("East" spanning two quarter rows, say) would otherwise
turn into real data with None in every row but the first. Note this is a deliberate trade-off,
not a free win: a merged *title* cell that horizontally spans what should be the blank-column
gap between two side-by-side tables will, once filled, make that gap look non-blank and can
cause the two tables to be detected as one. There's no heuristic that gets both cases right
every time; forward-fill was chosen because losing data silently is worse than an occasional
wrong split.
"""
import openpyxl
import pandas as pd

MIN_TABLE_ROWS = 2  # header + at least one data row
MIN_TABLE_COLS = 1


def load_sheets(file_path: str) -> dict:
    """Load every sheet as a raw 2D grid of cell values (list of lists), merged cells
    forward-filled. data_only=True reads formulas' last cached computed value, not the formula
    string - if the workbook was never opened/recalculated in Excel those cells come back None,
    same as any other missing value."""
    workbook = openpyxl.load_workbook(file_path, data_only=True)
    try:
        return {name: _sheet_grid(workbook[name]) for name in workbook.sheetnames}
    finally:
        workbook.close()


def _sheet_grid(worksheet) -> list:
    grid = [list(row) for row in worksheet.iter_rows(values_only=True)]
    for merged_range in worksheet.merged_cells.ranges:
        _fill_merged_range(grid, merged_range)
    return grid


def _fill_merged_range(grid: list, merged_range) -> None:
    if merged_range.min_row - 1 >= len(grid):
        return
    value = grid[merged_range.min_row - 1][merged_range.min_col - 1]
    for row in range(merged_range.min_row - 1, min(merged_range.max_row, len(grid))):
        row_values = grid[row]
        for col in range(merged_range.min_col - 1, min(merged_range.max_col, len(row_values))):
            row_values[col] = value


def detect_tables(grid: list) -> list:
    """Return every table island in one sheet's grid as {"dataframe", "anchor_row",
    "anchor_col"} dicts, in top-to-bottom, left-to-right order."""
    tables = []
    for row_start, row_end in _nonblank_ranges(grid):
        row_block = grid[row_start:row_end]
        columns_view = list(zip(*row_block)) if row_block else []
        for col_start, col_end in _nonblank_ranges(columns_view):
            block = [row[col_start:col_end] for row in row_block]
            table = _table_from_block(block, anchor_row=row_start, anchor_col=col_start)
            if table is not None:
                tables.append(table)
    return tables


def _nonblank_ranges(sequences: list) -> list:
    """Indices of contiguous non-blank runs in a list of cell sequences - rows for the row
    pass, transposed columns for the column pass. Same "is every cell in this sequence
    blank" check either way."""
    ranges = []
    start = None
    for i, seq in enumerate(sequences):
        blank = _is_blank_sequence(seq)
        if not blank and start is None:
            start = i
        elif blank and start is not None:
            ranges.append((start, i))
            start = None
    if start is not None:
        ranges.append((start, len(sequences)))
    return ranges


def _is_blank_sequence(seq) -> bool:
    return all(_is_blank_cell(cell) for cell in seq)


def _is_blank_cell(value) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def _table_from_block(block: list, anchor_row: int, anchor_col: int):
    if len(block) < MIN_TABLE_ROWS:
        return None

    header, *data_rows = block
    columns = _column_names(header)
    if len(columns) < MIN_TABLE_COLS:
        return None

    data_rows = [row for row in data_rows if not _is_blank_sequence(row)]
    if not data_rows:
        return None

    dataframe = pd.DataFrame(data_rows, columns=columns)
    return {"dataframe": dataframe, "anchor_row": anchor_row, "anchor_col": anchor_col}


def _column_names(header: list) -> list:
    seen: dict = {}
    names = []
    for i, cell in enumerate(header):
        base = str(cell).strip() if not _is_blank_cell(cell) else f"column_{i + 1}"
        if base in seen:
            seen[base] += 1
            base = f"{base}_{seen[base]}"
        else:
            seen[base] = 0
        names.append(base)
    return names
