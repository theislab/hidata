from __future__ import annotations

import io
import sys

from rich.box import DOUBLE
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

HIGHLIGHT_COLOR = "bold deep_pink2"
_SLOTS = ("obs", "var", "uns", "obsm", "varm", "layers", "obsp", "varp")
_HTML_FORMAT = (
    "<pre style=\"font-family: Menlo, 'DejaVu Sans Mono', consolas, monospace; line-height: 1.2\">{code}</pre>"
)


def _slot_lines(container, slots, highlight_keys, labels=None) -> list[Text]:
    labels = labels or {}
    lines = []
    for slot in slots:
        mapping = getattr(container, slot, None)
        if mapping is None:
            continue
        # anndata 0.13 lists X as layer None
        keys = [k for k in (mapping.columns if slot in ("obs", "var") else mapping.keys()) if k is not None]
        if not keys:
            continue
        line = Text(f"    {labels.get(slot, slot)}: ")
        for i, key in enumerate(keys):
            style = HIGHLIGHT_COLOR if slot in ("obs", "obsm") and key in highlight_keys else None
            line.append(f"'{key}'", style)
            if i < len(keys) - 1:
                line.append(", ")
        lines.append(line)
    return lines


def side_lines(container, kind: str, highlight_keys, unit: str = "groups") -> list[Text]:
    """Lines describing one level of a HiData."""
    if kind == "table":
        header = f"DataFrame with {container.n_obs:,} {unit} × {container.obs.shape[1]:,} columns"
        slots = ("obs", "obsm", "obsp")
    else:
        name = type(container).__name__
        header = f"{name} object with n_obs × n_vars = {container.n_obs:,} × {container.n_vars:,}"
        if getattr(container, "isbacked", False):
            header += f" backed at {str(container.filename)!r}"
        slots = _SLOTS
    if container.is_view:
        header = "View of " + header
    lines = [Text(header)]
    if kind == "mudata":
        lines.append(Text("    mod: " + ", ".join(f"'{m}'" for m in container.mod)))
    if _is_ehr(container):
        lines.append(Text(f"    tem: {container.n_t:,} timepoints"))
    labels = {"obs": "columns"} if kind == "table" else None
    lines.extend(_slot_lines(container, slots, highlight_keys, labels))
    return lines


def _is_ehr(container) -> bool:
    return type(container).__name__ == "EHRData" and hasattr(container, "n_t")


def _kind(container, is_table: bool) -> str:
    if is_table:
        return "table"
    return "mudata" if type(container).__name__ == "MuData" else "anndata"


def render(hd, html: bool = False) -> str:
    """Render a HiData as a side by side panel of its levels, as text or as HTML."""
    columns = [("G (groups)", side_lines(hd._G, _kind(hd._G, hd._G_is_table), set(hd._var_dims_to_sync)))]
    if hd._S is not None:
        per_group = hd.n_subgroups_per_group
        header = f"S (subgroups, {per_group.min():,}..{per_group.max():,} per group)"
        columns.append((header, side_lines(hd._S, _kind(hd._S, hd._S_is_table), {hd.group_id}, unit="subgroups")))
        counts = hd.n_cells_per_subgroup
        unit = "subgroup"
        cell_keys = {hd.subgroup_id}
    else:
        counts = hd.n_cells_per_group
        unit = "group"
        cell_keys = {hd.group_id}
    cell_header = f"C (cells, {counts.min():,}..{counts.max():,} per {unit})"
    columns.append((cell_header, side_lines(hd.C, _kind(hd.C, False), cell_keys)))

    n_rows = max(len(lines) for _, lines in columns)
    width = 50 if len(columns) == 2 else 40
    table = Table(show_header=True, header_style=HIGHLIGHT_COLOR)
    for header, lines in columns:
        lines += [Text("")] * (n_rows - len(lines))
        table.add_column(header, max_width=width, justify="left")
    for row in zip(*(lines for _, lines in columns), strict=True):
        table.add_row(*row)

    subgroups = "" if hd._S is None else f"n_subgroups={hd.n_subgroups:,}, "
    title = Text(f"HiData(n_groups={hd.n_groups:,}, {subgroups}n_cells={hd.n_cells:,})", style=HIGHLIGHT_COLOR)
    keys = f"group_id='{hd.group_id}'" + ("" if hd._S is None else f", subgroup_id='{hd.subgroup_id}'")
    panel = Panel(
        table, title=title, title_align="left", subtitle=Text(keys), subtitle_align="left", box=DOUBLE, expand=False
    )
    if html:
        console = Console(record=True, width=140, force_jupyter=False, file=io.StringIO())
        console.print(panel)
        return console.export_html(inline_styles=True, code_format=_HTML_FORMAT)
    console = Console(force_jupyter=False, color_system=None if "ipykernel" in sys.modules else "auto")
    with console.capture() as capture:
        console.print(panel)
    return capture.get()
