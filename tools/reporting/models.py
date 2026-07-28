from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class ChartSpec:
    """One chart in a dashboard. Only file_id, chart_type, and column NAMES are ever passed
    here - never actual data values, and never a path. The real numbers are read straight from
    the parquet file identified by file_id when the dashboard is built.

    Column names needed per chart_type:
    - "bar" / "line": EITHER label_column + value_columns (1+ numeric series - if omitted, the
      first non-numeric column and up to 5 numeric columns are used automatically) OR, when the
      data has TWO categorical dimensions and one metric (long/tidy format, e.g. columns Age,
      Gender, Customer Count), label_column + series_column + value_column - this produces one
      bar/line per distinct series_column value, grouped along label_column (e.g.
      label_column="Age", series_column="Gender", value_column="Customer Count" -> separate
      Male/Female series across ages). Use this whenever the result has more than one grouping
      column - do NOT just pick one grouping column and silently drop the other.
    - "timeline": time_column (required) plus EITHER value_columns (wide data - one series per
      column) OR series_column + value_column (long/tidy data - one series per distinct value
      in series_column, e.g. columns date, job_title, count -> series_column="job_title",
      value_column="count").
    - "scatter3d" / "surface": x_column, y_column, z_column (all required). "surface" needs
      every (x, y) combination present in the data to build a valid grid - use "scatter3d"
      instead if that can't be guaranteed.
    """

    file_id: str
    chart_type: Literal["bar", "line", "timeline", "scatter3d", "surface"] = "bar"
    title: Optional[str] = None
    # Only meaningful for persistent/refreshable dashboards (see OrchestratorTools.
    # generate_dashboard / ReportingTools.generate_realtime_dashboard_bundle).
    # Stable identifier this chart is saved under inside the dashboard's transform_script
    # (save(df, name=...)) - matched by prefix against the sandbox's fresh `saved` list on
    # every refresh, since save() appends a random suffix to the actual path each run.
    # Falls back to a slug of title/index if omitted.
    name: Optional[str] = None
    label_column: Optional[str] = None
    value_columns: Optional[list] = field(default=None)
    time_column: Optional[str] = None
    series_column: Optional[str] = None
    value_column: Optional[str] = None
    x_column: Optional[str] = None
    y_column: Optional[str] = None
    z_column: Optional[str] = None
