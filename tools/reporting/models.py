from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class ChartSpec:
    file_id: str
    chart_type: Literal["bar", "line", "timeline", "scatter3d", "surface"] = "bar"
    title: Optional[str] = None
    name: Optional[str] = None
    label_column: Optional[str] = None
    value_columns: Optional[list] = field(default=None)
    time_column: Optional[str] = None
    series_column: Optional[str] = None
    value_column: Optional[str] = None
    x_column: Optional[str] = None
    y_column: Optional[str] = None
    z_column: Optional[str] = None
