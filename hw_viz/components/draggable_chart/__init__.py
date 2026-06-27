"""A small dependency-free Streamlit component for dragging points on a chart.

The component renders an SVG line/scatter plot in which a handful of *anchor*
points can be dragged with the mouse or touch. It is deliberately generic so it
can drive both the initial-rate-curve editor (drag the rate of a tenor) and the
horizon-distribution editor (drag the rate value at a quantile).

The math (smoothing, distribution fitting) stays in Python: the component only
reports where anchors were dragged to and lets the server recompute the
authoritative state, which is then sent back on the next render.
"""

from __future__ import annotations

import os
from typing import Any

import streamlit.components.v1 as components

_COMPONENT_DIR = os.path.dirname(os.path.abspath(__file__))
_component_func = components.declare_component("hw_draggable_chart", path=_COMPONENT_DIR)


def draggable_chart(spec: dict[str, Any], key: str | None = None, default: Any = None) -> Any:
    """Render a draggable chart and return the latest drag event.

    Parameters
    ----------
    spec:
        Plot specification. See ``index.html`` for the full schema. Key fields:
        ``anchors`` (list of ``{id, x, y, label, color}``), ``background`` (list
        of polylines), ``xRange``/``yRange``, ``dragAxis`` (``"x"``, ``"y"`` or
        ``"xy"``), ``xType`` (``"linear"`` or ``"log"``), ``monotonicX`` and
        ``height``.
    key:
        Streamlit widget key; required when more than one chart is on the page.
    default:
        Value returned before the user has interacted with the chart.

    Returns
    -------
    The component value, a dict ``{"anchors": [{id, x, y}], "lastId": str,
    "event": int}`` once the user has dragged a point, otherwise ``default``.
    """

    return _component_func(spec=spec, key=key, default=default)
