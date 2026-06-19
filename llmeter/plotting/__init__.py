# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Utilities for building charts and visualizations of LLMeter results

These tools depend on [Plotly](https://plotly.com/python/), which you can either install separately
or via the `llmeter[plotting]` extra.
"""

from .defaults import DEFAULT_TEMPLATE, get_colorway
from .dynamic_load_test import (
    plot_dynamic_concurrency,
    plot_dynamic_latency,
    plot_dynamic_results,
    plot_dynamic_throughput,
)
from .percentage import percentage_point, percentage_points
from .plotting import (
    boxplot_by_dimension,
    color_sequences,
    histogram_by_dimension,
    plot_heatmap,
    plot_load_test_results,
    scatter_histogram_2d,
)

__all__ = [
    "DEFAULT_TEMPLATE",
    "boxplot_by_dimension",
    "color_sequences",
    "get_colorway",
    "histogram_by_dimension",
    "percentage_point",
    "percentage_points",
    "plot_dynamic_concurrency",
    "plot_dynamic_latency",
    "plot_dynamic_results",
    "plot_dynamic_throughput",
    "plot_heatmap",
    "plot_load_test_results",
    "scatter_histogram_2d",
]
