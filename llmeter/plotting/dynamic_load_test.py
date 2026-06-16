# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plotting functions for DynamicLoadTestResult visualizations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..utils import DeferredError

if not TYPE_CHECKING:
    try:
        import plotly.graph_objects as go
    except ImportError as e:
        go = DeferredError(e)
else:
    import plotly.graph_objects as go

if TYPE_CHECKING:
    from upath.types import WritablePathLike

    from llmeter.results import Result

    from ..experiments import ConcurrencyProfileInput

from ..utils import ensure_path


def plot_dynamic_concurrency(
    result: "Result",
    concurrency_log: list[tuple[float, int]],
    concurrency_profile: "ConcurrencyProfileInput",
    output_path: "WritablePathLike | None",
    show: bool = True,
    format: str = "html",
) -> "go.Figure":
    """Plot actual vs target concurrency over time.

    Creates a stepped line chart showing the actual concurrency from the
    controller log, overlaid with the target profile evaluated at the same
    timestamps.

    Args:
        result: The unified Result containing all InvocationResponse objects.
        concurrency_log: List of (elapsed_seconds, actual_active_clients) observations.
        concurrency_profile: The concurrency profile used during the test.
        output_path: Path where plots should be saved (if any).
        show: Whether to display the figure interactively. Defaults to True.
        format: Output format when saving (``"html"`` or ``"png"``).
            Defaults to ``"html"``.

    Returns:
        A Plotly :class:`~plotly.graph_objects.Figure`.

    Raises:
        ValueError: If there are zero successful responses in the result.
    """
    from llmeter.experiments import _resolve_concurrency

    # Validate that we have successful responses
    successful = [r for r in (result.responses or []) if r.error is None]
    if len(successful) == 0:
        raise ValueError(
            "Cannot plot concurrency: zero successful responses in the result."
        )

    # Extract actual concurrency from log
    timestamps = [entry[0] for entry in concurrency_log]
    actual_concurrency = [entry[1] for entry in concurrency_log]

    # Compute target concurrency at each log timestamp
    target_concurrency = [
        _resolve_concurrency(concurrency_profile, t, 0) for t in timestamps
    ]

    fig = go.Figure()

    # Actual concurrency as a stepped line
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=actual_concurrency,
            mode="lines",
            line={"shape": "hv"},
            name="Actual Concurrency",
        )
    )

    # Target profile overlay
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=target_concurrency,
            mode="lines",
            line={"dash": "dash"},
            name="Target Profile",
        )
    )

    fig.update_layout(
        title="Concurrency Over Time",
        xaxis_title="Elapsed Time (s)",
        yaxis_title="Concurrent Clients",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    # Save to output_path if configured
    if output_path is not None:
        save_path = ensure_path(output_path)
        if format == "html":
            fig.write_html(str(save_path / "concurrency_plot.html"))
        else:
            fig.write_image(str(save_path / f"concurrency_plot.{format}"))

    if show:
        fig.show()

    return fig


def plot_dynamic_throughput(
    result: "Result",
    output_path: "WritablePathLike | None",
    window_sec: float = 1.0,
    show: bool = True,
    format: str = "html",
) -> "go.Figure":
    """Plot requests per second over time as a line chart.

    Buckets successful responses by their ``request_time`` into windows of
    ``window_sec`` seconds and computes the throughput (requests/sec) for
    each window.

    Args:
        result: The unified Result containing all InvocationResponse objects.
        output_path: Path where plots should be saved (if any).
        window_sec: Window size in seconds for bucketing. Minimum 0.1.
            Defaults to 1.0.
        show: Whether to display the figure interactively. Defaults to True.
        format: Output format when saving (``"html"`` or ``"png"``).
            Defaults to ``"html"``.

    Returns:
        A plotly :class:`~plotly.graph_objects.Figure` with the throughput
        line chart.

    Raises:
        ValueError: If there are zero successful responses to plot.
    """
    # Clamp window_sec to minimum 0.1
    window_sec = max(window_sec, 0.1)

    # Filter successful responses (error is None)
    successful = [
        r
        for r in (result.responses or [])
        if r.error is None and r.request_time is not None
    ]

    if not successful:
        raise ValueError("No successful responses available to plot")

    # Sort by request_time
    successful.sort(key=lambda r: r.request_time)

    # Compute relative time from the first response
    first_time = successful[0].request_time
    relative_times = [(r.request_time - first_time).total_seconds() for r in successful]

    # Determine total time span and number of windows
    total_span = relative_times[-1]
    num_windows = max(1, int(total_span / window_sec) + 1)

    # Bucket responses into windows and compute requests/sec
    window_counts = [0] * num_windows
    for t in relative_times:
        bucket = int(t / window_sec)
        if bucket >= num_windows:
            bucket = num_windows - 1
        window_counts[bucket] += 1

    window_times = [i * window_sec for i in range(num_windows)]
    throughput = [count / window_sec for count in window_counts]

    # Create figure
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=window_times,
            y=throughput,
            mode="lines",
            name="Throughput (req/s)",
        )
    )
    fig.update_layout(
        title="Throughput Over Time",
        xaxis_title="Elapsed Time (s)",
        yaxis_title="Requests/sec",
    )

    # Save if output_path is configured
    if output_path is not None:
        save_path = ensure_path(output_path)
        if format == "html":
            fig.write_html(str(save_path / "throughput.html"))
        else:
            fig.write_image(str(save_path / f"throughput.{format}"))

    if show:
        fig.show()

    return fig


def plot_dynamic_latency(
    result: "Result",
    output_path: "WritablePathLike | None",
    window_sec: float = 1.0,
    show: bool = True,
    format: str = "html",
) -> "go.Figure":
    """Plot latency percentiles (p50, p95, p99) over time as a multi-line chart.

    Buckets successful responses by their ``request_time`` into windows of
    ``window_sec`` seconds and computes the p50, p95, and p99 of
    ``time_to_last_token`` for each window.

    Args:
        result: The unified Result containing all InvocationResponse objects.
        output_path: Path where plots should be saved (if any).
        window_sec: Window size in seconds for bucketing. Minimum 0.1.
            Defaults to 1.0.
        show: Whether to display the figure interactively. Defaults to True.
        format: Output format when saving (``"html"`` or ``"png"``).
            Defaults to ``"html"``.

    Returns:
        A plotly :class:`~plotly.graph_objects.Figure` with p50, p95, p99
        latency lines.

    Raises:
        ValueError: If there are zero successful responses to plot.
    """
    # Clamp window_sec to minimum 0.1
    window_sec = max(window_sec, 0.1)

    # Filter successful responses (error is None AND time_to_last_token is not None)
    successful = [
        r
        for r in (result.responses or [])
        if r.error is None
        and r.time_to_last_token is not None
        and r.request_time is not None
    ]

    if not successful:
        raise ValueError("No successful responses available to plot")

    # Sort by request_time
    successful.sort(key=lambda r: r.request_time)

    # Compute relative time from the first response
    first_time = successful[0].request_time
    relative_times = [(r.request_time - first_time).total_seconds() for r in successful]

    # Determine total time span and number of windows
    total_span = relative_times[-1]
    num_windows = max(1, int(total_span / window_sec) + 1)

    # Bucket responses into windows
    buckets: list[list[float]] = [[] for _ in range(num_windows)]
    for i, t in enumerate(relative_times):
        bucket = int(t / window_sec)
        if bucket >= num_windows:
            bucket = num_windows - 1
        buckets[bucket].append(successful[i].time_to_last_token)

    # Compute percentiles per window
    window_times: list[float] = []
    p50_values: list[float] = []
    p95_values: list[float] = []
    p99_values: list[float] = []

    for i, values in enumerate(buckets):
        if not values:
            continue
        sorted_values = sorted(values)
        n = len(sorted_values)
        p50 = sorted_values[int(n * 0.5)]
        p95 = sorted_values[int(n * 0.95)]
        p99 = sorted_values[int(n * 0.99)]

        window_times.append(i * window_sec)
        p50_values.append(p50)
        p95_values.append(p95)
        p99_values.append(p99)

    # Create figure with three lines
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=window_times,
            y=p50_values,
            mode="lines",
            name="p50",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=window_times,
            y=p95_values,
            mode="lines",
            name="p95",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=window_times,
            y=p99_values,
            mode="lines",
            name="p99",
        )
    )
    fig.update_layout(
        title="Latency Percentiles Over Time",
        xaxis_title="Elapsed Time (s)",
        yaxis_title="Time to Last Token (s)",
    )

    # Save if output_path is configured
    if output_path is not None:
        save_path = ensure_path(output_path)
        if format == "html":
            fig.write_html(str(save_path / "latency.html"))
        else:
            fig.write_image(str(save_path / f"latency.{format}"))

    if show:
        fig.show()

    return fig


def plot_dynamic_results(
    result: "Result",
    concurrency_log: list[tuple[float, int]],
    concurrency_profile: "ConcurrencyProfileInput",
    output_path: "WritablePathLike | None",
    show: bool = True,
    format: str = "html",
) -> "dict[str, go.Figure]":
    """Plot all standard charts for a dynamic load test.

    Convenience function that calls :func:`plot_dynamic_concurrency`,
    :func:`plot_dynamic_throughput`, and :func:`plot_dynamic_latency`,
    returning all figures in a single dictionary.

    Args:
        result: The unified Result containing all InvocationResponse objects.
        concurrency_log: List of (elapsed_seconds, actual_active_clients) observations.
        concurrency_profile: The concurrency profile used during the test.
        output_path: Path where plots should be saved (if any).
        show: Whether to display figures interactively. Defaults to True.
        format: Output format when saving (``"html"`` or ``"png"``).
            Defaults to ``"html"``.

    Returns:
        A dict mapping ``"concurrency"``, ``"throughput"``, and
        ``"latency"`` to their respective
        :class:`~plotly.graph_objects.Figure` instances.
    """
    return {
        "concurrency": plot_dynamic_concurrency(
            result=result,
            concurrency_log=concurrency_log,
            concurrency_profile=concurrency_profile,
            output_path=output_path,
            show=show,
            format=format,
        ),
        "throughput": plot_dynamic_throughput(
            result=result,
            output_path=output_path,
            show=show,
            format=format,
        ),
        "latency": plot_dynamic_latency(
            result=result,
            output_path=output_path,
            show=show,
            format=format,
        ),
    }
