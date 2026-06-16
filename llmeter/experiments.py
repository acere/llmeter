# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Higher-level experiments (generally combining multiple Runs)

This module provides utilities to run more complex "experiments" that go beyond the scope of a
single [Run][llmeter.runner.Runner].
"""

import asyncio
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from math import ceil
from typing import Callable, Literal

from tqdm.auto import tqdm
from upath import UPath as Path
from upath.types import ReadablePathLike, WritablePathLike

from .callbacks.base import Callback
from .endpoints.base import Endpoint, InvocationResponse
from .plotting import (
    color_sequences,
    plot_dynamic_concurrency,
    plot_dynamic_latency,
    plot_dynamic_results,
    plot_dynamic_throughput,
    plot_heatmap,
    plot_load_test_results,
)
from .prompt_utils import CreatePromptCollection
from .results import Result
from .runner import Runner, _ClientHandle, _client_loop

from .tokenizers import Tokenizer
from .utils import ensure_path

# Type aliases for dynamic concurrency profiles
Waypoint = tuple[float, int]
"""A (time_seconds, client_count) tuple representing desired concurrency at a point in time."""

ConcurrencyProfileInput = Callable[[float], int] | list[Waypoint]
"""Concurrency profile: either a callable f(t) -> clients, or a list of discrete (time, clients) steps."""

MAX_CLIENTS = 1000
"""Maximum number of concurrent clients allowed."""

logger = logging.getLogger(__name__)

# using a custom env variable because the TQDM one (https://github.com/tqdm/tqdm/issues/612#issuecomment-2015702344) doesn't work reliably
_disable_tqdm = False
if os.getenv("LLMETER_DISABLE_ALL_PROGRESS_BARS") == "1":
    logger.info("Disabling tqdm progress bars")
    _disable_tqdm = True


@dataclass
class LoadTestResult:
    results: dict[int, Result]
    test_name: str
    output_path: WritablePathLike | None = None

    def plot_results(self, show: bool = True, format: Literal["html", "png"] = "html"):
        figs = plot_load_test_results(self)

        # add individual color sequence for each plot
        c_seqs = [
            color_sequences.Bluered,
            color_sequences.Turbo,
            color_sequences.Sunsetdark_r,
            color_sequences.Blackbody,
            color_sequences.Viridis,
            color_sequences.Plasma,
        ]

        for i, (_, f) in enumerate(figs.items()):
            f.update_layout(colorway=c_seqs[i % len(c_seqs)])

        if self.output_path is not None:
            output_path = ensure_path(self.output_path)
            # save figure to the output path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            for k, f in figs.items():
                if format == "html":
                    f.write_html(output_path / f"{k}.{format}")
                else:
                    f.write_image(output_path / f"{k}.{format}")

        if show:
            [f.show() for _, f in figs.items()]
        return figs

    @classmethod
    def load(
        cls,
        load_path: ReadablePathLike | None,
        test_name: str | None = None,
        load_responses: bool = True,
    ) -> "LoadTestResult":
        """Load test results from a directory.

        Args:
            load_path: Directory path containing the load test results subdirectories
            test_name: Optional name for the test. If not provided, will use the directory name
            load_responses: Whether to load individual invocation responses. Defaults to True.
                When False, only summaries and pre-computed stats are loaded.

        Returns:
            LoadTestResult: A LoadTestResult object containing the loaded results

        Raises:
            FileNotFoundError: If load_path does not exist or is None/empty
            ValueError: If no results are found in the directory
        """
        if not load_path:
            raise FileNotFoundError("Load path cannot be None or empty")

        if not isinstance(load_path, Path):
            load_path = ensure_path(load_path)

        if not load_path.exists():
            raise FileNotFoundError(f"Load path {load_path} does not exist")

        results = [
            Result.load(x, load_responses=load_responses)
            for x in load_path.iterdir()
            if x.is_dir()
        ]

        if not results:
            raise ValueError(f"No results found in {load_path}")

        return LoadTestResult(
            results={r.clients: r for r in results},
            test_name=test_name or load_path.name,
            output_path=load_path.parent,
        )


@dataclass
class LoadTest:
    """Experiment to explore how performance changes at different concurrency levels.

    This experiment creates a series of Runs with different levels of concurrency, defined by
    ``sequence_of_clients``, and runs them one after the other.

    By default, each run sends a fixed number of requests (count-bound). Set ``run_duration``
    to run each concurrency level for a fixed number of seconds instead (time-bound), which
    gives a more realistic picture of sustained throughput.

    Attributes:
        endpoint (Endpoint): The LLM endpoint to test.
        payload (dict | list[dict]): The request payload(s) to send.
        sequence_of_clients (list[int]): Concurrency levels to test.
        min_requests_per_client (int): Minimum requests per client in count-bound mode.
        min_requests_per_run (int): Minimum total requests per run in count-bound mode.
        run_duration (int | float | None): When set, each concurrency level runs for this
            many seconds instead of a fixed request count. Mutually exclusive with
            ``min_requests_per_client`` / ``min_requests_per_run``.
        low_memory (bool): When ``True``, responses are written to disk but not kept in
            memory. Requires ``output_path``. Defaults to ``False``.
        progress_bar_stats (dict | None): Controls which live stats appear on the progress
            bar. See ``DEFAULT_DISPLAY_STATS`` in ``llmeter.live_display`` for the default.
        output_path (os.PathLike | str | None): Where to save results.
        tokenizer (Tokenizer | None): Optional tokenizer for token counting.
        test_name (str | None): Name for this test. Defaults to current date/time.
        callbacks (list[Callback] | None): Optional callbacks.

    Example::

        # Count-bound: 10 requests per client at each concurrency level
        load_test = LoadTest(
            endpoint=my_endpoint,
            payload=sample_payload,
            sequence_of_clients=[1, 5, 10, 20],
            min_requests_per_client=10,
            output_path="outputs/load_test",
        )
        result = await load_test.run()
        result.plot_results()

        # Time-bound: 60 seconds per concurrency level
        load_test = LoadTest(
            endpoint=my_endpoint,
            payload=sample_payload,
            sequence_of_clients=[1, 5, 10, 20],
            run_duration=60,
            output_path="outputs/load_test",
        )
        result = await load_test.run()

        # Time-bound with low-memory mode for large-scale tests
        load_test = LoadTest(
            endpoint=my_endpoint,
            payload=sample_payload,
            sequence_of_clients=[1, 5, 10, 20, 50],
            run_duration=120,
            low_memory=True,
            output_path="outputs/large_load_test",
        )
        result = await load_test.run()
    """

    endpoint: Endpoint
    payload: dict | list[dict]
    sequence_of_clients: list[int]
    min_requests_per_client: int = 1
    min_requests_per_run: int = 10
    run_duration: int | float | timedelta | None = None
    low_memory: bool = False
    progress_bar_stats: dict[str, str | tuple[str, str]] | None = None
    output_path: WritablePathLike | None = None
    tokenizer: Tokenizer | None = None
    test_name: str | None = None
    callbacks: list[Callback] | None = None

    def __post_init__(self) -> None:
        self._test_name = self.test_name or f"{datetime.now():%Y%m%d-%H%M}"

    def _get_n_requests(self, clients):
        if clients * self.min_requests_per_client < self.min_requests_per_run:
            return int(ceil(self.min_requests_per_run / clients))
        return int(self.min_requests_per_client)

    async def run(self, output_path: WritablePathLike | None = None):
        """Run the load test across all configured concurrency levels.

        Creates a :class:`~llmeter.runner.Runner` and iterates through
        ``sequence_of_clients``, running one test per concurrency level. In
        time-bound mode (``run_duration`` is set), each level runs for a fixed
        duration. In count-bound mode, each level sends a fixed number of
        requests per client.

        Args:
            load_path: Optional (local or remote) folder to save results. If provided, individual
            Run results will be written to `{output_path}/{test_name}/{NNNNN-clients}` subfolders.
            Default: `self.output_path` if set, else no files will be saved.

        Returns:
            LoadTestResult: A result object containing one
            :class:`~llmeter.results.Result` per concurrency level, keyed by
            client count.

        Example::

            load_test = LoadTest(
                endpoint=my_endpoint,
                payload=sample_payload,
                sequence_of_clients=[1, 5, 10],
                run_duration=30,
            )
            result = await load_test.run(output_path="outputs/my_test")

            # Access individual results by client count
            result.results[5].stats["requests_per_minute"]

            # Plot all standard charts
            result.plot_results()
        """
        output_path = ensure_path(output_path or self.output_path)
        if output_path:
            test_output_path = output_path / self._test_name
        else:
            test_output_path = None
        _runner = Runner(
            endpoint=self.endpoint,
            tokenizer=self.tokenizer,
            output_path=test_output_path,
        )

        self._results = []
        for c in tqdm(
            self.sequence_of_clients, desc="Configurations", disable=_disable_tqdm
        ):
            if self.run_duration is not None:
                result = await _runner.run(
                    payload=self.payload,
                    clients=c,
                    run_duration=self.run_duration,
                    run_name=f"{c:05.0f}-clients",
                    callbacks=self.callbacks,
                    low_memory=self.low_memory,
                    progress_bar_stats=self.progress_bar_stats,
                    output_path=test_output_path,
                )
            else:
                result = await _runner.run(
                    payload=self.payload,
                    clients=c,
                    n_requests=self._get_n_requests(c),
                    run_name=f"{c:05.0f}-clients",
                    callbacks=self.callbacks,
                    low_memory=self.low_memory,
                    progress_bar_stats=self.progress_bar_stats,
                    output_path=test_output_path,
                )
            self._results.append(result)

        return LoadTestResult(
            results={r.clients: r for r in self._results},
            test_name=self._test_name,
            output_path=test_output_path,
        )


@dataclass
class LatencyHeatmap:
    """
    Experiment to measure how latency varies by input and output token count

    This experiment uses a source text file to generate input prompts/payloads of different
    lengths, and measures how response time varies with both the input lengths and output/response
    lengths.

    Attributes:
        endpoint (Endpoint): The LLM endpoint to test.
        source_file (UPath | str): The source file from which prompts of different lengths will be
            sampled (see `llmeter.prompt_utils.CreatePromptCollection` for details).
        clients (int): The number of concurrent clients (requests) to use for the experiment. Note
            that using a high number of concurrent clients could impact observed latency.
        output_path (UPath | str | None): The (local or Cloud e.g. `s3://...`) path to save the
            results.
        input_lengths (Sequence[int]): The *approximate* input/prompt lengths to test. Since the
            locally-available `tokenizer` will often differ from the endpoint's own token counting,
            it's typically not possible to generate prompts with the exact specified token counts.
        output_lengths (Sequence[int]): The *target* output lengths to test. Since generation may
            stop early for certain prompts, and some endpoints may not report exact token counts in
            their responses, the results may not correspond exactly to these targets.
        requests_per_combination (int): The number of requests to make *for each combination* of
            input and output lengths.
        create_payload_fn (Callable | None): A function to create the actual endpoint payload for
            each invocation, from the sampled text prompt. Typically, you'll want to specify a
            prefix for your prompt in either this or the `create_payload_kwargs`. If not set, the
            endpoint's default `create_payload` method will be used.
        create_payload_kwargs (Dict): Keyword arguments to pass to the `create_payload_fn`.
        tokenizer (Tokenizer | None): A tokenizer to be used for sampling prompts of the specified
            lengths, and also estimating the generated output lengths if necessary for your
            endpoint. If not set, the `llmeter.tokenizers.DummyTokenizer` will be used.
    """

    endpoint: Endpoint
    source_file: ReadablePathLike
    clients: int = 4
    output_path: WritablePathLike | None = None
    input_lengths: list[int] = field(default_factory=lambda: [10, 50, 200, 500])
    output_lengths: list[int] = field(default_factory=lambda: [128, 256, 512, 1024])
    requests_per_combination: int = 1
    create_payload_fn: Callable[..., list[dict] | dict] | None = None
    create_payload_kwargs: dict = field(default_factory=dict)
    tokenizer: Tokenizer | None = None

    def __post_init__(self) -> None:
        _prompt_collection = CreatePromptCollection(
            requests_per_combination=self.requests_per_combination,
            input_lengths=self.input_lengths,
            output_lengths=self.output_lengths,
            source_file=ensure_path(self.source_file),
            tokenizer=self.tokenizer,  # type: ignore
        )

        self.create_payload_fn = self.create_payload_fn or self.endpoint.create_payload
        self.payload = _prompt_collection.create_collection()
        self.payload = [
            self.create_payload_fn(
                input_text, max_tokens=max_new_tokens, **self.create_payload_kwargs
            )
            for input_text, max_new_tokens in self.payload
        ]

        self._runner = Runner(
            endpoint=self.endpoint,
            output_path=ensure_path(self.output_path)
            if self.output_path is not None
            else None,
            tokenizer=self.tokenizer,
        )

    async def run(self, output_path=None):
        # Handle None output_path properly
        final_output_path = output_path or self.output_path
        if final_output_path is not None:
            final_output_path = ensure_path(final_output_path)

        heatmap_results = await self._runner.run(
            payload=self.payload,
            clients=self.clients,
            n_requests=len(self.input_lengths)
            * len(self.output_lengths)
            * self.requests_per_combination
            // self.clients,
            output_path=final_output_path,
        )
        self._results = heatmap_results
        return heatmap_results

    def plot_heatmaps(
        self, n_bins_x: int | None, n_bins_y: int | None, show: bool = True
    ):
        if not self._results:
            raise ValueError("No results to plot")
        f1 = plot_heatmap(
            self._results,
            "time_to_first_token",
            n_bins_x=n_bins_x,
            n_bins_y=n_bins_y,
            # output_path=self._results.output_path,
            show_scatter=True,
        )

        f2 = plot_heatmap(
            self._results,
            "time_to_last_token",
            n_bins_x=n_bins_x,
            n_bins_y=n_bins_y,
            # output_path=self._results.output_path,
            show_scatter=True,
        )

        if show:
            f1.show()
            f2.show()

        return f1, f2


def _resolve_concurrency(
    profile: ConcurrencyProfileInput,
    t: float,
    previous: int = 0,
) -> int:
    """Evaluate the concurrency profile at time t, returning desired client count.

    For waypoints: uses the most recent waypoint's value (step function).
    For callables: invokes f(t) directly.

    Returns the clamped [0, MAX_CLIENTS] and rounded result.
    On error (callable raises, returns non-numeric), returns `previous`.
    """
    if callable(profile):
        try:
            result = profile(t)
        except Exception:
            logger.warning(
                "Concurrency profile callable raised an exception at t=%.2f; "
                "keeping previous client count %d",
                t,
                previous,
            )
            return previous
        if isinstance(result, bool) or not isinstance(result, (int, float)):
            logger.warning(
                "Concurrency profile callable returned non-numeric value %r at t=%.2f; "
                "keeping previous client count %d",
                result,
                t,
                previous,
            )
            return previous
        # Round (half-even) and clamp to [0, MAX_CLIENTS]
        rounded = round(float(result))
        return max(0, min(rounded, MAX_CLIENTS))
    else:
        # Waypoints list: step function (use the most recent waypoint's value)
        waypoints: list[Waypoint] = profile
        if t <= waypoints[0][0]:
            value = waypoints[0][1]
        elif t >= waypoints[-1][0]:
            value = waypoints[-1][1]
        else:
            # Find the most recent waypoint at or before time t
            value = waypoints[0][1]
            for i in range(len(waypoints) - 1):
                if waypoints[i + 1][0] > t:
                    value = waypoints[i][1]
                    break
            else:
                value = waypoints[-1][1]
        return max(0, min(value, MAX_CLIENTS))


@dataclass
class DynamicLoadTestResult:
    """Result of a dynamic load test.

    Wraps the unified :class:`~llmeter.results.Result` with concurrency metadata
    and time-series visualization capabilities. Provides methods to plot
    concurrency, throughput, and latency over time, and supports save/load
    round-tripping for later analysis.

    Attributes:
        result: The unified Result containing all InvocationResponse objects.
        concurrency_profile: The concurrency profile used during the test.
        concurrency_log: List of (elapsed_seconds, actual_active_clients) observations.
        test_name: Name of the test run.
        control_interval: Seconds between controller evaluations.
        duration: Total test duration in seconds.
        output_path: Path where results were saved (if any).

    Example::

        # After running a dynamic load test:
        dynamic_test = DynamicLoadTest(
            endpoint=my_endpoint,
            payload={"prompt": "Hello"},
            clients_over_time=[(0, 1), (30, 20), (60, 1)],
            duration=60.0,
            output_path="outputs/dynamic_test",
        )
        result = await dynamic_test.run()

        # Plot concurrency over time (actual vs target)
        fig = result.plot_concurrency(show=True, format="html")

        # Plot all standard charts at once
        figures = result.plot_results(show=False, format="png")
        # figures["concurrency"], figures["throughput"], figures["latency"]

        # Load a previously saved result from disk
        loaded = DynamicLoadTestResult.load("outputs/dynamic_test/20250101-1200")
        loaded.plot_results()
    """

    result: Result
    concurrency_profile: ConcurrencyProfileInput
    concurrency_log: list[tuple[float, int]]
    test_name: str
    control_interval: float = 1.0
    duration: float = 0.0
    output_path: WritablePathLike | None = None

    @classmethod
    def load(
        cls, load_path: ReadablePathLike, load_responses: bool = True
    ) -> "DynamicLoadTestResult":
        """Load a dynamic load test result from disk.

        Checks for the presence of ``concurrency_log.json`` to confirm this is a
        dynamic load test result folder. Loads the base
        :class:`~llmeter.results.Result` via ``Result.load()``, then reconstructs
        the concurrency log and profile configuration from their respective JSON
        files.

        Args:
            load_path: Directory containing the saved dynamic load test result.
            load_responses: Whether to load individual invocation responses.
                Defaults to True.

        Returns:
            A :class:`DynamicLoadTestResult` with all metadata restored.

        Raises:
            ValueError: If ``concurrency_log.json`` is not found (indicating this
                is not a dynamic load test result folder).
        """
        path = ensure_path(load_path)

        # Check that this is a dynamic load test result folder
        concurrency_log_path = path / "concurrency_log.json"
        if not concurrency_log_path.exists():
            raise ValueError(
                f"Not a dynamic load test result folder: {path}. "
                "Missing concurrency_log.json."
            )

        # Load the base Result (responses.jsonl + summary.json)
        result = Result.load(path, load_responses=load_responses)

        # Load concurrency_log.json
        with concurrency_log_path.open("r") as f:
            concurrency_log_data = json.load(f)

        concurrency_log: list[tuple[float, int]] = [
            (float(entry[0]), int(entry[1])) for entry in concurrency_log_data["log"]
        ]
        control_interval = float(concurrency_log_data.get("control_interval", 1.0))
        duration = float(concurrency_log_data.get("duration", 0.0))

        # Load profile_config.json (reconstruct waypoints or mark as callable)
        profile_config_path = path / "profile_config.json"
        concurrency_profile: ConcurrencyProfileInput
        if profile_config_path.exists():
            with profile_config_path.open("r") as f:
                profile_data = json.load(f)

            if profile_data.get("type") == "waypoints":
                concurrency_profile = [
                    (float(wp[0]), int(wp[1])) for wp in profile_data["waypoints"]
                ]
            else:
                # Callable profiles cannot be deserialized; use a placeholder
                concurrency_profile = lambda t: 0  # noqa: E731
        else:
            # Fallback: no profile config, use placeholder
            concurrency_profile = lambda t: 0  # noqa: E731

        # Determine test_name from the result or directory name
        test_name = result.run_name or path.name

        return cls(
            result=result,
            concurrency_profile=concurrency_profile,
            concurrency_log=concurrency_log,
            test_name=test_name,
            control_interval=control_interval,
            duration=duration,
            output_path=path,
        )

    def save(self, output_path: WritablePathLike | None = None) -> None:
        """Save the dynamic load test result to disk.

        Writes the additional files that distinguish a dynamic load test from a
        standard run:

        - ``summary.json``: Re-saved with ``experiment_type: "dynamic_load_test"``
        - ``concurrency_log.json``: Controller observations over time
        - ``profile_config.json``: Profile metadata (waypoints or callable marker)

        The base ``Result.save()`` should be called before this method to write
        ``responses.jsonl`` and the initial ``summary.json``/``stats.json``.

        Args:
            output_path: Path to save results. Falls back to ``self.output_path``.

        Raises:
            ValueError: If no output path is provided.
        """
        resolved = output_path or self.output_path
        if resolved is None:
            raise ValueError("No output path provided")

        path = ensure_path(resolved)
        path.mkdir(parents=True, exist_ok=True)

        # Re-save summary.json with experiment_type field
        summary_path = path / "summary.json"
        summary_data: dict = {}
        if summary_path.exists():
            with summary_path.open("r") as f:
                summary_data = json.loads(f.read())
        summary_data["experiment_type"] = "dynamic_load_test"
        # Ensure key fields are present
        if "model_id" not in summary_data:
            summary_data["model_id"] = self.result.model_id
        if "total_requests" not in summary_data:
            summary_data["total_requests"] = self.result.total_requests
        if "total_test_time" not in summary_data:
            summary_data["total_test_time"] = self.result.total_test_time
        with summary_path.open("w") as f:
            f.write(json.dumps(summary_data, indent=4, default=str))

        # Write concurrency_log.json
        concurrency_log_path = path / "concurrency_log.json"
        concurrency_log_data = {
            "log": [[t, c] for t, c in self.concurrency_log],
            "control_interval": self.control_interval,
            "duration": self.duration,
        }
        with concurrency_log_path.open("w") as f:
            f.write(json.dumps(concurrency_log_data, indent=2))

        # Write profile_config.json
        profile_config_path = path / "profile_config.json"
        if isinstance(self.concurrency_profile, list):
            profile_data = {
                "type": "waypoints",
                "waypoints": [[t, c] for t, c in self.concurrency_profile],
            }
        else:
            profile_data = {
                "type": "callable",
                "note": (
                    "Original callable cannot be serialized. "
                    "Use concurrency_log.json for actual behavior."
                ),
            }
        with profile_config_path.open("w") as f:
            f.write(json.dumps(profile_data, indent=2))

    def plot_concurrency(self, show: bool = True, format: str = "html"):
        """Plot actual vs target concurrency over time.

        See :func:`~llmeter.plotting.plot_dynamic_concurrency` for details.
        """
        return plot_dynamic_concurrency(
            result=self.result,
            concurrency_log=self.concurrency_log,
            concurrency_profile=self.concurrency_profile,
            output_path=self.output_path,
            show=show,
            format=format,
        )

    def plot_throughput(
        self, window_sec: float = 1.0, show: bool = True, format: str = "html"
    ):
        """Plot requests per second over time.

        See :func:`~llmeter.plotting.plot_dynamic_throughput` for details.
        """
        return plot_dynamic_throughput(
            result=self.result,
            output_path=self.output_path,
            window_sec=window_sec,
            show=show,
            format=format,
        )

    def plot_latency(
        self, window_sec: float = 1.0, show: bool = True, format: str = "html"
    ):
        """Plot latency percentiles (p50, p95, p99) over time.

        See :func:`~llmeter.plotting.plot_dynamic_latency` for details.
        """
        return plot_dynamic_latency(
            result=self.result,
            output_path=self.output_path,
            window_sec=window_sec,
            show=show,
            format=format,
        )

    def plot_results(self, show: bool = True, format: str = "html") -> dict:
        """Plot all standard charts for the dynamic load test.

        See :func:`~llmeter.plotting.plot_dynamic_results` for details.
        """
        return plot_dynamic_results(
            result=self.result,
            concurrency_log=self.concurrency_log,
            concurrency_profile=self.concurrency_profile,
            output_path=self.output_path,
            show=show,
            format=format,
        )


@dataclass
class DynamicLoadTest:
    """Experiment with time-varying concurrency.

    Orchestrates a load test where the number of concurrent clients changes over
    time according to a user-provided concurrency profile. The profile can be
    either a callable ``f(t) -> clients`` or a list of waypoints that are linearly
    interpolated.

    Attributes:
        endpoint: The LLM endpoint to test.
        payload: The request payload(s) to send. A single dict is normalized to
            a one-element list.
        clients_over_time: Concurrency profile — either a callable mapping elapsed
            seconds to desired client count, or a list of (time, clients) waypoints.
        duration: Total test duration in seconds. Must be in [0.1, 86400].
        control_interval: Seconds between controller evaluations. Must be in [0.1, 60].
        output_path: Where to save results.
        tokenizer: Optional tokenizer for token counting.
        test_name: Name for this test. Defaults to current date/time (YYYYMMDD-HHMM).
        callbacks: Optional callbacks.
        timeout: Per-request timeout in seconds. Defaults to 60.0.

    Example::

        # Waypoints: ramp from 1 to 20 clients over 60s, then back down
        dynamic_test = DynamicLoadTest(
            endpoint=my_endpoint,
            payload=sample_payload,
            clients_over_time=[(0, 1), (30, 20), (60, 1)],
            duration=60.0,
            output_path="outputs/dynamic_test",
        )
        result = await dynamic_test.run()

        # Callable: sinusoidal concurrency pattern
        import math
        dynamic_test = DynamicLoadTest(
            endpoint=my_endpoint,
            payload=sample_payload,
            clients_over_time=lambda t: int(5 + 5 * math.sin(t / 10)),
            duration=120.0,
        )
        result = await dynamic_test.run()
    """

    endpoint: Endpoint
    payload: dict | list[dict]
    clients_over_time: ConcurrencyProfileInput
    duration: float
    control_interval: float = 1.0
    output_path: WritablePathLike | None = None
    tokenizer: Tokenizer | None = None
    test_name: str | None = None
    callbacks: list[Callback] | None = None
    timeout: float = 60.0

    def __post_init__(self) -> None:
        # Validate duration
        if (
            not isinstance(self.duration, (int, float))
            or self.duration < 0.1
            or self.duration > 86400
        ):
            raise ValueError(
                f"duration must be a positive number in [0.1, 86400], got {self.duration!r}"
            )

        # Validate control_interval
        if (
            not isinstance(self.control_interval, (int, float))
            or self.control_interval < 0.1
            or self.control_interval > 60
        ):
            raise ValueError(
                f"control_interval must be in [0.1, 60], got {self.control_interval!r}"
            )

        # Validate clients_over_time
        if isinstance(self.clients_over_time, list):
            if len(self.clients_over_time) < 2:
                raise ValueError(
                    "clients_over_time waypoints must have at least 2 entries, "
                    f"got {len(self.clients_over_time)}"
                )
            for i, entry in enumerate(self.clients_over_time):
                t = entry[0]
                if t < 0:
                    raise ValueError(
                        f"clients_over_time waypoint times must be non-negative, "
                        f"got {t} at index {i}"
                    )
            for i in range(1, len(self.clients_over_time)):
                if self.clients_over_time[i][0] <= self.clients_over_time[i - 1][0]:
                    raise ValueError(
                        "clients_over_time waypoint times must be strictly increasing, "
                        f"but time at index {i} ({self.clients_over_time[i][0]}) "
                        f"is not greater than time at index {i - 1} "
                        f"({self.clients_over_time[i - 1][0]})"
                    )
        elif not callable(self.clients_over_time):
            raise TypeError(
                f"clients_over_time must be a callable or a list of waypoints, "
                f"got {type(self.clients_over_time).__name__}"
            )

        # Normalize payload to list[dict]
        if isinstance(self.payload, dict):
            self.payload = [self.payload]

        # Set default test name
        self._test_name = self.test_name or f"{datetime.now():%Y%m%d-%H%M}"

    async def _controller(
        self,
        queue: asyncio.Queue,
        output_path: Path | None,
    ) -> tuple[list[tuple[float, int]], float, float, float]:
        """Main control loop. Spawns/stops clients to match concurrency profile.

        Periodically evaluates the concurrency profile and adjusts the number of
        active client handles accordingly. Handles are spawned as asyncio tasks
        running ``_client_loop`` in threads.

        Returns:
            A tuple of (concurrency_log, total_time, start_t, end_t).
        """
        active_handles: list[_ClientHandle] = []
        concurrency_log: list[tuple[float, int]] = []
        previous = 0

        start_t = time.perf_counter()

        while True:
            elapsed = time.perf_counter() - start_t
            if elapsed >= self.duration:
                break

            # Resolve desired concurrency at this point in time
            desired = _resolve_concurrency(self.clients_over_time, elapsed, previous)
            previous = desired

            # Prune dead handles
            active_handles = [h for h in active_handles if h.is_alive()]

            current = len(active_handles)

            # Scale up: spawn new client handles
            if desired > current:
                for _ in range(desired - current):
                    try:
                        stop_event = threading.Event()
                        task = asyncio.ensure_future(
                            asyncio.to_thread(
                                _client_loop,
                                self.endpoint,
                                self.payload,
                                stop_event,
                                queue,
                                self.callbacks,
                                self.timeout,
                            )
                        )
                        handle = _ClientHandle(
                            _stop_event=stop_event,
                            _task=task,
                            _spawn_time=time.perf_counter(),
                        )
                        active_handles.append(handle)
                    except OSError as e:
                        logger.warning(
                            "Resource error spawning client handle: %s. "
                            "Continuing with %d available handles.",
                            e,
                            len([h for h in active_handles if h.is_alive()]),
                        )
                        break

            # Scale down: signal excess handles in LIFO order
            elif desired < current:
                excess = current - desired
                for _ in range(excess):
                    handle = active_handles.pop()
                    handle.signal_stop()

            # Log actual concurrency
            alive_count = len([h for h in active_handles if h.is_alive()])
            concurrency_log.append((elapsed, alive_count))

            await asyncio.sleep(self.control_interval)

        # Signal all remaining handles to stop
        for handle in active_handles:
            handle.signal_stop()

        # Wait up to 30s grace period for all tasks to complete
        pending_tasks = [h._task for h in active_handles if h.is_alive()]
        if pending_tasks:
            done, still_running = await asyncio.wait(pending_tasks, timeout=30)
            if still_running:
                logger.warning(
                    "Grace period exceeded: %d client task(s) still running after 30s. "
                    "Continuing with partial results.",
                    len(still_running),
                )

        # Send sentinel to signal response processor that no more responses are coming
        await queue.put(None)

        end_t = time.perf_counter()
        total_time = end_t - start_t

        return (concurrency_log, total_time, start_t, end_t)

    async def _process_responses(
        self,
        queue: asyncio.Queue,
        output_path: Path | None,
    ) -> list[InvocationResponse]:
        """Process responses from the queue. Applies tokenizer and callbacks.

        Reads from the queue until a ``None`` sentinel is received. For each
        response, applies token counting (if a tokenizer is configured),
        computes ``time_per_output_token``, invokes ``after_invoke`` callbacks,
        and accumulates all responses into a list.

        If ``output_path`` is provided, streams each response to a
        ``responses.jsonl`` file in append mode.

        Returns:
            List of all collected :class:`InvocationResponse` objects.
        """
        responses: list[InvocationResponse] = []

        # Resolve the output file path for streaming
        output_file: Path | None = None
        if output_path is not None:
            output_file = ensure_path(output_path) / "responses.jsonl"
            output_file.parent.mkdir(parents=True, exist_ok=True)

        while True:
            response: InvocationResponse | None = await queue.get()

            if response is None:
                logger.debug("Response processor received sentinel, stopping.")
                break

            # Apply tokenizer: count input/output tokens
            if self.tokenizer is not None and response.error is None:
                if response.num_tokens_input is None:
                    text = response.input_prompt
                    if text is not None:
                        if not isinstance(text, str):
                            try:
                                text = str(text)
                            except Exception:
                                text = None
                        if text is not None:
                            response.num_tokens_input = len(self.tokenizer.encode(text))

                if response.num_tokens_output is None:
                    text = response.response_text
                    if text is not None:
                        if not isinstance(text, str):
                            try:
                                text = str(text)
                            except Exception:
                                text = None
                        if text is not None:
                            response.num_tokens_output = len(
                                self.tokenizer.encode(text)
                            )

            # Compute time_per_output_token
            if response.time_per_output_token is None:
                if (
                    response.time_to_last_token
                    and response.num_tokens_output
                    and response.num_tokens_output > 1
                    and response.time_to_first_token
                ):
                    response.time_per_output_token = (
                        response.time_to_last_token - response.time_to_first_token
                    ) / (response.num_tokens_output - 1)

            # Call after_invoke callbacks
            if self.callbacks is not None:
                for cb in self.callbacks:
                    await cb.after_invoke(response)

            # Accumulate response
            responses.append(response)

            # Stream to disk if output_path configured
            if output_file is not None:
                with output_file.open("a") as f:
                    f.write(response.to_json() + "\n")

        return responses

    async def run(
        self, output_path: WritablePathLike | None = None
    ) -> "DynamicLoadTestResult":
        """Execute the dynamic load test.

        This is the main entry point for users. Orchestrates the controller loop,
        response processing, callback invocation, and result construction.

        Args:
            output_path: Optional path to save results. Overrides ``self.output_path``
                if provided.

        Returns:
            A :class:`DynamicLoadTestResult` containing the unified Result and
            concurrency metadata.
        """
        # Resolve output path
        resolved_output: Path | None = None
        raw_output = output_path or self.output_path
        if raw_output is not None:
            resolved_output = ensure_path(raw_output) / self._test_name
            resolved_output.mkdir(parents=True, exist_ok=True)

        # Call before_run callbacks
        if self.callbacks is not None:
            for cb in self.callbacks:
                await cb.before_run(self)

        # Set up the asyncio.Queue for response collection
        queue: asyncio.Queue = asyncio.Queue()

        # Start response processor as a background task
        processor_task = asyncio.create_task(
            self._process_responses(queue, resolved_output)
        )

        # Run the controller (awaits until duration elapses + grace period)
        concurrency_log, total_time, start_t, end_t = await self._controller(
            queue, resolved_output
        )

        # Await response processor completion (it stops after receiving sentinel)
        responses = await processor_task

        # Construct Result from collected responses
        result = Result(
            responses=responses,
            total_requests=len(responses),
            clients=0,  # dynamic — no fixed client count
            n_requests=len(responses),
            total_test_time=total_time,
            model_id=self.endpoint.model_id,
            output_path=resolved_output,
            provider=self.endpoint.provider,
            endpoint_name=self.endpoint.endpoint_name,
            run_name=self._test_name,
        )

        # Call after_run callbacks on the final Result
        if self.callbacks is not None:
            for cb in self.callbacks:
                await cb.after_run(result)

        # Save to disk if output_path is configured
        if resolved_output is not None:
            result.save(resolved_output)

        # Construct and return DynamicLoadTestResult
        dynamic_result = DynamicLoadTestResult(
            result=result,
            concurrency_profile=self.clients_over_time,
            concurrency_log=concurrency_log,
            test_name=self._test_name,
            control_interval=self.control_interval,
            duration=self.duration,
            output_path=resolved_output,
        )

        # Save additional dynamic-test-specific files
        if resolved_output is not None:
            dynamic_result.save(resolved_output)

        return dynamic_result
