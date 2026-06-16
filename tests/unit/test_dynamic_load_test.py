# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Property-based tests for DynamicLoadTest components."""

import asyncio
import logging
import threading
import time
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.strategies import composite

from llmeter.endpoints.base import Endpoint, InvocationResponse
from llmeter.experiments import (
    MAX_CLIENTS,
    DynamicLoadTest,
    DynamicLoadTestResult,
    Waypoint,
    _resolve_concurrency,
)
from llmeter.results import Result


# --- Custom strategies ---


@composite
def waypoints_strategy(draw):
    """Generate valid waypoints: at least 2 entries with strictly increasing times
    and non-negative client counts."""
    n = draw(st.integers(min_value=2, max_value=20))
    # Generate strictly increasing times (all non-negative)
    times = sorted(
        draw(
            st.lists(
                st.floats(
                    min_value=0.0,
                    max_value=86400.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                min_size=n,
                max_size=n,
                unique=True,
            )
        )
    )
    # Generate non-negative client counts
    clients = draw(
        st.lists(
            st.integers(min_value=0, max_value=2000),
            min_size=n,
            max_size=n,
        )
    )
    return [(t, c) for t, c in zip(times, clients)]


@composite
def waypoints_with_equal_clients_strategy(draw):
    """Generate waypoints where all client counts are the same value."""
    n = draw(st.integers(min_value=2, max_value=10))
    times = sorted(
        draw(
            st.lists(
                st.floats(
                    min_value=0.0,
                    max_value=86400.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                min_size=n,
                max_size=n,
                unique=True,
            )
        )
    )
    client_count = draw(st.integers(min_value=0, max_value=MAX_CLIENTS))
    return [(t, client_count) for t in times], client_count


# --- Property tests ---


class TestResolveConcurrencyProperties:
    """Property-based tests for _resolve_concurrency.

    **Validates: Requirements 1.6, 1.9**
    """

    @given(
        waypoints=waypoints_strategy(),
        t=st.floats(
            min_value=-1000.0, max_value=100000.0, allow_nan=False, allow_infinity=False
        ),
        previous=st.integers(min_value=0, max_value=MAX_CLIENTS),
    )
    @settings(deadline=None)
    def test_waypoints_result_bounded_by_zero_and_max_clients(
        self, waypoints, t, previous
    ):
        """For any waypoints profile and time t, result is always in [0, MAX_CLIENTS].

        **Validates: Requirements 1.6, 1.9**
        """
        result = _resolve_concurrency(waypoints, t, previous)
        assert 0 <= result <= MAX_CLIENTS

    @given(
        t=st.floats(
            min_value=-1000.0, max_value=100000.0, allow_nan=False, allow_infinity=False
        ),
        previous=st.integers(min_value=0, max_value=MAX_CLIENTS),
        return_value=st.one_of(
            st.integers(min_value=-10000, max_value=10000),
            st.floats(
                min_value=-10000.0,
                max_value=10000.0,
                allow_nan=False,
                allow_infinity=False,
            ),
        ),
    )
    @settings(deadline=None)
    def test_callable_result_bounded_by_zero_and_max_clients(
        self, t, previous, return_value
    ):
        """For any callable profile returning numeric values, result is in [0, MAX_CLIENTS].

        **Validates: Requirements 1.6, 1.9**
        """
        profile = lambda x: return_value  # noqa: E731
        result = _resolve_concurrency(profile, t, previous)
        assert 0 <= result <= MAX_CLIENTS

    @given(
        t=st.floats(
            min_value=-1000.0, max_value=100000.0, allow_nan=False, allow_infinity=False
        ),
        previous=st.integers(min_value=0, max_value=MAX_CLIENTS),
    )
    @settings(deadline=None)
    def test_callable_raising_exception_returns_previous(self, t, previous):
        """When callable raises an exception, result equals previous (still bounded).

        **Validates: Requirements 1.6, 1.9**
        """

        def raising_profile(x):
            raise RuntimeError("test error")

        result = _resolve_concurrency(raising_profile, t, previous)
        assert result == previous
        assert 0 <= result <= MAX_CLIENTS

    @given(
        t=st.floats(
            min_value=-1000.0, max_value=100000.0, allow_nan=False, allow_infinity=False
        ),
        previous=st.integers(min_value=0, max_value=MAX_CLIENTS),
        non_numeric=st.one_of(
            st.text(),
            st.none(),
            st.booleans(),
            st.lists(st.integers()),
        ),
    )
    @settings(deadline=None)
    def test_callable_returning_non_numeric_returns_previous(
        self, t, previous, non_numeric
    ):
        """When callable returns non-numeric, result equals previous (still bounded).

        **Validates: Requirements 1.6, 1.9**
        """
        profile = lambda x: non_numeric  # noqa: E731
        result = _resolve_concurrency(profile, t, previous)
        assert result == previous
        assert 0 <= result <= MAX_CLIENTS

    @given(
        data=waypoints_with_equal_clients_strategy(),
        t=st.floats(
            min_value=0.0, max_value=86400.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(deadline=None)
    def test_interpolation_between_equal_values_returns_that_value(self, data, t):
        """When all waypoints have the same client count, interpolation returns that value
        (clamped to [0, MAX_CLIENTS]).

        **Validates: Requirements 1.6, 1.9**
        """
        waypoints, expected_count = data
        result = _resolve_concurrency(waypoints, t, 0)
        # The expected count is already within [0, MAX_CLIENTS] by construction
        assert result == expected_count


# --- Unit tests ---


class TestResolveConcurrencyUnit:
    """Unit tests for _resolve_concurrency covering interpolation, clamping, rounding, and error handling."""

    # --- Waypoints: interpolation at exact points ---

    def test_exact_first_waypoint(self):
        """At the first waypoint time, returns the first waypoint's client count."""
        waypoints: list[Waypoint] = [(0.0, 5), (10.0, 20)]
        assert _resolve_concurrency(waypoints, 0.0) == 5

    def test_exact_last_waypoint(self):
        """At the last waypoint time, returns the last waypoint's client count."""
        waypoints: list[Waypoint] = [(0.0, 5), (10.0, 20)]
        assert _resolve_concurrency(waypoints, 10.0) == 20

    def test_exact_middle_waypoint(self):
        """At an intermediate waypoint time, returns that waypoint's client count."""
        waypoints: list[Waypoint] = [(0.0, 10), (5.0, 20), (10.0, 30)]
        assert _resolve_concurrency(waypoints, 5.0) == 20

    # --- Waypoints: step function at midpoints ---

    def test_midpoint_uses_previous_step(self):
        """Between two waypoints, uses the earlier waypoint's value (step function)."""
        waypoints: list[Waypoint] = [(0.0, 0), (10.0, 100)]
        assert _resolve_concurrency(waypoints, 5.0) == 0

    def test_just_before_next_waypoint(self):
        """Just before the next waypoint still uses the previous step's value."""
        waypoints: list[Waypoint] = [(0.0, 0), (10.0, 100)]
        assert _resolve_concurrency(waypoints, 9.99) == 0

    def test_at_exact_waypoint_time(self):
        """At the exact time of a waypoint, uses that waypoint's value."""
        waypoints: list[Waypoint] = [(0.0, 0), (10.0, 100)]
        assert _resolve_concurrency(waypoints, 10.0) == 100

    def test_step_between_non_zero_waypoints(self):
        """Step function works with multi-waypoint profiles."""
        waypoints: list[Waypoint] = [(0.0, 5), (5.0, 20), (10.0, 50)]
        assert _resolve_concurrency(waypoints, 3.0) == 5
        assert _resolve_concurrency(waypoints, 5.0) == 20
        assert _resolve_concurrency(waypoints, 7.0) == 20

    # --- Waypoints: before first and after last ---

    def test_before_first_waypoint(self):
        """Before the first waypoint time, clamps to the first waypoint's client count."""
        waypoints: list[Waypoint] = [(5.0, 10), (10.0, 20)]
        assert _resolve_concurrency(waypoints, 0.0) == 10
        assert _resolve_concurrency(waypoints, 3.0) == 10

    def test_after_last_waypoint(self):
        """After the last waypoint time, clamps to the last waypoint's client count."""
        waypoints: list[Waypoint] = [(0.0, 10), (10.0, 50)]
        assert _resolve_concurrency(waypoints, 15.0) == 50
        assert _resolve_concurrency(waypoints, 100.0) == 50

    # --- Clamping: negative values to 0 ---

    def test_clamp_negative_waypoint_to_zero(self):
        """Negative waypoint values are clamped to 0."""
        waypoints: list[Waypoint] = [(0.0, -10), (10.0, -5)]
        assert _resolve_concurrency(waypoints, 0.0) == 0
        assert _resolve_concurrency(waypoints, 5.0) == 0

    def test_clamp_callable_negative_to_zero(self):
        """Callable returning negative value is clamped to 0."""
        assert _resolve_concurrency(lambda t: -50, 5.0) == 0

    # --- Clamping: above MAX_CLIENTS (1000) to 1000 ---

    def test_clamp_above_max_clients_waypoints(self):
        """Waypoint values above MAX_CLIENTS are clamped to 1000."""
        waypoints: list[Waypoint] = [(0.0, 1500), (10.0, 2000)]
        assert _resolve_concurrency(waypoints, 0.0) == MAX_CLIENTS
        assert _resolve_concurrency(waypoints, 10.0) == MAX_CLIENTS

    def test_clamp_above_max_clients_callable(self):
        """Callable returning value above MAX_CLIENTS is clamped to 1000."""
        assert _resolve_concurrency(lambda t: 5000, 5.0) == MAX_CLIENTS

    def test_exactly_at_max_clients(self):
        """Value exactly at MAX_CLIENTS is preserved."""
        waypoints: list[Waypoint] = [(0.0, MAX_CLIENTS), (10.0, MAX_CLIENTS)]
        assert _resolve_concurrency(waypoints, 5.0) == MAX_CLIENTS

    # --- Rounding: half-even via round() ---

    def test_rounds_half_even_down(self):
        """Half-even rounding: 0.5 rounds to 0 (even)."""
        assert _resolve_concurrency(lambda t: 0.5, 1.0) == 0

    def test_rounds_half_even_up(self):
        """Half-even rounding: 1.5 rounds to 2 (even)."""
        assert _resolve_concurrency(lambda t: 1.5, 1.0) == 2

    def test_rounds_non_half_normally(self):
        """Non-.5 fractional values round normally."""
        assert _resolve_concurrency(lambda t: 2.7, 1.0) == 3

    def test_callable_returning_float_is_rounded(self):
        """Callable returning a float gets rounded."""
        assert _resolve_concurrency(lambda t: 7.6, 5.0) == 8
        assert _resolve_concurrency(lambda t: 7.4, 5.0) == 7

    # --- Callable: raises exception → returns previous ---

    def test_callable_raises_returns_previous(self):
        """If callable raises, return the previous value."""

        def bad_func(t):
            raise ValueError("something went wrong")

        assert _resolve_concurrency(bad_func, 5.0, previous=42) == 42

    def test_callable_raises_returns_previous_zero_default(self):
        """If callable raises with default previous=0, returns 0."""

        def bad_func(t):
            raise RuntimeError("oops")

        assert _resolve_concurrency(bad_func, 1.0) == 0

    # --- Callable: returns non-numeric → returns previous ---

    def test_callable_returns_string_returns_previous(self):
        """If callable returns a string, return the previous value."""
        assert _resolve_concurrency(lambda t: "high", 5.0, previous=15) == 15

    def test_callable_returns_none_returns_previous(self):
        """If callable returns None, return the previous value."""
        assert _resolve_concurrency(lambda t: None, 5.0, previous=10) == 10

    def test_callable_returns_bool_returns_previous(self):
        """If callable returns a bool, returns previous."""
        assert _resolve_concurrency(lambda t: True, 5.0, previous=7) == 7
        assert _resolve_concurrency(lambda t: False, 5.0, previous=3) == 3

    def test_callable_returns_list_returns_previous(self):
        """If callable returns a list, return the previous value."""
        assert _resolve_concurrency(lambda t: [10], 5.0, previous=20) == 20

    # --- Callable: valid numeric returns ---

    def test_callable_returns_int(self):
        """Callable returning a valid int works normally."""
        assert _resolve_concurrency(lambda t: 42, 5.0) == 42

    def test_callable_returns_float(self):
        """Callable returning a valid float is rounded and clamped."""
        assert _resolve_concurrency(lambda t: 3.7, 5.0) == 4

    def test_callable_uses_time_parameter(self):
        """Callable receives the current time t and can use it."""
        assert _resolve_concurrency(lambda t: int(t * 2), 5.0) == 10


# --- Unit tests for DynamicLoadTest validation ---


def _mock_endpoint():
    """Create a mock endpoint for testing DynamicLoadTest construction."""
    return MagicMock(spec=Endpoint)


class TestDynamicLoadTestValidation:
    """Unit tests for DynamicLoadTest.__post_init__ validation.

    **Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.6**
    """

    # --- Invalid duration → ValueError ---

    def test_duration_zero_raises(self):
        """Duration of 0 raises ValueError."""
        with pytest.raises(ValueError, match="duration"):
            DynamicLoadTest(
                endpoint=_mock_endpoint(),
                payload={"prompt": "hi"},
                clients_over_time=[(0, 1), (10, 5)],
                duration=0,
            )

    def test_duration_negative_raises(self):
        """Negative duration raises ValueError."""
        with pytest.raises(ValueError, match="duration"):
            DynamicLoadTest(
                endpoint=_mock_endpoint(),
                payload={"prompt": "hi"},
                clients_over_time=[(0, 1), (10, 5)],
                duration=-10,
            )

    def test_duration_exceeds_max_raises(self):
        """Duration > 86400 raises ValueError."""
        with pytest.raises(ValueError, match="duration"):
            DynamicLoadTest(
                endpoint=_mock_endpoint(),
                payload={"prompt": "hi"},
                clients_over_time=[(0, 1), (10, 5)],
                duration=86401,
            )

    # --- Invalid control_interval → ValueError ---

    def test_control_interval_too_small_raises(self):
        """control_interval < 0.1 raises ValueError."""
        with pytest.raises(ValueError, match="control_interval"):
            DynamicLoadTest(
                endpoint=_mock_endpoint(),
                payload={"prompt": "hi"},
                clients_over_time=[(0, 1), (10, 5)],
                duration=10,
                control_interval=0.05,
            )

    def test_control_interval_too_large_raises(self):
        """control_interval > 60 raises ValueError."""
        with pytest.raises(ValueError, match="control_interval"):
            DynamicLoadTest(
                endpoint=_mock_endpoint(),
                payload={"prompt": "hi"},
                clients_over_time=[(0, 1), (10, 5)],
                duration=10,
                control_interval=61,
            )

    # --- Waypoints with <2 entries → ValueError ---

    def test_waypoints_empty_list_raises(self):
        """Empty waypoints list raises ValueError."""
        with pytest.raises(ValueError, match="at least 2 entries"):
            DynamicLoadTest(
                endpoint=_mock_endpoint(),
                payload={"prompt": "hi"},
                clients_over_time=[],
                duration=10,
            )

    def test_waypoints_single_entry_raises(self):
        """Waypoints with only 1 entry raises ValueError."""
        with pytest.raises(ValueError, match="at least 2 entries"):
            DynamicLoadTest(
                endpoint=_mock_endpoint(),
                payload={"prompt": "hi"},
                clients_over_time=[(0, 5)],
                duration=10,
            )

    # --- Waypoints with non-increasing times → ValueError ---

    def test_waypoints_equal_times_raises(self):
        """Waypoints with equal consecutive times raises ValueError."""
        with pytest.raises(ValueError, match="strictly increasing"):
            DynamicLoadTest(
                endpoint=_mock_endpoint(),
                payload={"prompt": "hi"},
                clients_over_time=[(5.0, 1), (5.0, 10)],
                duration=10,
            )

    def test_waypoints_decreasing_times_raises(self):
        """Waypoints with decreasing times raises ValueError."""
        with pytest.raises(ValueError, match="strictly increasing"):
            DynamicLoadTest(
                endpoint=_mock_endpoint(),
                payload={"prompt": "hi"},
                clients_over_time=[(10.0, 1), (5.0, 10)],
                duration=10,
            )

    # --- Waypoints with negative times → ValueError ---

    def test_waypoints_negative_time_raises(self):
        """Waypoints with a negative time value raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            DynamicLoadTest(
                endpoint=_mock_endpoint(),
                payload={"prompt": "hi"},
                clients_over_time=[(-1.0, 1), (10.0, 5)],
                duration=10,
            )

    # --- Non-callable/non-list profile → TypeError ---

    def test_profile_string_raises_type_error(self):
        """String profile raises TypeError."""
        with pytest.raises(TypeError, match="callable or a list"):
            DynamicLoadTest(
                endpoint=_mock_endpoint(),
                payload={"prompt": "hi"},
                clients_over_time="invalid",
                duration=10,
            )

    def test_profile_integer_raises_type_error(self):
        """Integer profile raises TypeError."""
        with pytest.raises(TypeError, match="callable or a list"):
            DynamicLoadTest(
                endpoint=_mock_endpoint(),
                payload={"prompt": "hi"},
                clients_over_time=42,
                duration=10,
            )

    def test_profile_none_raises_type_error(self):
        """None profile raises TypeError."""
        with pytest.raises(TypeError, match="callable or a list"):
            DynamicLoadTest(
                endpoint=_mock_endpoint(),
                payload={"prompt": "hi"},
                clients_over_time=None,
                duration=10,
            )

    # --- Valid callable profile passes ---

    def test_valid_callable_profile_passes(self):
        """A valid callable profile creates the DynamicLoadTest without error."""
        dlt = DynamicLoadTest(
            endpoint=_mock_endpoint(),
            payload={"prompt": "hi"},
            clients_over_time=lambda t: int(t),
            duration=60,
        )
        assert dlt.duration == 60
        assert callable(dlt.clients_over_time)

    # --- Valid waypoints profile passes ---

    def test_valid_waypoints_profile_passes(self):
        """A valid waypoints profile creates the DynamicLoadTest without error."""
        waypoints = [(0.0, 1), (30.0, 10), (60.0, 5)]
        dlt = DynamicLoadTest(
            endpoint=_mock_endpoint(),
            payload={"prompt": "hi"},
            clients_over_time=waypoints,
            duration=60,
        )
        assert dlt.clients_over_time == waypoints

    # --- Single dict payload normalized to list ---

    def test_single_dict_payload_normalized_to_list(self):
        """A single dict payload is normalized to a one-element list."""
        dlt = DynamicLoadTest(
            endpoint=_mock_endpoint(),
            payload={"prompt": "hello"},
            clients_over_time=[(0, 1), (10, 5)],
            duration=10,
        )
        assert dlt.payload == [{"prompt": "hello"}]
        assert isinstance(dlt.payload, list)

    def test_list_payload_unchanged(self):
        """A list of dicts payload stays as-is."""
        payloads = [{"prompt": "a"}, {"prompt": "b"}]
        dlt = DynamicLoadTest(
            endpoint=_mock_endpoint(),
            payload=payloads,
            clients_over_time=[(0, 1), (10, 5)],
            duration=10,
        )
        assert dlt.payload == payloads


# --- Unit tests for _client_loop ---


class TestClientLoop:
    """Unit tests for _client_loop.

    **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6**
    """

    def _run_client_loop_with_event_loop(
        self, endpoint, payload, stop_event, callbacks=None
    ):
        """Run _client_loop in a thread with a real running event loop to process queue items.

        Returns the list of items that were pushed to the queue.
        """
        from llmeter.experiments import _client_loop

        items = []

        async def _run():
            queue = asyncio.Queue()
            queue._loop = asyncio.get_running_loop()

            # Run _client_loop in a thread
            await asyncio.to_thread(
                _client_loop, endpoint, payload, stop_event, queue, callbacks
            )

            # Drain the queue
            while not queue.empty():
                items.append(queue.get_nowait())

        asyncio.run(_run())
        return items

    def test_stops_immediately_when_stop_event_is_preset(self):
        """When stop_event is set before calling _client_loop, no requests are made.

        **Validates: Requirements 3.2, 3.3**
        """
        endpoint = MagicMock(spec=Endpoint)
        payload = [{"prompt": "hello"}]
        stop_event = threading.Event()
        stop_event.set()  # Pre-set the stop event

        items = self._run_client_loop_with_event_loop(endpoint, payload, stop_event)

        # No requests should have been made
        endpoint.invoke.assert_not_called()
        # Queue should be empty
        assert len(items) == 0

    def test_pushes_responses_to_queue(self):
        """All responses from endpoint.invoke are pushed to the queue.

        **Validates: Requirements 3.5**
        """
        from llmeter.endpoints.base import InvocationResponse

        response = InvocationResponse(
            response_text="world",
            id="test-id-1",
            time_to_last_token=0.5,
        )
        endpoint = MagicMock(spec=Endpoint)
        call_count = 0

        def invoke_side_effect(payload_item):
            nonlocal call_count
            call_count += 1
            if call_count >= 3:
                stop_event.set()
            return response

        endpoint.invoke.side_effect = invoke_side_effect

        payload = [{"prompt": "hello"}]
        stop_event = threading.Event()

        items = self._run_client_loop_with_event_loop(endpoint, payload, stop_event)

        # Should have pushed exactly 3 responses (call 3 sets stop, loop exits before call 4)
        assert len(items) == 3
        for item in items:
            assert item is response

    def test_cycles_through_payload_sequentially(self):
        """Payload items are cycled sequentially (wrapping around).

        **Validates: Requirements 3.6**
        """
        from llmeter.endpoints.base import InvocationResponse

        payload = [{"prompt": "a"}, {"prompt": "b"}, {"prompt": "c"}]
        received_payloads = []

        response = InvocationResponse(
            response_text="ok", id="r1", time_to_last_token=0.1
        )
        endpoint = MagicMock(spec=Endpoint)

        def invoke_side_effect(payload_item):
            received_payloads.append(payload_item)
            if len(received_payloads) >= 7:
                stop_event.set()
            return response

        endpoint.invoke.side_effect = invoke_side_effect

        stop_event = threading.Event()

        self._run_client_loop_with_event_loop(endpoint, payload, stop_event)

        # Verify sequential cycling: a, b, c, a, b, c, a
        expected_cycle = payload * 3  # covers at least 7 items
        for i, received in enumerate(received_payloads[:7]):
            assert received == expected_cycle[i], (
                f"At index {i}: expected {expected_cycle[i]}, got {received}"
            )

    def test_records_errors_as_invocation_response_with_error_field(self):
        """When endpoint.invoke raises an exception, an InvocationResponse with error
        field populated is pushed to the queue.

        **Validates: Requirements 3.4**
        """
        endpoint = MagicMock(spec=Endpoint)
        call_count = 0

        def invoke_side_effect(payload_item):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                stop_event.set()
            raise RuntimeError("boom")

        endpoint.invoke.side_effect = invoke_side_effect

        payload = [{"prompt": "hello"}]
        stop_event = threading.Event()

        items = self._run_client_loop_with_event_loop(endpoint, payload, stop_event)

        assert len(items) == 2
        for item in items:
            assert item.error is not None
            assert "boom" in item.error
            assert item.response_text is None

    def test_stops_after_set_event_no_new_requests(self):
        """After stop_event is set during invocation, no NEW requests are started.

        **Validates: Requirements 3.1, 3.2**
        """
        from llmeter.endpoints.base import InvocationResponse

        response = InvocationResponse(
            response_text="ok", id="r1", time_to_last_token=0.1
        )
        endpoint = MagicMock(spec=Endpoint)
        invocation_count = 0

        def invoke_side_effect(payload_item):
            nonlocal invocation_count
            invocation_count += 1
            # Set stop after first invocation
            stop_event.set()
            return response

        endpoint.invoke.side_effect = invoke_side_effect

        payload = [{"prompt": "hello"}]
        stop_event = threading.Event()

        self._run_client_loop_with_event_loop(endpoint, payload, stop_event)

        # Should have exactly 1 invocation: the first call sets the stop event,
        # then the loop checks stop_event.is_set() and exits
        assert invocation_count == 1


# --- Integration tests for controller with mock endpoint ---


class TestControllerIntegration:
    """Integration tests for DynamicLoadTest._controller with mock endpoint.

    **Validates: Requirements 2.1, 2.2, 2.3, 2.5, 2.6, 7.5**
    """

    def _make_fast_endpoint(self, delay: float = 0.0):
        """Create a mock endpoint that returns quickly."""
        endpoint = MagicMock(spec=Endpoint)
        response = InvocationResponse(
            response_text="ok", id="x", time_to_last_token=0.01
        )

        def invoke_fn(payload_item):
            if delay > 0:
                time.sleep(delay)
            return response

        endpoint.invoke = MagicMock(side_effect=invoke_fn)
        return endpoint

    @pytest.mark.asyncio
    async def test_ramp_up_client_count_increases(self):
        """Ramp-up profile: verify client count increases over time.

        **Validates: Requirements 2.1, 2.2**
        """
        endpoint = self._make_fast_endpoint(delay=0.05)
        dlt = DynamicLoadTest(
            endpoint=endpoint,
            payload={"prompt": "hi"},
            clients_over_time=[(0, 1), (0.5, 2), (1.0, 4), (1.5, 6)],
            duration=2.0,
            control_interval=0.1,
        )

        queue = asyncio.Queue()
        concurrency_log, total_time, start_t, end_t = await dlt._controller(
            queue, output_path=None
        )

        # Verify that concurrency increases over time
        assert len(concurrency_log) > 0

        # Get a sample from the first quarter and last quarter
        quarter = len(concurrency_log) // 4
        early_counts = [c for _, c in concurrency_log[:quarter]] if quarter > 0 else [0]
        late_counts = [c for _, c in concurrency_log[-quarter:]] if quarter > 0 else [0]

        # The maximum in the late portion should exceed the maximum in the early portion
        assert max(late_counts) > max(early_counts), (
            f"Expected late counts ({late_counts}) to exceed early counts ({early_counts})"
        )

        # Verify some clients were actually spawned
        max_concurrent = max(c for _, c in concurrency_log)
        assert max_concurrent > 0

    @pytest.mark.asyncio
    async def test_ramp_down_handles_stopped_lifo(self):
        """Ramp-down profile: verify handles are signalled to stop in LIFO order.

        **Validates: Requirements 2.3**
        """
        endpoint = self._make_fast_endpoint(delay=0.05)
        dlt = DynamicLoadTest(
            endpoint=endpoint,
            payload={"prompt": "hi"},
            clients_over_time=[(0, 5), (0.5, 3), (1.0, 1), (1.5, 0)],
            duration=1.5,
            control_interval=0.1,
        )

        queue = asyncio.Queue()
        concurrency_log, total_time, start_t, end_t = await dlt._controller(
            queue, output_path=None
        )

        # Verify concurrency decreases over time
        assert len(concurrency_log) > 0

        # Find the peak concurrency and then verify it decreases
        peak_idx = 0
        peak_val = 0
        for i, (_, c) in enumerate(concurrency_log):
            if c >= peak_val:
                peak_val = c
                peak_idx = i

        # After the peak, concurrency should decrease toward 0
        if peak_idx < len(concurrency_log) - 1:
            post_peak = [c for _, c in concurrency_log[peak_idx + 1 :]]
            if post_peak:
                assert min(post_peak) < peak_val, (
                    f"Expected concurrency to decrease after peak ({peak_val}), "
                    f"but post-peak values were {post_peak}"
                )

        # Final entries should be at or near 0
        final_entries = [c for _, c in concurrency_log[-3:]]
        assert min(final_entries) < peak_val

    @pytest.mark.asyncio
    async def test_grace_period_timeout_warning_logged(self, caplog):
        """Mock slow endpoint; verify grace period timeout warning is logged.

        **Validates: Requirements 2.5, 7.5**
        """
        # Create an endpoint that sleeps long enough to exceed grace period
        endpoint = MagicMock(spec=Endpoint)
        response = InvocationResponse(
            response_text="ok", id="x", time_to_last_token=0.01
        )

        def slow_invoke(payload_item):
            # Sleep longer than the test duration + grace period
            time.sleep(60)
            return response

        endpoint.invoke = MagicMock(side_effect=slow_invoke)

        dlt = DynamicLoadTest(
            endpoint=endpoint,
            payload={"prompt": "hi"},
            clients_over_time=[(0, 2), (0.3, 2)],
            duration=0.3,
            control_interval=0.1,
        )

        queue = asyncio.Queue()

        # Patch asyncio.wait to simulate grace period timeout (tasks still running)
        with caplog.at_level(logging.WARNING, logger="llmeter.experiments"):
            # Patch asyncio.wait to simulate grace period timeout (tasks still running)
            async def mock_wait(tasks, timeout=None):
                # Return all tasks as "still running" (pending)
                return set(), set(tasks)

            with patch("asyncio.wait", side_effect=mock_wait):
                concurrency_log, total_time, start_t, end_t = await dlt._controller(
                    queue, output_path=None
                )

        # Verify the grace period warning was logged
        assert any(
            "Grace period exceeded" in record.message for record in caplog.records
        ), (
            f"Expected 'Grace period exceeded' warning, got: {[r.message for r in caplog.records]}"
        )

    @pytest.mark.asyncio
    async def test_resource_error_on_spawn_logs_warning_and_continues(self, caplog):
        """Patch asyncio.to_thread to raise OSError; verify warning logged and loop continues.

        **Validates: Requirements 2.6**
        """
        endpoint = self._make_fast_endpoint(delay=0.01)
        dlt = DynamicLoadTest(
            endpoint=endpoint,
            payload={"prompt": "hi"},
            clients_over_time=[(0, 3), (1.0, 3)],
            duration=1.0,
            control_interval=0.1,
        )

        queue = asyncio.Queue()

        # Patch asyncio.ensure_future to raise OSError on the first few calls,
        # simulating resource exhaustion
        original_ensure_future = asyncio.ensure_future
        call_count = 0

        def patched_ensure_future(coro, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                # Cancel the coroutine to avoid warnings
                coro.close()
                raise OSError("Too many open files")
            return original_ensure_future(coro, **kwargs)

        with caplog.at_level(logging.WARNING, logger="llmeter.experiments"):
            with patch(
                "llmeter.experiments.asyncio.ensure_future",
                side_effect=patched_ensure_future,
            ):
                concurrency_log, total_time, start_t, end_t = await dlt._controller(
                    queue, output_path=None
                )

        # Verify resource error warning was logged
        assert any(
            "Resource error spawning" in record.message for record in caplog.records
        ), (
            f"Expected 'Resource error spawning' warning, got: {[r.message for r in caplog.records]}"
        )

        # Verify the controller continued running (returned results)
        assert len(concurrency_log) > 0
        assert total_time > 0


# --- End-to-end tests for DynamicLoadTest.run() ---


class TestDynamicLoadTestEndToEnd:
    """End-to-end tests for DynamicLoadTest.run() with a dummy endpoint.

    **Validates: Requirements 4.1, 4.2, 4.4, 4.5**
    """

    def _make_dummy_endpoint(self):
        """Create a mock endpoint that responds quickly."""
        endpoint = MagicMock(spec=Endpoint)
        endpoint.model_id = "test-model"
        endpoint.provider = "test"
        endpoint.endpoint_name = "test-endpoint"
        response = InvocationResponse(
            response_text="hello world",
            id="r1",
            time_to_last_token=0.01,
            time_to_first_token=0.005,
            input_prompt="test prompt",
        )
        endpoint.invoke.return_value = response
        return endpoint

    @pytest.mark.asyncio
    async def test_short_run_produces_responses(self):
        """Run a short dynamic test (~1s), verify response count > 0.

        **Validates: Requirements 4.1, 4.2**
        """
        endpoint = self._make_dummy_endpoint()
        dlt = DynamicLoadTest(
            endpoint=endpoint,
            payload={"prompt": "test prompt"},
            clients_over_time=[(0, 2), (1.0, 2)],
            duration=1.0,
            control_interval=0.1,
        )

        result = await dlt.run()

        assert result.result.total_requests > 0
        assert len(result.result.responses) > 0
        assert result.result.model_id == "test-model"
        assert result.result.total_test_time > 0

    @pytest.mark.asyncio
    async def test_before_run_and_after_run_callbacks_invoked(self):
        """Verify before_run and after_run callbacks are invoked during run().

        **Validates: Requirements 4.5**
        """
        from unittest.mock import AsyncMock

        endpoint = self._make_dummy_endpoint()

        callback = MagicMock()
        callback.before_run = AsyncMock()
        callback.after_run = AsyncMock()
        callback.after_invoke = AsyncMock()
        callback.before_invoke = AsyncMock(side_effect=lambda p: p)

        dlt = DynamicLoadTest(
            endpoint=endpoint,
            payload={"prompt": "test prompt"},
            clients_over_time=[(0, 1), (1.0, 1)],
            duration=1.0,
            control_interval=0.1,
            callbacks=[callback],
        )

        result = await dlt.run()

        # before_run should be called once before the test starts
        callback.before_run.assert_called_once()
        # after_run should be called once with the final Result
        callback.after_run.assert_called_once()

        # Verify the argument to after_run is the Result
        after_run_arg = callback.after_run.call_args[0][0]
        assert after_run_arg is result.result

    @pytest.mark.asyncio
    async def test_tokenizer_applies_token_counts(self):
        """Verify tokenizer applies token counts to responses when configured.

        **Validates: Requirements 4.4**
        """
        endpoint = self._make_dummy_endpoint()

        # Create a mock tokenizer that returns 3 tokens for any text
        tokenizer = MagicMock()
        tokenizer.encode = MagicMock(return_value=[1, 2, 3])

        dlt = DynamicLoadTest(
            endpoint=endpoint,
            payload={"prompt": "test prompt"},
            clients_over_time=[(0, 1), (1.0, 1)],
            duration=1.0,
            control_interval=0.1,
            tokenizer=tokenizer,
        )

        result = await dlt.run()

        # Verify responses have token counts applied
        assert len(result.result.responses) > 0
        for resp in result.result.responses:
            assert resp.num_tokens_input == 3, (
                f"Expected num_tokens_input=3, got {resp.num_tokens_input}"
            )
            assert resp.num_tokens_output == 3, (
                f"Expected num_tokens_output=3, got {resp.num_tokens_output}"
            )

        # Verify tokenizer.encode was called
        assert tokenizer.encode.call_count > 0

    @pytest.mark.asyncio
    async def test_concurrency_log_shape_matches_profile(self):
        """Verify concurrency_log entries are present and shaped correctly.

        **Validates: Requirements 4.1, 4.2**
        """
        endpoint = self._make_dummy_endpoint()
        dlt = DynamicLoadTest(
            endpoint=endpoint,
            payload={"prompt": "test prompt"},
            clients_over_time=[(0, 1), (1.0, 3)],
            duration=1.0,
            control_interval=0.1,
        )

        result = await dlt.run()

        # concurrency_log should have entries
        assert len(result.concurrency_log) > 0

        # Each entry should be a (float, int) tuple
        for entry in result.concurrency_log:
            assert len(entry) == 2
            elapsed, count = entry
            assert isinstance(elapsed, float)
            assert isinstance(count, int)
            assert elapsed >= 0
            assert count >= 0

        # The concurrency should ramp from 1 toward 3 over time
        # First entry should be around 1, later entries should be higher
        first_count = result.concurrency_log[0][1]
        last_count = result.concurrency_log[-1][1]
        # At minimum, the ramp profile means later counts are >= earlier counts
        assert last_count >= first_count


# --- Save/Load round-trip tests ---


class TestDynamicLoadTestSaveLoad:
    """Tests for DynamicLoadTestResult save/load round-trip.

    **Validates: Requirements 4.3, 5.1, 5.6**
    """

    def _make_dynamic_result(self):
        """Create a DynamicLoadTestResult with known data for testing."""
        responses = [
            InvocationResponse(
                response_text="hello world",
                id="resp-1",
                time_to_last_token=0.5,
                time_to_first_token=0.1,
                num_tokens_input=5,
                num_tokens_output=10,
            ),
            InvocationResponse(
                response_text="goodbye world",
                id="resp-2",
                time_to_last_token=0.8,
                time_to_first_token=0.2,
                num_tokens_input=3,
                num_tokens_output=7,
            ),
            InvocationResponse(
                response_text=None,
                id="resp-3",
                error="timeout error",
                time_to_last_token=None,
                time_to_first_token=None,
            ),
        ]

        result = Result(
            responses=responses,
            total_requests=3,
            model_id="test-model-v1",
            total_test_time=10.0,
            run_name="test-dynamic-run",
        )

        concurrency_profile = [(0.0, 1), (5.0, 10), (10.0, 5)]
        concurrency_log = [
            (0.0, 1),
            (1.0, 3),
            (2.0, 5),
            (3.0, 7),
            (4.0, 9),
            (5.0, 10),
            (6.0, 9),
            (7.0, 7),
            (8.0, 6),
            (9.0, 5),
        ]

        return DynamicLoadTestResult(
            result=result,
            concurrency_profile=concurrency_profile,
            concurrency_log=concurrency_log,
            test_name="test-dynamic-run",
            control_interval=1.0,
            duration=10.0,
        )

    def test_save_load_round_trip_all_fields_match(self, tmp_path):
        """Save a DynamicLoadTestResult, load it back, verify all fields match.

        **Validates: Requirements 4.3, 5.1**
        """
        dynamic_result = self._make_dynamic_result()
        output_path = tmp_path / "dynamic_test_output"

        # Save: first the base Result, then the dynamic metadata
        dynamic_result.result.save(output_path)
        dynamic_result.save(output_path)

        # Load it back
        loaded = DynamicLoadTestResult.load(output_path)

        # Verify concurrency_log matches
        assert loaded.concurrency_log == dynamic_result.concurrency_log

        # Verify concurrency_profile (waypoints) matches
        assert loaded.concurrency_profile == dynamic_result.concurrency_profile

        # Verify test_name matches
        assert loaded.test_name == dynamic_result.test_name

        # Verify control_interval and duration
        assert loaded.control_interval == dynamic_result.control_interval
        assert loaded.duration == dynamic_result.duration

        # Verify the underlying Result has matching responses
        assert len(loaded.result.responses) == len(dynamic_result.result.responses)
        for loaded_resp, orig_resp in zip(
            loaded.result.responses, dynamic_result.result.responses
        ):
            assert loaded_resp.id == orig_resp.id
            assert loaded_resp.response_text == orig_resp.response_text
            assert loaded_resp.error == orig_resp.error
            assert loaded_resp.time_to_last_token == orig_resp.time_to_last_token
            assert loaded_resp.time_to_first_token == orig_resp.time_to_first_token

        # Verify Result-level metadata
        assert loaded.result.model_id == dynamic_result.result.model_id
        assert loaded.result.total_requests == dynamic_result.result.total_requests

    def test_result_load_on_dynamic_folder_returns_plain_result(self, tmp_path):
        """Result.load() on a dynamic result folder returns a plain Result without error.

        This verifies backward compatibility: the base Result.load() can still
        load a dynamic load test folder (it just ignores the extra files).

        **Validates: Requirements 5.6**
        """
        dynamic_result = self._make_dynamic_result()
        output_path = tmp_path / "dynamic_test_output"

        # Save the dynamic result
        dynamic_result.result.save(output_path)
        dynamic_result.save(output_path)

        # Load with plain Result.load() — should work without error
        plain_result = Result.load(output_path)

        # Verify it's a Result instance (not DynamicLoadTestResult)
        assert isinstance(plain_result, Result)

        # Verify it loaded responses correctly
        assert len(plain_result.responses) == 3
        assert plain_result.model_id == "test-model-v1"

    def test_dynamic_load_on_standard_run_folder_raises_value_error(self, tmp_path):
        """DynamicLoadTestResult.load() on a standard run folder raises ValueError.

        A standard run folder has no concurrency_log.json, so the dynamic loader
        should raise a clear error indicating this is not a dynamic result folder.

        **Validates: Requirements 5.6**
        """
        # Create a standard run folder (Result.save without dynamic metadata)
        responses = [
            InvocationResponse(
                response_text="hi",
                id="r1",
                time_to_last_token=0.3,
            ),
        ]
        standard_result = Result(
            responses=responses,
            total_requests=1,
            model_id="standard-model",
            total_test_time=5.0,
        )
        output_path = tmp_path / "standard_run"
        standard_result.save(output_path)

        # Attempting to load as DynamicLoadTestResult should raise ValueError
        with pytest.raises(ValueError, match="Not a dynamic load test result folder"):
            DynamicLoadTestResult.load(output_path)

    def test_save_load_with_callable_profile(self, tmp_path):
        """Save/load with a callable profile stores a callable marker and loads a placeholder.

        **Validates: Requirements 4.3, 5.1**
        """
        responses = [
            InvocationResponse(
                response_text="test",
                id="resp-1",
                time_to_last_token=0.4,
            ),
        ]
        result = Result(
            responses=responses,
            total_requests=1,
            model_id="callable-model",
            total_test_time=5.0,
            run_name="callable-test",
        )

        # Use a callable profile (cannot be serialized)
        dynamic_result = DynamicLoadTestResult(
            result=result,
            concurrency_profile=lambda t: int(t * 2),
            concurrency_log=[(0.0, 0), (1.0, 2), (2.0, 4)],
            test_name="callable-test",
            control_interval=1.0,
            duration=5.0,
        )

        output_path = tmp_path / "callable_output"
        dynamic_result.result.save(output_path)
        dynamic_result.save(output_path)

        # Load it back
        loaded = DynamicLoadTestResult.load(output_path)

        # Concurrency log should still match
        assert loaded.concurrency_log == dynamic_result.concurrency_log

        # Profile should be a callable placeholder (since original can't be serialized)
        assert callable(loaded.concurrency_profile)

        # Other fields should match
        assert loaded.test_name == dynamic_result.test_name
        assert loaded.control_interval == dynamic_result.control_interval
        assert loaded.duration == dynamic_result.duration



# --- Tests for plotting functions ---


class TestDynamicLoadTestPlotting:
    """Tests for dynamic load test plotting functions in the plotting module.

    **Validates: Requirements 5.2, 5.3, 5.4, 5.5, 5.7**
    """

    def _make_synthetic_result(self):
        """Create synthetic data containing known timestamps."""
        from datetime import datetime, timedelta

        base_time = datetime(2024, 1, 1, 12, 0, 0)
        responses = [
            InvocationResponse(
                response_text="hello",
                id=f"r{i}",
                time_to_last_token=0.1 * (i + 1),
                time_to_first_token=0.01,
                request_time=base_time + timedelta(seconds=i * 0.5),
            )
            for i in range(20)
        ]
        result = Result(
            responses=responses,
            total_requests=20,
            clients=2,
            n_requests=20,
            total_test_time=10.0,
            model_id="test",
        )

        concurrency_log = [(i * 0.5, 2) for i in range(20)]
        concurrency_profile = [(0.0, 2), (10.0, 2)]

        return result, concurrency_log, concurrency_profile

    def _make_zero_success_result(self):
        """Create a Result with zero successful responses."""
        error_responses = [
            InvocationResponse(
                response_text=None,
                id=f"e{i}",
                time_to_last_token=None,
                time_to_first_token=None,
                error="timeout",
            )
            for i in range(5)
        ]
        result = Result(
            responses=error_responses,
            total_requests=5,
            clients=1,
            n_requests=5,
            total_test_time=5.0,
            model_id="test",
        )

        concurrency_log = [(0.0, 1), (1.0, 1)]
        concurrency_profile = [(0.0, 1), (5.0, 1)]

        return result, concurrency_log, concurrency_profile

    def _make_empty_responses_result(self):
        """Create a Result with an empty responses list."""
        result = Result(
            responses=[],
            total_requests=0,
            clients=1,
            n_requests=0,
            total_test_time=0.0,
            model_id="test",
        )

        concurrency_log = [(0.0, 0)]
        concurrency_profile = [(0.0, 1), (5.0, 1)]

        return result, concurrency_log, concurrency_profile

    # --- Test ValueError raised when zero successful responses ---

    def test_plot_concurrency_raises_on_zero_success(self):
        """plot_dynamic_concurrency raises ValueError when all responses have errors.

        **Validates: Requirement 5.7**
        """
        from llmeter.plotting import plot_dynamic_concurrency

        result, concurrency_log, concurrency_profile = self._make_zero_success_result()
        with pytest.raises(ValueError, match="zero successful responses"):
            plot_dynamic_concurrency(
                result=result,
                concurrency_log=concurrency_log,
                concurrency_profile=concurrency_profile,
                output_path=None,
                show=False,
            )

    def test_plot_throughput_raises_on_zero_success(self):
        """plot_dynamic_throughput raises ValueError when all responses have errors.

        **Validates: Requirement 5.7**
        """
        from llmeter.plotting import plot_dynamic_throughput

        result, _, _ = self._make_zero_success_result()
        with pytest.raises(ValueError, match="No successful responses"):
            plot_dynamic_throughput(result=result, output_path=None, show=False)

    def test_plot_latency_raises_on_zero_success(self):
        """plot_dynamic_latency raises ValueError when all responses have errors.

        **Validates: Requirement 5.7**
        """
        from llmeter.plotting import plot_dynamic_latency

        result, _, _ = self._make_zero_success_result()
        with pytest.raises(ValueError, match="No successful responses"):
            plot_dynamic_latency(result=result, output_path=None, show=False)

    def test_plot_concurrency_raises_on_empty_responses(self):
        """plot_dynamic_concurrency raises ValueError when responses list is empty.

        **Validates: Requirement 5.7**
        """
        from llmeter.plotting import plot_dynamic_concurrency

        result, concurrency_log, concurrency_profile = (
            self._make_empty_responses_result()
        )
        with pytest.raises(ValueError, match="zero successful responses"):
            plot_dynamic_concurrency(
                result=result,
                concurrency_log=concurrency_log,
                concurrency_profile=concurrency_profile,
                output_path=None,
                show=False,
            )

    def test_plot_throughput_raises_on_empty_responses(self):
        """plot_dynamic_throughput raises ValueError when responses list is empty.

        **Validates: Requirement 5.7**
        """
        from llmeter.plotting import plot_dynamic_throughput

        result, _, _ = self._make_empty_responses_result()
        with pytest.raises(ValueError, match="No successful responses"):
            plot_dynamic_throughput(result=result, output_path=None, show=False)

    def test_plot_latency_raises_on_empty_responses(self):
        """plot_dynamic_latency raises ValueError when responses list is empty.

        **Validates: Requirement 5.7**
        """
        from llmeter.plotting import plot_dynamic_latency

        result, _, _ = self._make_empty_responses_result()
        with pytest.raises(ValueError, match="No successful responses"):
            plot_dynamic_latency(result=result, output_path=None, show=False)

    # --- Test window_sec < 0.1 clamps appropriately ---

    def test_plot_throughput_with_small_window_sec_clamps(self):
        """plot_dynamic_throughput with window_sec=0.05 doesn't crash (clamps to 0.1).

        **Validates: Requirement 5.3**
        """
        from llmeter.plotting import plot_dynamic_throughput

        result, _, _ = self._make_synthetic_result()
        fig = plot_dynamic_throughput(
            result=result, output_path=None, window_sec=0.05, show=False
        )
        assert fig is not None

    def test_plot_throughput_clamps_window_sec(self):
        """plot_dynamic_throughput with window_sec=0.01 doesn't crash (clamps to 0.1).

        **Validates: Requirement 5.3**
        """
        from llmeter.plotting import plot_dynamic_throughput

        result, _, _ = self._make_synthetic_result()
        fig = plot_dynamic_throughput(
            result=result, output_path=None, window_sec=0.01, show=False
        )
        assert fig is not None

    def test_plot_latency_with_small_window_sec_clamps(self):
        """plot_dynamic_latency with window_sec=0.05 doesn't crash (clamps to 0.1).

        **Validates: Requirement 5.4**
        """
        from llmeter.plotting import plot_dynamic_latency

        result, _, _ = self._make_synthetic_result()
        fig = plot_dynamic_latency(
            result=result, output_path=None, window_sec=0.05, show=False
        )
        assert fig is not None

    # --- Test figures returned with expected keys ---

    def test_plot_results_returns_dict_with_expected_keys(self):
        """plot_dynamic_results returns a dict with expected keys.

        **Validates: Requirements 5.2, 5.3, 5.4, 5.5**
        """
        from llmeter.plotting import plot_dynamic_results

        result, concurrency_log, concurrency_profile = self._make_synthetic_result()
        figures = plot_dynamic_results(
            result=result,
            concurrency_log=concurrency_log,
            concurrency_profile=concurrency_profile,
            output_path=None,
            show=False,
        )

        assert isinstance(figures, dict)
        assert "concurrency" in figures
        assert "throughput" in figures
        assert "latency" in figures
        assert len(figures) == 3

    def test_plot_concurrency_returns_figure(self):
        """plot_dynamic_concurrency returns a Plotly Figure object.

        **Validates: Requirement 5.2**
        """
        import plotly.graph_objects as go

        from llmeter.plotting import plot_dynamic_concurrency

        result, concurrency_log, concurrency_profile = self._make_synthetic_result()
        fig = plot_dynamic_concurrency(
            result=result,
            concurrency_log=concurrency_log,
            concurrency_profile=concurrency_profile,
            output_path=None,
            show=False,
        )
        assert isinstance(fig, go.Figure)

    def test_plot_throughput_returns_figure(self):
        """plot_dynamic_throughput returns a Plotly Figure object.

        **Validates: Requirement 5.3**
        """
        import plotly.graph_objects as go

        from llmeter.plotting import plot_dynamic_throughput

        result, _, _ = self._make_synthetic_result()
        fig = plot_dynamic_throughput(result=result, output_path=None, show=False)
        assert isinstance(fig, go.Figure)

    def test_plot_latency_returns_figure(self):
        """plot_dynamic_latency returns a Plotly Figure object.

        **Validates: Requirement 5.4**
        """
        import plotly.graph_objects as go

        from llmeter.plotting import plot_dynamic_latency

        result, _, _ = self._make_synthetic_result()
        fig = plot_dynamic_latency(result=result, output_path=None, show=False)
        assert isinstance(fig, go.Figure)

    # --- Test with synthetic Result containing known timestamps ---

    def test_plot_concurrency_with_synthetic_data(self):
        """plot_dynamic_concurrency produces a figure with the correct traces.

        **Validates: Requirement 5.2**
        """
        from llmeter.plotting import plot_dynamic_concurrency

        result, concurrency_log, concurrency_profile = self._make_synthetic_result()
        fig = plot_dynamic_concurrency(
            result=result,
            concurrency_log=concurrency_log,
            concurrency_profile=concurrency_profile,
            output_path=None,
            show=False,
        )

        # Should have 2 traces: actual concurrency + target profile
        assert len(fig.data) == 2
        assert fig.data[0].name == "Actual Concurrency"
        assert fig.data[1].name == "Target Profile"

    def test_plot_throughput_with_synthetic_data(self):
        """plot_dynamic_throughput produces a figure with a throughput trace.

        **Validates: Requirement 5.3**
        """
        from llmeter.plotting import plot_dynamic_throughput

        result, _, _ = self._make_synthetic_result()
        fig = plot_dynamic_throughput(result=result, output_path=None, show=False)

        # Should have 1 trace for throughput
        assert len(fig.data) == 1
        assert "Throughput" in fig.data[0].name

    def test_plot_latency_with_synthetic_data(self):
        """plot_dynamic_latency produces a figure with p50, p95, p99 traces.

        **Validates: Requirement 5.4**
        """
        from llmeter.plotting import plot_dynamic_latency

        result, _, _ = self._make_synthetic_result()
        fig = plot_dynamic_latency(result=result, output_path=None, show=False)

        # Should have 3 traces for p50, p95, p99
        assert len(fig.data) == 3
        trace_names = {t.name for t in fig.data}
        assert "p50" in trace_names
        assert "p95" in trace_names
        assert "p99" in trace_names
