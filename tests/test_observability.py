"""Tests for rakuten_common.observability.

DELIBERATELY FASTAPI-FREE. requirements-ci.txt does not install fastapi, so a
test importing api/ would skip in BOTH CI jobs and vanish from CI silently.
Driving the middleware at the ASGI level keeps it covered for the cost of one
line (prometheus-client) in requirements-ci.txt.

Each test builds its own CollectorRegistry, so no test can see another's
counters.
"""
from __future__ import annotations

import asyncio

import pytest

from rakuten_common.observability import (
    UNMATCHED,
    PrometheusMiddleware,
    ServiceMetrics,
    route_template,
)


class FakeRoute:
    """Stands in for a starlette APIRoute: only .path is read."""

    def __init__(self, path):
        self.path = path


def make_app(status=200, raises=None, body=b"ok"):
    """A minimal ASGI app that optionally raises before responding."""

    async def app(scope, receive, send):
        if raises is not None:
            raise raises
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"text/plain")]})
        await send({"type": "http.response.body", "body": body})

    return app


def drive(middleware, scope, receive=None):
    """Run one ASGI call to completion, collecting the sent messages."""
    sent = []

    async def send(message):
        sent.append(message)

    async def default_receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    asyncio.run(middleware(scope, receive or default_receive, send))
    return sent


def http_scope(path="/predict", method="POST", route_path="/predict"):
    scope = {"type": "http", "path": path, "method": method}
    if route_path is not None:
        # The router sets this DURING the downstream call. Pre-seeding it is
        # equivalent for the middleware, which only reads it afterwards.
        scope["route"] = FakeRoute(route_path)
    return scope


# --------------------------------------------------------------------------
# route_template
# --------------------------------------------------------------------------

def test_route_template_returns_the_template_not_the_raw_path():
    scope = {"type": "http", "path": "/items/42", "route": FakeRoute("/items/{item_id}")}
    assert route_template(scope) == "/items/{item_id}"


@pytest.mark.parametrize("scope", [
    {"type": "http", "path": "/nope"},                       # no route at all
    {"type": "http", "path": "/nope", "route": None},        # route is None
    {"type": "http", "path": "/nope", "route": FakeRoute("")},    # empty path
    {"type": "http", "path": "/nope", "route": FakeRoute(None)},  # non-str path
])
def test_route_template_falls_back_to_the_constant(scope):
    """THE CARDINALITY GUARD. An unmatched request must never contribute its
    raw URL as a label value, or anyone could mint unbounded label values by
    requesting random paths."""
    assert route_template(scope) == UNMATCHED
    assert "nope" not in route_template(scope)


def test_unmatched_paths_collapse_to_one_series():
    metrics = ServiceMetrics("image", registry=__import__(
        "prometheus_client").CollectorRegistry())
    app = PrometheusMiddleware(make_app(status=404), metrics)
    for i in range(5):
        drive(app, {"type": "http", "path": f"/random/{i}", "method": "GET"})

    value = metrics.registry.get_sample_value(
        "rakuten_http_requests_total",
        {"service": "image", "method": "GET", "path": UNMATCHED, "status": "404"},
    )
    assert value == 5
    names = {s.labels.get("path") for m in metrics.registry.collect() for s in m.samples
             if "path" in s.labels}
    assert names == {UNMATCHED}


# --------------------------------------------------------------------------
# middleware
# --------------------------------------------------------------------------

def fresh(service="text"):
    from prometheus_client import CollectorRegistry
    return ServiceMetrics(service, registry=CollectorRegistry())


def test_records_count_and_latency_for_a_normal_request():
    metrics = fresh()
    app = PrometheusMiddleware(make_app(status=200), metrics)
    sent = drive(app, http_scope())

    assert sent[0]["status"] == 200          # response still delivered intact
    assert sent[1]["body"] == b"ok"
    assert metrics.registry.get_sample_value(
        "rakuten_http_requests_total",
        {"service": "text", "method": "POST", "path": "/predict", "status": "200"},
    ) == 1
    assert metrics.registry.get_sample_value(
        "rakuten_http_request_duration_seconds_count",
        {"service": "text", "method": "POST", "path": "/predict"},
    ) == 1


def test_a_raising_handler_is_recorded_as_500_and_the_exception_still_propagates():
    """VERIFIED against starlette 0.37.2: when a handler raises, the exception
    passes through this middleware with NO http.response.start sent - the 500
    the client sees is produced by ServerErrorMiddleware ABOVE us. Recording
    only on a clean return would therefore make every 500 invisible."""
    metrics = fresh()
    app = PrometheusMiddleware(make_app(raises=RuntimeError("kaboom")), metrics)

    with pytest.raises(RuntimeError, match="kaboom"):
        drive(app, http_scope(method="GET"))

    assert metrics.registry.get_sample_value(
        "rakuten_http_requests_total",
        {"service": "text", "method": "GET", "path": "/predict", "status": "500"},
    ) == 1


def test_non_http_scopes_pass_through_unrecorded():
    """The lifespan scope is routed through user middleware and stays open for
    the whole process lifetime; timing it as a request would record one
    observation of the container's entire uptime."""
    metrics = fresh()
    seen = []

    async def app(scope, receive, send):
        seen.append(scope["type"])

    mw = PrometheusMiddleware(app, metrics)
    drive(mw, {"type": "lifespan"})
    drive(mw, {"type": "websocket", "path": "/ws"})

    assert seen == ["lifespan", "websocket"]
    total = [s for m in metrics.registry.collect() for s in m.samples
             if s.name == "rakuten_http_requests_total"]
    assert total == []


def test_status_is_taken_from_the_response_not_assumed():
    metrics = fresh()
    app = PrometheusMiddleware(make_app(status=503), metrics)
    drive(app, http_scope(method="GET"))
    assert metrics.registry.get_sample_value(
        "rakuten_http_requests_total",
        {"service": "text", "method": "GET", "path": "/predict", "status": "503"},
    ) == 1


# --------------------------------------------------------------------------
# domain metrics
# --------------------------------------------------------------------------

def test_predictions_are_counted_per_class():
    metrics = fresh()
    metrics.observe_prediction(2583, 0.91)
    metrics.observe_prediction(2583, 0.42)
    metrics.observe_prediction(1180, 0.30)

    get = metrics.registry.get_sample_value
    assert get("rakuten_predictions_total", {"service": "text", "prdtypecode": "2583"}) == 2
    assert get("rakuten_predictions_total", {"service": "text", "prdtypecode": "1180"}) == 1
    assert get("rakuten_prediction_confidence_count", {"service": "text"}) == 3


def test_set_model_info_clears_the_previous_source():
    """An info gauge that kept its old label set would leave TWO series reading
    1 and the dashboard could not say which model is actually serving."""
    metrics = fresh()
    metrics.set_model_info("text", "local:logistic_regression.pkl")
    metrics.set_model_info("text", "registry:rakuten-text-classifier@production/v1")

    get = metrics.registry.get_sample_value
    assert get("rakuten_model_info", {"service": "text", "modality": "text",
                                      "source": "local:logistic_regression.pkl"}) is None
    assert get("rakuten_model_info",
               {"service": "text", "modality": "text",
                "source": "registry:rakuten-text-classifier@production/v1"}) == 1


def test_fusion_labels_describe_what_actually_contributed():
    metrics = fresh("gateway")
    metrics.observe_fusion(["text", "image"], fused=True, degraded=False)
    metrics.observe_fusion(["text"], fused=False, degraded=True)

    get = metrics.registry.get_sample_value
    assert get("rakuten_fusion_requests_total",
               {"service": "gateway", "modalities": "image+text",
                "fused": "true", "degraded": "false"}) == 1
    assert get("rakuten_fusion_requests_total",
               {"service": "gateway", "modalities": "text",
                "fused": "false", "degraded": "true"}) == 1


def test_upstream_failures_are_counted_per_upstream():
    metrics = fresh("gateway")
    metrics.observe_upstream_failure("image")
    metrics.observe_upstream_failure("image")
    metrics.observe_upstream_failure("text")

    get = metrics.registry.get_sample_value
    assert get("rakuten_upstream_failures_total",
               {"service": "gateway", "upstream": "image"}) == 2
    assert get("rakuten_upstream_failures_total",
               {"service": "gateway", "upstream": "text"}) == 1


def test_recording_never_raises_on_bad_input():
    """A metrics failure must not turn a served prediction into a 500."""
    metrics = fresh()
    metrics.observe_prediction(2583, float("nan"))   # NaN is a valid float
    metrics.observe_prediction(None, "not-a-number")  # probability unusable
    assert metrics.registry.get_sample_value(
        "rakuten_predictions_total", {"service": "text", "prdtypecode": "None"}) == 1


def test_render_returns_prometheus_text_format():
    metrics = fresh()
    metrics.observe_prediction(1180, 0.5)
    body, content_type = metrics.render()

    assert isinstance(body, bytes)
    assert "text/plain" in content_type
    text = body.decode()
    assert "rakuten_predictions_total" in text
    assert 'prdtypecode="1180"' in text
    assert "# HELP rakuten_predictions_total" in text
