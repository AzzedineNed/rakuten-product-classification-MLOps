#!/usr/bin/env python3
"""Prometheus instrumentation shared by all three services.

WHY THIS IS PURE ASGI AND IMPORTS NO FASTAPI
    requirements-ci.txt deliberately omits fastapi, so a test that imports
    `api/` skips in BOTH CI jobs — it would disappear from CI without a sound,
    which is the exact failure the second CI job exists to prevent. Keeping the
    middleware at the ASGI level (scope / receive / send) means its tests need
    nothing but prometheus-client, so one line in requirements-ci.txt buys real
    CI coverage of the instrumentation. The FastAPI-specific part is then three
    lines per service (add_middleware + a /metrics route).

WHY NOT prometheus-fastapi-instrumentator
    MEASURED against this project's pins: the current release (8.1.0) requires
    starlette>=1.0.0 and upgrades starlette 0.37.2 -> 1.6.0, which breaks
    fastapi==0.111.0 (pip reports the conflict only AFTER performing it).
    Version 7.0.0 is compatible, but it would couple the project's
    fastapi/starlette pins to a third-party release cycle, and this repo already
    carries a pin it genuinely cannot move (numpy <2, required by torch 2.2.2).
    prometheus-client has ZERO dependencies and therefore cannot disturb any
    pin, ever. The metrics that actually matter here
    (per-class prediction counts, which model is serving, fusion degradation)
    are custom and no instrumentator would supply them.

SINGLE PROCESS ASSUMPTION
    Each container runs one uvicorn worker (no --workers in any compose
    `command:`), so the default in-process registry is correct. If a service is
    ever given multiple workers, these counters become per-worker and the
    scrape will return whichever worker answered. Fixing that needs
    prometheus_client's multiprocess mode and a shared directory; do not add
    --workers without doing that.

CARDINALITY
    The `path` label is the ROUTE TEMPLATE ("/items/{item_id}"), never the raw
    URL, and an unmatched request is recorded as "unmatched". This is
    load-bearing: labelling by raw path would let anyone mint unbounded label
    values by requesting random URLs, which is how a Prometheus server gets
    killed. prdtypecode is bounded by the 27 canonical classes.
"""
from __future__ import annotations

import time
from typing import Any, Awaitable, Callable, Iterable, MutableMapping, Optional

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]

# Request latency buckets, in seconds. Chosen for THIS pipeline, not copied
# from a default: a text prediction is a TF-IDF transform plus a dot product
# (single-digit ms), while the image service's first /predict imports torch and
# may resolve a registry version, which was measured in the tens of seconds.
# A bucket set that stops at 10s would put every interesting slow request in
# +Inf and make the p99 unreadable.
DURATION_BUCKETS = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5,
    1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, float("inf"),
)

# Confidence is a probability, so the range is known exactly.
CONFIDENCE_BUCKETS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0)

UNMATCHED = "unmatched"


def route_template(scope: Scope) -> str:
    """Return the matched route TEMPLATE, or "unmatched".

    VERIFIED against starlette 0.37.2: scope["route"] is absent when the
    middleware is entered and is populated by the router only after the
    downstream app has been awaited, so this must be called AFTER the await.
    For a request that matched no route it is never set at all — hence the
    constant fallback rather than scope["path"].
    """
    route = scope.get("route")
    path = getattr(route, "path", None)
    if not isinstance(path, str) or not path:
        return UNMATCHED
    return path


class ServiceMetrics:
    """The metric family for one service.

    Instantiated once per process. Tests pass their own CollectorRegistry so
    each test is isolated; passing None uses the global default registry, which
    also carries prometheus-client's built-in process and GC collectors
    (process_resident_memory_bytes among them — directly useful for checking
    the compose mem_limit values against reality).
    """

    def __init__(self, service: str, registry: Optional[CollectorRegistry] = None):
        self.service = service
        self.registry = REGISTRY if registry is None else registry

        self.requests = Counter(
            "rakuten_http_requests_total",
            "HTTP requests handled, by route template and status.",
            ["service", "method", "path", "status"],
            registry=self.registry,
        )
        self.duration = Histogram(
            "rakuten_http_request_duration_seconds",
            "Wall-clock time to handle an HTTP request.",
            ["service", "method", "path"],
            buckets=DURATION_BUCKETS,
            registry=self.registry,
        )
        self.predictions = Counter(
            "rakuten_predictions_total",
            "Predictions returned, by winning product type code.",
            ["service", "prdtypecode"],
            registry=self.registry,
        )
        self.confidence = Histogram(
            "rakuten_prediction_confidence",
            "Probability assigned to the winning class.",
            ["service"],
            buckets=CONFIDENCE_BUCKETS,
            registry=self.registry,
        )
        self.model_info = Gauge(
            "rakuten_model_info",
            "Always 1. The labels carry which model this service is serving.",
            ["service", "modality", "source"],
            registry=self.registry,
        )
        self.upstream_failures = Counter(
            "rakuten_upstream_failures_total",
            "Upstream calls that failed, by upstream name.",
            ["service", "upstream"],
            registry=self.registry,
        )
        self.fusion_requests = Counter(
            "rakuten_fusion_requests_total",
            "Gateway predictions, by which modalities contributed.",
            ["service", "modalities", "fused", "degraded"],
            registry=self.registry,
        )

    # -- recording -------------------------------------------------------

    def observe_request(self, method: str, path: str, status: int,
                        duration_seconds: float) -> None:
        self.requests.labels(self.service, method, path, str(status)).inc()
        self.duration.labels(self.service, method, path).observe(duration_seconds)

    def observe_prediction(self, prdtypecode: Any, probability: Optional[float] = None) -> None:
        """Record one returned prediction.

        Never raises: a metrics failure must not turn a good prediction into a
        500. The caller is a response path, not a monitoring job.
        """
        try:
            self.predictions.labels(self.service, str(prdtypecode)).inc()
            if probability is not None:
                self.confidence.labels(self.service).observe(float(probability))
        except Exception:  # noqa: BLE001 - see docstring
            pass

    def set_model_info(self, modality: str, source: str) -> None:
        """Publish which model is being served.

        Clears first: this is an info-style gauge, so a stale label set left
        behind after the source changes would leave TWO series both reading 1,
        and the dashboard could not tell which is current.
        """
        try:
            self.model_info.clear()
            self.model_info.labels(self.service, modality, str(source)).set(1)
        except Exception:  # noqa: BLE001
            pass

    def observe_upstream_failure(self, upstream: str) -> None:
        try:
            self.upstream_failures.labels(self.service, upstream).inc()
        except Exception:  # noqa: BLE001
            pass

    def observe_fusion(self, modalities: Iterable[str], fused: bool,
                       degraded: bool) -> None:
        try:
            joined = "+".join(sorted(modalities)) or "none"
            self.fusion_requests.labels(
                self.service, joined, str(bool(fused)).lower(),
                str(bool(degraded)).lower(),
            ).inc()
        except Exception:  # noqa: BLE001
            pass

    # -- exposition ------------------------------------------------------

    def render(self) -> tuple[bytes, str]:
        """Return (body, content_type) for a /metrics response."""
        return generate_latest(self.registry), CONTENT_TYPE_LATEST


class PrometheusMiddleware:
    """Pure-ASGI middleware recording count and latency for every request."""

    def __init__(self, app, metrics: ServiceMetrics):
        self.app = app
        self.metrics = metrics

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # VERIFIED: starlette routes the "lifespan" scope through the user
        # middleware stack, and that scope stays open for the whole process
        # lifetime. Timing it as a request would record one observation of
        # however long the container has been up. Websockets are passed through
        # for the same reason: they are not request/response shaped.
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        start = time.perf_counter()
        captured: dict[str, int] = {}

        async def send_wrapper(message: Message) -> None:
            if message.get("type") == "http.response.start":
                captured["status"] = int(message["status"])
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            # VERIFIED against starlette 0.37.2: when a route handler raises,
            # the exception propagates THROUGH this middleware with no
            # http.response.start ever sent — ServerErrorMiddleware sits above
            # us and turns it into the 500 the client sees. Recording only on a
            # clean return would therefore make every 500 invisible, which is
            # precisely the thing a dashboard exists to show. Hence: finally,
            # and an explicit 500 when nothing was sent.
            status = captured.get("status", 500)
            try:
                self.metrics.observe_request(
                    method=scope.get("method", "GET"),
                    path=route_template(scope),
                    status=status,
                    duration_seconds=time.perf_counter() - start,
                )
            except Exception:  # noqa: BLE001
                # Instrumentation must never convert a served request into an
                # error, nor mask the exception already in flight.
                pass
