"""Characterization tests for api/gateway_main.py - the service AS IT IS TODAY.

WHY THIS FILE EXISTS. The gateway is the endpoint a user actually calls, and it
is the ONLY place the measured fusion weight (0.85) is applied at serving time.
Until now nothing pinned that: scripts/tune_fusion_weight.py measured the number
and rakuten_common/fusion.py records it, but no test proved the serving path
uses it, or that it combines the two vectors the way the sweep assumed. A silent
regression there would change every fused prediction and break no test.

Second reason: DEGRADATION. The gateway deliberately answers 200 with one
modality when the other upstream fails, flagged with "degraded": true. That
design only holds if the flag is always set, so the flag is asserted on every
degraded path here.

NO UPSTREAM IS EVER CONTACTED. The `requests` module is replaced inside
api.gateway_main only, never patched globally, so a test cannot depend on
whether image-api and text-api happen to be running on the machine executing it.
Every upstream reply is constructed here.

COUNTERS ARE ASSERTED AS DELTAS, for the same reason as tests/test_text_api.py:
ServiceMetrics instances live for the whole process, so absolute values pass
alone and fail in a full run. Read before, act, read after.
"""
from __future__ import annotations

import numpy as np
import pytest
import requests as real_requests

from conftest import import_service_module

# fastapi is installed by requirements-ci.txt. importorskip keeps this file
# honest if somebody trims the CI subset again: it would skip loudly with a
# reason rather than vanish.
pytest.importorskip("fastapi", reason="fastapi not installed (CI subset)")

from fastapi.testclient import TestClient  # noqa: E402

from rakuten_common import fusion  # noqa: E402
from rakuten_img import config  # noqa: E402

# Imported through the helper because a second api module cannot share the
# global Prometheus registry. See tests/conftest.py for the measurement.
gw = import_service_module("api.gateway_main")

N = config.NUM_CLASSES


# --------------------------------------------------------------------------- #
# Upstream doubles
# --------------------------------------------------------------------------- #
def _vector(winner: int, peak: float = 0.6) -> list[float]:
    """A normalised, NON-degenerate probability vector peaking at `winner`.

    Non-degenerate on purpose: a one-hot vector would make a wrong fusion
    weight produce a right-looking argmax, which is exactly the bug these tests
    have to be able to see.
    """
    rest = (1.0 - peak) / (N - 1)
    vec = [rest] * N
    vec[winner] = peak
    return vec


def _payload(winner: int, *, classes=None, probabilities=None) -> dict:
    """An upstream /predict body. Mirrors the SHAPE both services return."""
    body: dict = {
        "probabilities": _vector(winner) if probabilities is None else probabilities,
    }
    if classes is not False:
        body["canonical_classes"] = (
            list(config.CANONICAL_CLASSES) if classes is None else classes
        )
    return body


class _Resp:
    def __init__(self, payload, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise real_requests.HTTPError(f"{self.status_code} from upstream")

    def json(self):
        return self._payload


class _FakeRequests:
    """Stands in for the `requests` module INSIDE api.gateway_main.

    Replacing the attribute on the gateway module rather than monkeypatching
    the real requests package keeps the blast radius to this one module. It
    must expose RequestException because _upstream_health catches it by that
    name.
    """

    RequestException = real_requests.RequestException

    def __init__(self, image=None, text=None, health=None):
        self._image = image
        self._text = text
        self._health = health
        self.posts: list[tuple[str, dict]] = []
        self.gets: list[str] = []

    @staticmethod
    def _deliver(outcome, url):
        if outcome is None:
            raise AssertionError(f"unexpected upstream call to {url}")
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def post(self, url, **kwargs):
        self.posts.append((url, kwargs))
        target = self._image if url.startswith(gw.IMAGE_API) else self._text
        return self._deliver(target, url)

    def get(self, url, **kwargs):
        self.gets.append(url)
        return self._deliver(self._health, url)


def _install(monkeypatch, **kwargs) -> _FakeRequests:
    fake = _FakeRequests(**kwargs)
    monkeypatch.setattr(gw, "requests", fake)
    return fake


def _sample(name: str, labels: dict) -> float:
    value = gw.METRICS.registry.get_sample_value(name, labels)
    return 0.0 if value is None else value


def _client() -> TestClient:
    return TestClient(gw.app)


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #
def test_gateway_declares_that_it_owns_no_model():
    """The model_info series is published at import, not on first request.

    A dashboard panel listing all three services would otherwise show a gap for
    this one and read as a broken service rather than a coordinator.
    """
    assert _sample("rakuten_model_info", {
        "service": "gateway", "modality": "fusion", "source": "none (coordinator)",
    }) == 1.0


def test_metrics_endpoint_renders_the_prometheus_exposition():
    with _client() as client:
        resp = client.get("/metrics")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    assert "rakuten_model_info" in resp.text


# --------------------------------------------------------------------------- #
# /health
# --------------------------------------------------------------------------- #
def test_health_reports_both_upstreams_when_they_answer(monkeypatch):
    _install(monkeypatch, health=_Resp({"status": "ok"}))
    with _client() as client:
        body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["service"] == "fusion-gateway"
    assert body["num_classes"] == N
    for side, url in (("image", gw.IMAGE_API), ("text", gw.TEXT_API)):
        assert body["upstreams"][side]["url"] == url
        assert body["upstreams"][side]["reachable"] is True
        assert body["upstreams"][side]["status_code"] == 200
        assert body["upstreams"][side]["body"] == {"status": "ok"}


def test_health_is_200_even_when_every_upstream_is_dead(monkeypatch):
    """Deliberate: the gateway itself is healthy and can still serve one
    modality. A 503 here would take the whole service out of a load balancer
    over a downstream problem it can survive."""
    _install(monkeypatch, health=real_requests.ConnectionError("refused"))
    with _client() as client:
        resp = client.get("/health")

    assert resp.status_code == 200
    upstreams = resp.json()["upstreams"]
    for side in ("image", "text"):
        assert upstreams[side]["reachable"] is False
        assert "ConnectionError" in upstreams[side]["error"]


def test_health_treats_a_non_200_upstream_as_unreachable(monkeypatch):
    _install(monkeypatch, health=_Resp({"status": "ok"}, status_code=503))
    with _client() as client:
        image = client.get("/health").json()["upstreams"]["image"]

    assert image["reachable"] is False
    assert image["status_code"] == 503
    # Not parsed: only a 200 body is read, so a 503 error page cannot be
    # mistaken for an upstream health report.
    assert image["body"] is None


def test_health_advertises_the_weight_the_gateway_actually_uses(monkeypatch):
    _install(monkeypatch, health=_Resp({"status": "ok"}))
    with _client() as client:
        advertised = client.get("/health").json()["default_text_weight"]
    assert advertised == fusion.DEFAULT_TEXT_WEIGHT


def test_the_default_weight_is_the_measured_085():
    """Pins the number itself, not just the plumbing.

    0.85 was measured by scripts/tune_fusion_weight.py on the shared validation
    split (0.7945 weighted F1, against 0.7818 for text alone and 0.5468 for
    image alone). If a retrain moves it, the sweep is re-run and this line is
    updated deliberately - it should never drift silently.
    """
    assert fusion.DEFAULT_TEXT_WEIGHT == 0.85


# --------------------------------------------------------------------------- #
# /predict - input validation
# --------------------------------------------------------------------------- #
def test_predict_with_no_modality_at_all_is_400(monkeypatch):
    fake = _install(monkeypatch)
    with _client() as client:
        resp = client.post("/predict")

    assert resp.status_code == 400
    assert "error" in resp.json()
    assert fake.posts == []


def test_a_blank_designation_does_not_count_as_text(monkeypatch):
    """want_text strips before testing, so whitespace is not a modality. A
    request of only spaces must be refused, not sent to the text model."""
    fake = _install(monkeypatch)
    with _client() as client:
        resp = client.post("/predict", data={"designation": "   "})

    assert resp.status_code == 400
    assert fake.posts == []


@pytest.mark.parametrize("top_k", [0, -1, 28, 100])
def test_top_k_outside_the_class_count_is_rejected(monkeypatch, top_k):
    fake = _install(monkeypatch)
    with _client() as client:
        resp = client.post("/predict", data={"designation": "chaise"},
                           params={"top_k": top_k})

    assert resp.status_code == 422
    # Pydantic validates BEFORE the handler runs, so no upstream is contacted
    # for a request that was never going to be answered.
    assert fake.posts == []


@pytest.mark.parametrize("weight", [-0.1, 1.1, 2.0])
def test_a_weight_outside_0_to_1_is_rejected(monkeypatch, weight):
    fake = _install(monkeypatch)
    with _client() as client:
        resp = client.post("/predict", data={"designation": "chaise"},
                           params={"text_weight": weight})

    assert resp.status_code == 422
    assert fake.posts == []


# --------------------------------------------------------------------------- #
# /predict - the fusion arithmetic, the reason this service exists
# --------------------------------------------------------------------------- #
def _both(monkeypatch, image_winner=0, text_winner=1):
    return _install(monkeypatch,
                    image=_Resp(_payload(image_winner)),
                    text=_Resp(_payload(text_winner)))


def test_both_modalities_are_fused_with_the_default_weight(monkeypatch):
    _both(monkeypatch)
    with _client() as client:
        body = client.post("/predict", files={"file": ("a.jpg", b"bytes")},
                           data={"designation": "chaise"}).json()

    expected = fusion.weighted_average(np.array(_vector(0)), np.array(_vector(1)),
                                       text_weight=fusion.DEFAULT_TEXT_WEIGHT)
    assert body["fused"] is True
    assert body["modalities"] == ["image", "text"]
    assert body["text_weight"] == fusion.DEFAULT_TEXT_WEIGHT
    assert body["probabilities"] == pytest.approx(list(expected))
    assert "degraded" not in body
    assert "errors" not in body


def test_a_weight_override_changes_the_arithmetic(monkeypatch):
    _both(monkeypatch)
    with _client() as client:
        body = client.post("/predict", files={"file": ("a.jpg", b"bytes")},
                           data={"designation": "chaise"},
                           params={"text_weight": 0.25}).json()

    expected = fusion.weighted_average(np.array(_vector(0)), np.array(_vector(1)),
                                       text_weight=0.25)
    assert body["text_weight"] == 0.25
    assert body["probabilities"] == pytest.approx(list(expected))


@pytest.mark.parametrize("weight,winner", [(0.0, 0), (1.0, 1)])
def test_the_extreme_weights_reduce_to_one_modality(monkeypatch, weight, winner):
    """weight 0 must be the image vector exactly and weight 1 the text vector
    exactly. Anything else means the two are the wrong way round - a mistake
    that still produces a plausible-looking answer."""
    _both(monkeypatch)
    with _client() as client:
        body = client.post("/predict", files={"file": ("a.jpg", b"bytes")},
                           data={"designation": "chaise"},
                           params={"text_weight": weight}).json()

    assert body["probabilities"] == pytest.approx(_vector(winner))
    # Still reported as fused: both models ran and both contributed, even
    # though one contributed nothing at this weight.
    assert body["fused"] is True


def test_the_image_upstream_receives_the_uploaded_bytes(monkeypatch):
    fake = _both(monkeypatch)
    with _client() as client:
        client.post("/predict", files={"file": ("product.jpg", b"rawbytes")},
                    data={"designation": "chaise"})

    url, kwargs = next(p for p in fake.posts if p[0].startswith(gw.IMAGE_API))
    assert url == f"{gw.IMAGE_API}/predict"
    assert kwargs["files"]["file"][0] == "product.jpg"
    assert kwargs["files"]["file"][1] == b"rawbytes"
    assert kwargs["timeout"] == gw.UPSTREAM_TIMEOUT_S


def test_the_text_upstream_receives_designation_and_description(monkeypatch):
    fake = _both(monkeypatch)
    with _client() as client:
        client.post("/predict", data={"designation": "chaise",
                                      "description": "en bois"})

    url, kwargs = next(p for p in fake.posts if p[0].startswith(gw.TEXT_API))
    assert url == f"{gw.TEXT_API}/predict"
    assert kwargs["json"] == {"designation": "chaise", "description": "en bois"}


def test_a_missing_description_is_sent_as_an_empty_string(monkeypatch):
    """The text service expects the field, so None would be a 422 from it."""
    fake = _install(monkeypatch, text=_Resp(_payload(1)))
    with _client() as client:
        client.post("/predict", data={"designation": "chaise"})

    _, kwargs = fake.posts[0]
    assert kwargs["json"]["description"] == ""


# --------------------------------------------------------------------------- #
# /predict - single modality is NOT fusion, and says so
# --------------------------------------------------------------------------- #
def test_image_only_returns_the_image_vector_untouched(monkeypatch):
    _install(monkeypatch, image=_Resp(_payload(3)))
    with _client() as client:
        body = client.post("/predict", files={"file": ("a.jpg", b"bytes")}).json()

    assert body["modalities"] == ["image"]
    assert body["fused"] is False
    # None, not 0.0: no weight was applied at all. Reporting a number here
    # would let a caller attribute the measured fusion score to an answer that
    # was never fused.
    assert body["text_weight"] is None
    assert body["probabilities"] == pytest.approx(_vector(3))


def test_text_only_returns_the_text_vector_untouched(monkeypatch):
    _install(monkeypatch, text=_Resp(_payload(4)))
    with _client() as client:
        body = client.post("/predict", data={"designation": "chaise"}).json()

    assert body["modalities"] == ["text"]
    assert body["fused"] is False
    assert body["text_weight"] is None
    assert body["probabilities"] == pytest.approx(_vector(4))


# --------------------------------------------------------------------------- #
# /predict - degradation
# --------------------------------------------------------------------------- #
def test_a_dead_image_upstream_degrades_to_text(monkeypatch):
    _install(monkeypatch,
             image=real_requests.ConnectionError("no route to host"),
             text=_Resp(_payload(2)))
    with _client() as client:
        resp = client.post("/predict", files={"file": ("a.jpg", b"bytes")},
                           data={"designation": "chaise"})

    body = resp.json()
    assert resp.status_code == 200
    assert body["degraded"] is True
    assert "ConnectionError" in body["errors"]["image"]
    assert body["modalities"] == ["text"]
    assert body["fused"] is False
    assert body["text_weight"] is None
    assert body["probabilities"] == pytest.approx(_vector(2))


def test_a_dead_text_upstream_degrades_to_image(monkeypatch):
    _install(monkeypatch,
             image=_Resp(_payload(5)),
             text=real_requests.Timeout("read timed out"))
    with _client() as client:
        body = client.post("/predict", files={"file": ("a.jpg", b"bytes")},
                           data={"designation": "chaise"}).json()

    assert body["degraded"] is True
    assert "Timeout" in body["errors"]["text"]
    assert body["modalities"] == ["image"]


def test_an_http_error_from_an_upstream_is_a_failure_not_a_vector(monkeypatch):
    """raise_for_status is called before the body is read, so a 500 page can
    never be parsed as probabilities."""
    _install(monkeypatch,
             image=_Resp({"probabilities": _vector(0)}, status_code=500),
             text=_Resp(_payload(1)))
    with _client() as client:
        body = client.post("/predict", files={"file": ("a.jpg", b"bytes")},
                           data={"designation": "chaise"}).json()

    assert body["degraded"] is True
    assert "HTTPError" in body["errors"]["image"]


def test_every_requested_modality_failing_is_a_502(monkeypatch):
    _install(monkeypatch,
             image=real_requests.ConnectionError("down"),
             text=real_requests.ConnectionError("down too"))
    with _client() as client:
        resp = client.post("/predict", files={"file": ("a.jpg", b"bytes")},
                           data={"designation": "chaise"})

    assert resp.status_code == 502
    body = resp.json()
    assert set(body["errors"]) == {"image", "text"}
    # No partial answer: a 502 body carries no probabilities to misread.
    assert "probabilities" not in body


def test_a_single_requested_modality_failing_is_also_a_502(monkeypatch):
    _install(monkeypatch, text=real_requests.ConnectionError("down"))
    with _client() as client:
        resp = client.post("/predict", data={"designation": "chaise"})

    assert resp.status_code == 502
    assert set(resp.json()["errors"]) == {"text"}


# --------------------------------------------------------------------------- #
# /predict - the class-order guard
# --------------------------------------------------------------------------- #
def test_an_upstream_claiming_a_different_class_order_is_refused(monkeypatch):
    """THE failure this guard exists for: fusing misordered vectors produces a
    confident, plausible, wrong answer that nothing downstream could detect."""
    scrambled = list(config.CANONICAL_CLASSES)
    scrambled[0], scrambled[1] = scrambled[1], scrambled[0]
    _install(monkeypatch,
             image=_Resp(_payload(0, classes=scrambled)),
             text=_Resp(_payload(1)))
    with _client() as client:
        body = client.post("/predict", files={"file": ("a.jpg", b"bytes")},
                           data={"designation": "chaise"}).json()

    assert body["degraded"] is True
    assert "class order" in body["errors"]["image"]
    assert body["modalities"] == ["text"]


def test_a_vector_of_the_wrong_length_is_refused(monkeypatch):
    _install(monkeypatch,
             image=_Resp(_payload(0, probabilities=[0.5, 0.5])),
             text=_Resp(_payload(1)))
    with _client() as client:
        body = client.post("/predict", files={"file": ("a.jpg", b"bytes")},
                           data={"designation": "chaise"}).json()

    assert body["degraded"] is True
    assert "expected" in body["errors"]["image"]
    assert body["modalities"] == ["text"]


def test_a_reply_with_no_probabilities_field_is_refused(monkeypatch):
    _install(monkeypatch,
             image=_Resp({"prediction": {"code": 10}}),
             text=_Resp(_payload(1)))
    with _client() as client:
        body = client.post("/predict", files={"file": ("a.jpg", b"bytes")},
                           data={"designation": "chaise"}).json()

    assert body["degraded"] is True
    assert "probabilities" in body["errors"]["image"]


def test_an_upstream_that_claims_no_order_is_trusted(monkeypatch):
    """canonical_classes is optional in the check: absent means unverifiable,
    and the length check is all that remains. Pinned because it is the
    difference between a guard and a hard requirement."""
    _install(monkeypatch,
             image=_Resp(_payload(0, classes=False)),
             text=_Resp(_payload(1)))
    with _client() as client:
        body = client.post("/predict", files={"file": ("a.jpg", b"bytes")},
                           data={"designation": "chaise"}).json()

    assert body["fused"] is True
    assert "degraded" not in body


# --------------------------------------------------------------------------- #
# /predict - response shape
# --------------------------------------------------------------------------- #
def test_the_response_carries_the_full_canonical_vector(monkeypatch):
    _both(monkeypatch)
    with _client() as client:
        body = client.post("/predict", files={"file": ("a.jpg", b"bytes")},
                           data={"designation": "chaise"}).json()

    assert body["canonical_classes"] == list(config.CANONICAL_CLASSES)
    assert len(body["probabilities"]) == N


@pytest.mark.parametrize("top_k", [1, 5, 27])
def test_top_k_is_honoured_and_ordered_by_descending_probability(monkeypatch, top_k):
    _both(monkeypatch)
    with _client() as client:
        body = client.post("/predict", files={"file": ("a.jpg", b"bytes")},
                           data={"designation": "chaise"},
                           params={"top_k": top_k}).json()

    probabilities = [entry["probability"] for entry in body["top_k"]]
    assert len(body["top_k"]) == top_k
    assert probabilities == sorted(probabilities, reverse=True)
    assert body["prediction"] == body["top_k"][0]


def test_each_top_k_entry_maps_code_to_label_canonically(monkeypatch):
    _both(monkeypatch)
    with _client() as client:
        body = client.post("/predict", files={"file": ("a.jpg", b"bytes")},
                           data={"designation": "chaise"}).json()

    for entry in body["top_k"]:
        index = list(config.CANONICAL_CLASSES).index(entry["prdtypecode"])
        assert entry["label"] == config.CANONICAL_LABELS[index]
        assert body["probabilities"][index] == pytest.approx(entry["probability"])


def test_per_modality_reports_each_upstream_before_fusion(monkeypatch):
    """The point of the block: it shows what each model said on its own, so a
    disagreement is visible rather than averaged away."""
    _both(monkeypatch, image_winner=2, text_winner=7)
    with _client() as client:
        body = client.post("/predict", files={"file": ("a.jpg", b"bytes")},
                           data={"designation": "chaise"}).json()

    assert set(body["per_modality"]) == {"image", "text"}
    assert body["per_modality"]["image"]["prdtypecode"] == config.CANONICAL_CLASSES[2]
    assert body["per_modality"]["text"]["prdtypecode"] == config.CANONICAL_CLASSES[7]
    assert body["per_modality"]["image"]["probability"] == pytest.approx(0.6)
    assert body["per_modality"]["text"]["label"] == config.CANONICAL_LABELS[7]


def test_per_modality_omits_a_modality_that_failed(monkeypatch):
    _install(monkeypatch,
             image=real_requests.ConnectionError("down"),
             text=_Resp(_payload(1)))
    with _client() as client:
        body = client.post("/predict", files={"file": ("a.jpg", b"bytes")},
                           data={"designation": "chaise"}).json()

    assert set(body["per_modality"]) == {"text"}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def test_a_fused_prediction_is_counted_with_its_winning_code(monkeypatch):
    _both(monkeypatch)
    labels = {"service": "gateway",
              "prdtypecode": str(config.CANONICAL_CLASSES[1])}
    before = _sample("rakuten_predictions_total", labels)
    before_confidence = _sample("rakuten_prediction_confidence_count",
                                {"service": "gateway"})

    with _client() as client:
        body = client.post("/predict", files={"file": ("a.jpg", b"bytes")},
                           data={"designation": "chaise"}).json()

    # The text winner takes the fused argmax at weight 0.85, which is what the
    # counter must record: the code actually returned.
    assert body["prediction"]["prdtypecode"] == config.CANONICAL_CLASSES[1]
    assert _sample("rakuten_predictions_total", labels) - before == 1.0
    assert _sample("rakuten_prediction_confidence_count",
                   {"service": "gateway"}) - before_confidence == 1.0


def test_the_fusion_counter_records_which_modalities_contributed(monkeypatch):
    _both(monkeypatch)
    labels = {"service": "gateway", "modalities": "image+text",
              "fused": "true", "degraded": "false"}
    before = _sample("rakuten_fusion_requests_total", labels)

    with _client() as client:
        client.post("/predict", files={"file": ("a.jpg", b"bytes")},
                    data={"designation": "chaise"})

    assert _sample("rakuten_fusion_requests_total", labels) - before == 1.0


def test_a_degraded_answer_is_labelled_degraded_not_just_single_modality(monkeypatch):
    """This label is what makes "how often are we silently serving one
    modality?" answerable, which is the question the whole degradation design
    exists to expose."""
    _install(monkeypatch,
             image=real_requests.ConnectionError("down"),
             text=_Resp(_payload(1)))
    labels = {"service": "gateway", "modalities": "text",
              "fused": "false", "degraded": "true"}
    failures = {"service": "gateway", "upstream": "image"}
    before = _sample("rakuten_fusion_requests_total", labels)
    before_failures = _sample("rakuten_upstream_failures_total", failures)

    with _client() as client:
        client.post("/predict", files={"file": ("a.jpg", b"bytes")},
                    data={"designation": "chaise"})

    assert _sample("rakuten_fusion_requests_total", labels) - before == 1.0
    assert _sample("rakuten_upstream_failures_total", failures) - before_failures == 1.0


def test_a_deliberate_single_modality_request_is_not_marked_degraded(monkeypatch):
    """Asking for one modality and getting it is not degradation. If both
    landed in the same bucket the metric could not answer its own question."""
    _install(monkeypatch, text=_Resp(_payload(1)))
    labels = {"service": "gateway", "modalities": "text",
              "fused": "false", "degraded": "false"}
    before = _sample("rakuten_fusion_requests_total", labels)

    with _client() as client:
        client.post("/predict", data={"designation": "chaise"})

    assert _sample("rakuten_fusion_requests_total", labels) - before == 1.0


def test_a_502_records_the_failures_but_no_prediction(monkeypatch):
    _install(monkeypatch, text=real_requests.ConnectionError("down"))
    failures = {"service": "gateway", "upstream": "text"}
    confidence = {"service": "gateway"}
    before_failures = _sample("rakuten_upstream_failures_total", failures)
    before_confidence = _sample("rakuten_prediction_confidence_count", confidence)

    with _client() as client:
        assert client.post("/predict", data={"designation": "chaise"}).status_code == 502

    assert _sample("rakuten_upstream_failures_total", failures) - before_failures == 1.0
    # Nothing was predicted, so nothing may be counted as a prediction.
    assert _sample("rakuten_prediction_confidence_count", confidence) == before_confidence


def test_requests_are_counted_under_the_route_template(monkeypatch):
    """Never the raw path: labelling by raw URL would let a caller mint
    unbounded label values and kill the Prometheus server."""
    _install(monkeypatch, text=_Resp(_payload(1)))
    labels = {"service": "gateway", "method": "POST", "path": "/predict",
              "status": "200"}
    before = _sample("rakuten_http_requests_total", labels)

    with _client() as client:
        client.post("/predict", data={"designation": "chaise"})

    assert _sample("rakuten_http_requests_total", labels) - before == 1.0


def test_an_unmatched_route_does_not_create_a_new_label_value():
    labels = {"service": "gateway", "method": "GET", "path": "unmatched",
              "status": "404"}
    before = _sample("rakuten_http_requests_total", labels)

    with _client() as client:
        assert client.get("/no-such-route-12345").status_code == 404

    assert _sample("rakuten_http_requests_total", labels) - before == 1.0
