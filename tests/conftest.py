import importlib
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def import_service_module(module_name: str):
    """Import an api.* service module against a PRIVATE Prometheus registry.

    MEASURED, not assumed: importing a second service into one process fails
    outright.

        DuplicateTimeseries: Duplicated timeseries in CollectorRegistry:
        {'rakuten_http_requests_created', 'rakuten_http_requests',
         'rakuten_http_requests_total'}

    ServiceMetrics defaults to prometheus_client's global REGISTRY and every
    service builds one instance at import, so the first api module imported
    claims those names for the whole process. In production this never bites -
    each service is its own container, its own process, one ServiceMetrics. It
    bites only here, where pytest imports every test module into one
    interpreter.

    The fix stays on the test side ON PURPOSE. Giving each service a private
    registry in production would drop prometheus-client's built-in process and
    GC collectors from /metrics, process_resident_memory_bytes among them,
    which is what the compose mem_limit values were measured against. So the
    swap is temporary and the global is restored: the module under test keeps a
    registry nobody else writes to, and tests/test_text_api.py, which imports
    api.text_main directly, still gets the global one it was written against.

    Counters are still per-module and still accumulate across a file, so tests
    must read deltas rather than absolute values.
    """
    from prometheus_client import CollectorRegistry

    from rakuten_common import observability

    # api/ is a package at the repo ROOT, and only src/ is on the path above.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    saved = observability.REGISTRY
    observability.REGISTRY = CollectorRegistry()
    try:
        return importlib.import_module(module_name)
    finally:
        observability.REGISTRY = saved


@pytest.fixture(autouse=True)
def _isolate_mlflow_global_state():
    """Undo the process-global state MLflow leaves behind, so no test file can
    make another one order-dependent.

    THREE pieces leak, all MEASURED rather than assumed:

      1. mlflow.set_tracking_uri() does not only set a module global - it also
         WRITES MLFLOW_TRACKING_URI INTO os.environ. So it beats the environment
         variable permanently, and monkeypatch.setenv/.delenv on that variable
         is inert in any later test in the same process.
      2. mlflow.set_experiment() likewise exports MLFLOW_EXPERIMENT_ID. Carried
         into a test pointing at a different database, a later bare start_run()
         dies with "No Experiment with id=N exists".
      3. Both are cached in private module globals as well as the environment,
         and the two must agree.

    Restoring by CALLING the setters does not work: with nothing previously set,
    get_tracking_uri() returns a COMPUTED default (sqlite:///<cwd>/mlflow.db,
    NOT ./mlruns, verified on MLflow 3.14.0), and feeding that back turns an
    unset variable into a set one pointing at the repo root. So the snapshot is
    of the raw state, restored by assignment.

    test_registry.py, test_text_registry.py, test_promote.py and
    test_img_registry.py all call set_tracking_uri; until this existed the suite
    passed by alphabetical luck. Found when the image tracking tests broke
    test_registry.py.
    """
    try:
        import mlflow  # noqa: F401
        from mlflow.tracking import fluent
        from mlflow.tracking._tracking_service import utils
    except ImportError:  # CI subset - mlflow absent, nothing global to restore
        yield
        return

    env = {k: os.environ.get(k) for k in
           ("MLFLOW_TRACKING_URI", "MLFLOW_EXPERIMENT_ID", "MLFLOW_EXPERIMENT_NAME")}
    uri, experiment = utils._tracking_uri, fluent._active_experiment_id
    yield
    utils._tracking_uri, fluent._active_experiment_id = uri, experiment
    for key, value in env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
