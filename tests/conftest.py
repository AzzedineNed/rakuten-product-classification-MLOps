import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(autouse=True)
def _isolate_mlflow_global_state():
    """Undo the process-global state MLflow leaves behind, so no test file can
    make another one order-dependent.

    THREE pieces leak, all MEASURED rather than assumed:

      1. mlflow.set_tracking_uri() does not only set a module global — it also
         WRITES MLFLOW_TRACKING_URI INTO os.environ. So it beats the environment
         variable permanently, and monkeypatch.setenv/.delenv on that variable
         is inert in any later test in the same process.
      2. mlflow.set_experiment() likewise exports MLFLOW_EXPERIMENT_ID. Carried
         into a test pointing at a different database, a later bare start_run()
         dies with "No Experiment with id=N exists".
      3. Both are cached in private module globals as well as the environment,
         and the two must agree.

    Restoring by CALLING the setters does not work: with nothing previously set,
    get_tracking_uri() returns a COMPUTED default (sqlite:///<cwd>/mlflow.db —
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
    except ImportError:  # CI subset — mlflow absent, nothing global to restore
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
