import logging
import os
import pathlib

import equinox as eqx
import jax.numpy as jnp
import matplotlib
import requests
from celery import Celery
from celery.signals import celeryd_init, task_success, task_postrun
from dotenv import load_dotenv

from mnist_cnn.cnn.model.model import Model

logger = logging.getLogger(__name__)
load_dotenv()

CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND")
CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL")
FASTAPI_URL = os.environ.get("FASTAPI_URL")

assert CELERY_BROKER_URL is not None, "Celery broker URL not set"
assert FASTAPI_URL is not None, "FastAPI URL not set"
assert CELERY_RESULT_BACKEND is not None, "Celery result backend not set"


celery = Celery(__name__, broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)
celery.conf.update(broker_connection_retry_on_startup=True)
model: Model | None = None


@celeryd_init.connect
def init_celery(**kwargs):
    print("Initializing celery...")
    print("Loading model")
    global model
    model_dir = pathlib.Path(__name__).parent.absolute() / "mnist_cnn/models/model.eqx"
    init_model = Model()
    try:
        model = eqx.tree_deserialise_leaves(model_dir, init_model)
    except Exception as e:
        print("Model loading failed", e)
    print("Model loaded")


# @task_postrun.connect
def task_postrun_handler(
    task_id,
    task,
    retval,
    state,
    **kwargs,
):
    logger.info(f"Task {task_id} succeeded; sending webhook")
    req = requests.post(
        f"{FASTAPI_URL}/predict/webhook",
        headers={"Content-Type": "application/json"},
        json={"prediction": retval, "task_id": task_id},
    )
    if req.status_code != 200:
        logger.error(
            f"Failed to send webhook for task {task_id}, got {req.status_code}"
        )
    else:
        logger.info(f"Webhook sent for task {task_id}")


@celery.task(name="predict_number")
def predict_number(array: list):
    global model
    if not model:
        raise Exception("Model not loaded")
    array = jnp.array(array, dtype=float).reshape(1, 28, 28)
    prediction = jnp.argmax(model(array, key=None))
    return int(prediction)
