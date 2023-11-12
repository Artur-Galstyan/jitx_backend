import logging
import os
import pathlib

import equinox as eqx
import jax.numpy as jnp
import matplotlib
import requests
from celery import Celery
from celery.signals import celeryd_init, task_success
from dotenv import load_dotenv

from mnist_cnn.cnn.model.model import Model

logger = logging.getLogger(__name__)
load_dotenv()

db_path = pathlib.Path(__name__).parent.absolute() / "results.db"
CELERY_RESULT_BACKEND = f"db+sqlite:///{db_path}"
CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL")
FASTAPI_URL = os.environ.get("FASTAPI_URL")

assert CELERY_BROKER_URL is not None, "Celery broker URL not set"
assert FASTAPI_URL is not None, "FastAPI URL not set"


celery = Celery(__name__, broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)
celery.conf.update(broker_connection_retry_on_startup=True)
model: Model | None = None


@celeryd_init.connect
def init_celery(**kwargs):
    logger.info("Initializing celery...")
    matplotlib.use("Agg")
    logger.info("Loading model")
    global model
    model_dir = pathlib.Path(__name__).parent.absolute() / "mnist_cnn/models/model.eqx"
    init_model = Model()
    try:
        model = eqx.tree_deserialise_leaves(model_dir, init_model)
    except Exception as e:
        logger.error("Model loading failed", e)
    logger.info("Model loaded")


@task_success.connect
def task_success_handler(result, **kwargs):
    task_id = kwargs["sender"].request.id

    req = requests.post(
        f"{FASTAPI_URL}/predict/webhook",
        json={"prediction": result, "task_id": task_id},
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
    # save in another thread

    prediction = jnp.argmax(model(array, key=None))
    return int(prediction)
