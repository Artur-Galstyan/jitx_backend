To run the worker:
```python
poetry run python3 -m  celery -A jitx_backend.worker.celery worker --loglevel=INFO 
```
