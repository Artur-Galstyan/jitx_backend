To run the worker:
```python
poetry run python3 -m  celery -A mnist_cnn.worker.celery worker --loglevel=INFO 
```
