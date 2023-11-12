# Dockerfile
FROM python:3.11-slim-buster AS app

WORKDIR /app
COPY . /app

RUN apt-get update && \
    apt-get install -y  gcc musl-dev libpq-dev curl ca-certificates curl gnupg && \
    pip3 install poetry

RUN apt-get update

RUN poetry config virtualenvs.create false

# Install any needed packages specified in pyproject.toml
RUN poetry install


CMD ["poetry", "run", "start"]
