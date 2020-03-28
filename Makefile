SHELL := /bin/bash

test:
	pytest --cov=src tests/

install:
	python3 -m venv venv
	source venv/bin/activate; pip install -r requirements-dev.txt; pre-commit install
