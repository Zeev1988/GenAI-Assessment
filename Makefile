PYTHON ?= python3
VENV   ?= .venv
BIN    := $(VENV)/bin
PKGS   := common form_extraction medical_chatbot tests

.PHONY: help install venv run run-extraction test test-integration lint typecheck format clean docker-build docker-run

help:
	@echo "Targets:"
	@echo "  install          Create venv and install project + dev deps"
	@echo "  run              Launch the Streamlit UI (Form 283 extractor)"
	@echo "  test             Run unit tests (integration tests require RUN_AZURE_TESTS=1)"
	@echo "  test-integration Run tests including Azure-live integration tests"
	@echo "  lint             Run ruff"
	@echo "  typecheck        Run mypy"
	@echo "  format           Auto-fix with ruff"
	@echo "  docker-build     Build the Streamlit container image"
	@echo "  docker-run       Run the Streamlit container locally (reads .env)"
	@echo "  clean            Remove caches and build artifacts"

venv:
	@test -d $(VENV) || $(PYTHON) -m venv $(VENV)

install: venv
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[dev]"
	@test -f .env || cp .env.example .env
	@echo "Edit .env with your Azure credentials before running."

run: run-extraction

run-extraction:
	$(BIN)/streamlit run form_extraction/frontend/streamlit_app.py

test:
	$(BIN)/pytest

test-integration:
	RUN_AZURE_TESTS=1 $(BIN)/pytest

lint:
	$(BIN)/ruff check $(PKGS)

typecheck:
	$(BIN)/mypy common form_extraction

format:
	$(BIN)/ruff check --fix $(PKGS)
	$(BIN)/ruff format $(PKGS)

docker-build:
	docker build -t genai-assessment-extraction:latest .

docker-run:
	docker run --rm -it -p 8501:8501 --env-file .env genai-assessment-extraction:latest

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist .cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name "*.egg-info" -prune -exec rm -rf {} +
