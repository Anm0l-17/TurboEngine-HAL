.PHONY: install test lint demo api dashboard
install:
	python -m pip install -e ".[dev,api,dashboard,reports]"
test:
	python -m pytest --cov=src --cov-report=term-missing
lint:
	python -m ruff check .
	python -m black --check .
demo:
	python pipeline.py demo
api:
	python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000
dashboard:
	python -m streamlit run src/viz/dashboard.py
