.PHONY: format lint

format:
	uv run black .
	uv run isort .

lint:
	uv run black --check .
	uv run isort --check .
