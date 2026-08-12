CHECK_FILES := $(shell git ls-files '*.py' '*.ipynb')

.PHONY: check lint format-check type-check

check: lint format-check type-check

lint:
	uv run ruff check $(CHECK_FILES)

format-check:
	uv run ruff format --check $(CHECK_FILES)

type-check:
	uv run ty check $(CHECK_FILES) --python .venv/ --error-on-warning
