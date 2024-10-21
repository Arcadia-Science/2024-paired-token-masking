PUB_DIR := ./pub

.PHONY: lint
lint:
	ruff check --exit-zero .
	ruff format --check .

.PHONY: format
format:
	ruff check --fix .
	ruff format .

.PHONY: typecheck
typecheck:
	pyright --project pyproject.toml .

.PHONY: pre-commit
pre-commit:
	pre-commit run --all-files

.PHONY: test
test:
	pytest -v .

# By default, `execute` will just execute notebook.ipynb (overwriting current
# state of notebook) using `jupyter nbconvert`. If the analysis includes prior
# setup, `execute` should be updated to include these steps.
.PHONY: execute
execute:
	modal deploy src/analysis/modal_esm/predict.py
	jupyter nbconvert --to notebook --execute --inplace notebook.ipynb

.PHONY: pub
pub:
	$(MAKE) -C pub/ clean-and-build-html
	$(MAKE) -C pub/ view-html
