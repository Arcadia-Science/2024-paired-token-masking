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

# By default, `run-notebook` will just execute notebook.ipynb (overwriting
# current state of notebook) using `jupyter nbconvert`. If the analysis
# includes additional setup, `run-notebook` should be updated to include these
# steps.
.PHONY: run-notebook
run-notebook:
	modal deploy src/analysis/modal_esm/predict.py
	jupyter nbconvert --to notebook --execute --inplace notebook.ipynb

.PHONY: pub
pub:
	$(MAKE) -C pub/ clean-and-build-html
	$(MAKE) -C pub/ view-html


.PHONY: markdown
markdown:
	jupyter nbconvert --to markdown notebook.ipynb --config nbconvert_config.py
	zip -r notebook.zip notebook.md notebook_files/
	rm -rf notebook.md notebook_files/
