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

.PHONY: pub
pub:
	quarto render --execute

.PHONY: markdown
markdown:
	jupyter nbconvert --to markdown notebook.ipynb --config nbconvert_config.py
	zip -r notebook.zip notebook.md notebook_files/
	rm -rf notebook.md notebook_files/
