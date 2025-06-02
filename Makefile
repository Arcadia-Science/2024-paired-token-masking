.PHONY: lint
lint:
	ruff check --exit-zero .
	ruff format --check .

.PHONY: format
format:
	ruff check --fix .
	ruff format .

.PHONY: pre-commit
pre-commit:
	pre-commit run --all-files

.PHONY: test
test:
	pytest -v .

.PHONY: execute
execute:
	# Project-wide render without executing code cells. Instead, rely on
	# pre-computed results present in _freeze/
	quarto render

	# Now, render `index.ipynb`, with code execution. This will populate
	# _freeze/index/ with pre-computed results.
	quarto render index.ipynb --execute

.PHONY: preview
preview:
	quarto preview

.PHONY: bump-version
bump-version:
	python _bump_version.py

.PHONY: install
install:
	git submodule update --init --recursive
	pip install -e .
	pip install third_party/py-mfdca
	# Add getcontacts to PATH if it's not already there
	@if ! echo $$PATH | grep -q "$$(pwd)/third_party/getcontacts"; then \
		echo "Adding getcontacts to PATH..."; \
		echo 'export PATH=$$PATH:$$(pwd)/third_party/getcontacts' >> ~/.bashrc; \
		echo "⚠️ Run 'source ~/.bashrc' to activate the change."; \
		echo "And then reactivate your conda environment"; \
	fi
