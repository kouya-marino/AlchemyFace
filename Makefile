.PHONY: help install requirements clean format check_formatted check_type lint test test_all build publish doc licenses_list check_license
.DEFAULT_GOAL := install

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = alchemyface
SRC = src/alchemyface/
TEST = tests/
PATH_COV_BADGE = docs/coverage.svg

# Poetry only recognises the pyenv virtualenv when VIRTUAL_ENV is exported.
# `pyenv local` activates via shims and never sets it, so without this Poetry
# installs into its own interpreter (Homebrew's python@3.14) instead.
# Locally the interpreter comes from the pyenv virtualenv. On CI there is no
# pyenv: GitHub Actions sets CI=true and setup-python already put the right
# interpreter on PATH, so the export is skipped and Poetry installs into that.
# Anywhere else, a missing virtualenv is a mistake worth stopping for.
PYENV_ENV_NAME = alchemyface
VENV := $(shell pyenv prefix $(PYENV_ENV_NAME) 2>/dev/null)

ifneq ($(VENV),)
export VIRTUAL_ENV := $(VENV)
export PATH := $(VENV)/bin:$(PATH)
else ifeq ($(CI),)
$(error pyenv virtualenv '$(PYENV_ENV_NAME)' not found. Run: pyenv virtualenv 3.10.6 $(PYENV_ENV_NAME))
endif

# Point the model-dependent tests at the local weights if they are present.
export ALCHEMYFACE_MODEL_DIR ?= $(PROJECT_DIR)/_local/onnx

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Set up the environment (expects the pyenv virtualenv from .python-version to be active)
install:
	poetry install --with dev,docs

## Install the greeter example's extra dependencies as well
install_greeter:
	poetry install --with dev,docs,greeter

## Install poetry itself
install_poetry:
	brew install poetry

## Export Python dependencies as requirements.txt (needs poetry-plugin-export)
requirements:
	@poetry export -f requirements.txt --without-hashes > requirements.txt \
		|| echo "poetry export unavailable — run: poetry self add poetry-plugin-export"

## Delete all compiled Python files and build artefacts
clean:
	find . -not -path "./.venv/*" -not -path "./_local/*" -type f -name "*.py[co]" -exec rm -rf {} + 2>/dev/null || true
	find . -not -path "./.venv/*" -not -path "./_local/*" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -not -path "./.venv/*" -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist build htmlcov .coverage coverage.xml .hypothesis .pytest_cache .mypy_cache

## Format code using black/isort
format:
	poetry run black $(SRC) $(TEST) --config pyproject.toml
	poetry run isort $(SRC) $(TEST)

## Check that code is already formatted (this is mainly for CI)
check_formatted:
	poetry run black $(SRC) $(TEST) --check --config pyproject.toml
	poetry run isort $(SRC) $(TEST) --check

## Type check using mypy
check_type:
	poetry run mypy $(SRC)

# The linter generates many false positives for test code, so those checks are
# suppressed there. C0116: missing-function-docstring, W0621: redefined-outer-name,
# W0212: protected-access, W0611: unused-import, R0801: duplicate-code
## Lint using pylint
lint:
	poetry run pylint --fail-under 8.0 $(SRC) --exit-zero
	poetry run pylint --fail-under 8.0 -d C0116,W0621,W0212,W0611,R0801 $(TEST) --exit-zero

## Run the fast tests (no models, no camera)
test:
	poetry run pytest $(TEST) -m "not models and not camera" \
		--cov=$(SRC) --cov-branch --cov-report term-missing --cov-report html:./htmlcov \
		--cov-report xml:./coverage.xml --cov-fail-under 80
	poetry run genbadge coverage -i ./coverage.xml -o $(PATH_COV_BADGE)

## Run every test, including those needing the real ONNX weights
test_all:
	poetry run pytest $(TEST) -m "not camera" --cov=$(SRC) --cov-branch --cov-report term-missing

## Build wheel and sdist
build: clean
	poetry build

## Verify the built artefacts, then upload to PyPI
publish: build
	poetry run python -m twine check dist/*
	poetry publish

## Browse documentation
doc:
	poetry run mkdocs serve

## Create the licenses list
licenses_list:
	poetry run pip-licenses

## Fail if any dependency carries a copyleft licence
check_license:
	poetry run pip-licenses --fail-on \
	"GNU General Public License v2 (GPLv2); \
	GNU Lesser General Public License v2 (LGPLv2); \
	GNU Affero General Public License v3 or later (AGPLv3+)" \
	--ignore-packages pylint astroid pygit2 grandalf

#################################################################################
# SELF-DOCUMENTING HELP                                                         #
#################################################################################

help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
