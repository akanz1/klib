PYTHON_VERSION = 3.12

.PHONY: lock-dependencies
lock-dependencies:  ## Lock dependencies based on pyproject.toml
	@echo "$(COLOR_BLUE)Locking backend dependencies..$(COLOR_RESET)"
	@uv lock --python $(PYTHON_VERSION) --upgrade
	@echo "🚀 $(COLOR_GREEN)Dependencies locked$(COLOR_RESET) 🚀"

.PHONY: create-venv
create-venv:  ## Create fresh python venv and install dev requirements
	@echo "$(COLOR_BLUE)Creating venv..$(COLOR_RESET)"
	@curl -LsSf https://astral.sh/uv/install.sh | sh
	@uv venv --python $(PYTHON_VERSION) && . .venv/bin/activate && uv sync --frozen --all-extras
	@echo "🚀 $(COLOR_GREEN)Venv created$(COLOR_RESET) 🚀"
	@echo

.PHONY: pc
pc:  ## run pre-commit on all files
	@pre-commit run --all-files
