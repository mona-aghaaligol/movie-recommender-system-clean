# --------------------------------------------------------------
# Recommender System — Pipeline Execution Makefile
# --------------------------------------------------------------
# Usage:
#   make similarity_user    → Compute user-user similarity
#   make similarity_item    → Compute item-item similarity
#   make similarity_all     → Compute both similarities
#   make test               → Run tests
#   make eval               → Run baseline evaluation (writes metrics.json)
#   make eval_md            → Render metrics.md from metrics.json
#   make eval-gate          → Run evaluation gate (baseline enforced, model optional)
#   make clean              → Clean generated outputs
#
# Evaluation gate args:
#   make eval-gate
#   make eval-gate ARGS="--enforce-model"
#   make eval-gate ARGS="--enforce-model --margin 0.001"
# --------------------------------------------------------------

# Python interpreter
PYTHON=python

# Activate virtual environment (optional, if needed)
VENV=. venv/bin/activate &&

# --------------------------------------------------------------
# User-User Similarity Pipeline
# --------------------------------------------------------------
similarity_user:
	$(VENV) \
	$(PYTHON) -m src.recommender.compute_similarity_user

# --------------------------------------------------------------
# Item-Item Similarity Pipeline
# --------------------------------------------------------------
similarity_item:
	$(VENV) \
	$(PYTHON) -m src.recommender.compute_similarity_item

# --------------------------------------------------------------
# Run both pipelines
# --------------------------------------------------------------
similarity_all:
	$(VENV) \
	$(PYTHON) -m src.recommender.compute_similarity_user
	$(VENV) \
	$(PYTHON) -m src.recommender.compute_similarity_item

# --------------------------------------------------------------
# Testing
# --------------------------------------------------------------
test:
	$(VENV) \
	pytest -q

# --------------------------------------------------------------
# Evaluation (Baselines)
# --------------------------------------------------------------
eval:
	$(VENV) \
	$(PYTHON) scripts/run_evaluation_baselines.py

eval_md: eval
	$(VENV) \
	$(PYTHON) scripts/render_metrics_md.py

# --------------------------------------------------------------
# Clean generated outputs
# --------------------------------------------------------------
clean:
	rm -f data/similarity_matrix_user_user.csv
	rm -f data/similarity_matrix_item_item.csv

# --------------------------------------------------------------
# Evaluation Gate (CI-safe)
# --------------------------------------------------------------
eval-gate:
	python -m src.recommender.evaluation.cli_gate $(ARGS)

# --------------------------------------------------------------
# Docker (Reproducible Runtime)
# --------------------------------------------------------------
docker-build:
	docker build -t movie-recommender:latest .

docker-test:
	docker run --rm movie-recommender:latest pytest -q

docker-eval-gate:
	docker run --rm movie-recommender:latest python -m src.recommender.evaluation.cli_gate $(ARGS)

docker-run:
	docker run --rm -p 8000:8000 movie-recommender:latest
