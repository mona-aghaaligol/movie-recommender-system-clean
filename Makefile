# --------------------------------------------------------------
# Recommender System — Pipeline Execution Makefile
# --------------------------------------------------------------
# Usage:
#   make similarity_user    → Compute user-user similarity
#   make similarity_item    → Compute item-item similarity
#   make similarity_all     → Compute both similarities
#   make test              → Run tests
#   make eval              → Run baseline evaluation (writes metrics.json)
#   make eval_md           → Render metrics.md from metrics.json
#   make clean             → Clean generated outputs
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

eval-gate:
	python -m src.recommender.evaluation.cli_gate

