.PHONY: setup test clean data train evaluate

# Development setup
setup:
	python -m venv venv
	. venv/bin/activate && pip install -e ".[dev]"

# Generate sample data
data:
	python scripts/generate_sample_data.py
	python scripts/prepare_data.py --input data/raw/sample_data.csv

# Train the model
train:
	python scripts/train_model.py --data data/processed/train.csv

# Evaluate the model
evaluate:
	python scripts/evaluate_model.py --model models/model.joblib --test-data data/processed/test.csv

# Run tests
test:
	pytest tests/

# Clean generated files
clean:
	rm -rf data/raw/*.csv
	rm -rf data/processed/*.csv
	rm -rf models/*.joblib
	rm -rf logs/*.log
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
