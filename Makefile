.PHONY: install test run

install:
	python3 -m venv .venv
	. .venv/bin/activate && python -m pip install --upgrade pip && pip install -r requirements.txt

test:
	. .venv/bin/activate && python -m pytest -q

run:
	. .venv/bin/activate && python weather_ml.py
