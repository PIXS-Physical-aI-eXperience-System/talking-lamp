# Talking Lamp - dev tasks
#
# `PYTHONPATH=` clears the /opt/ros/* entries a sourced ROS shell injects, which
# otherwise shadow the venv and break pytest's plugin autoload. Run from the
# repo root; the venv at ./.venv is used directly (no activation needed).

PY := PYTHONPATH= $(CURDIR)/.venv/bin/python

.PHONY: help venv test test-fallback demo drive jog goto check build lint

help:
	@echo "make venv           create .venv and install deps"
	@echo "make test           run the test suite (ruckig backend)"
	@echo "make test-fallback  run the suite with the analytic trajectory backend"
	@echo "make demo           run the scripted motion demo -> sim/out/"
	@echo "make drive          live viewer; hold W/A/S/D in THIS terminal to move the target"
	@echo "make jog            same control from a second terminal (writes sim/.target)"
	@echo 'make goto ARGS="x y z"     one-shot: jump the target to a point'
	@echo "make check          headless scene sanity check"
	@echo "make build          regenerate sim/lelamp_arm.xml from the vendored CAD export"
	@echo "make lint           pyflakes over src/ tests/ sim/"

venv:
	uv venv --python 3.12 .venv
	uv pip install --python .venv -e ".[dev]" pyflakes

test:
	$(PY) -m pytest tests

test-fallback:
	TALKING_LAMP_NO_RUCKIG=1 $(PY) -m pytest tests

demo:
	$(PY) sim/motion_demo.py

drive:
	$(PY) sim/drive.py $(ARGS)

jog:
	$(PY) sim/jog.py $(ARGS)

goto:
	@echo "$(ARGS)" > sim/.target && echo "sent: $(ARGS)"

check:
	$(PY) sim/check.py

build:
	$(PY) sim/build_arm.py

lint:
	$(PY) -m pyflakes src/motion tests sim/*.py
