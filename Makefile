.PHONY: setup data test dashboard e2e-mock e2e clean

PY := $(shell conda info --base)/envs/analytics-agent/bin/python
STREAMLIT_PORT ?= 8601

setup:
	conda env create -f environment.yml || conda env update -f environment.yml --prune
	@echo "环境就绪：conda activate analytics-agent"

data:
	$(PY) simulate_data.py --force

test:
	$(PY) -m pytest -q

dashboard:
	$(PY) -m streamlit run app/dashboard.py --server.port $(STREAMLIT_PORT)

e2e-mock:
	$(PY) scripts/run_e2e_test.py --mode mock --force

e2e:
	./run_e2e_test.sh

clean:
	rm -rf .pytest_cache __pycache__ src/__pycache__ src/nodes/__pycache__ \
		tests/__pycache__ app/__pycache__ outputs/e2e_result.html \
		outputs/e2e_result.json outputs/e2e_preview_mock.html
