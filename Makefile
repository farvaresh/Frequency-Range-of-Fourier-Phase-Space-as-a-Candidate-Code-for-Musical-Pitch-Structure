# Convenience targets. Run `make help` for a list.

PYTHON ?= python

.PHONY: help install test stimuli smoke medium full analyse figure paper clean

help:
	@echo "install   install dependencies and the package (editable)"
	@echo "test      run the reproducibility test suite"
	@echo "stimuli   export all task stimulus sets to data/stimuli/"
	@echo "smoke     ~2 min   sanity run"
	@echo "medium    ~40 min  does the pattern appear?"
	@echo "full      hours    the numbers reported in the paper"
	@echo "analyse   re-run statistics on an existing results_full/unit_fits.csv"
	@echo "figure    regenerate Figure 6 from the shipped counts"
	@echo "paper     compile the manuscript (needs a LaTeX installation)"
	@echo "clean     remove build artefacts and LaTeX intermediates"

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

test:
	$(PYTHON) -m pytest -q

stimuli:
	$(PYTHON) scripts/export_stimuli.py

smoke:
	$(PYTHON) run_config.py smoke

medium:
	$(PYTHON) run_config.py medium

full:
	$(PYTHON) run_config.py full

analyse:
	$(PYTHON) -m iranian_anns.analyze_results results_full/unit_fits.csv --out results_full/report

figure:
	$(PYTHON) paper/make_fig6.py

paper:
	cd paper && pdflatex -interaction=nonstopmode main.tex \
	  && bibtex main && pdflatex -interaction=nonstopmode main.tex \
	  && pdflatex -interaction=nonstopmode main.tex

clean:
	rm -rf build dist *.egg-info src/*.egg-info .pytest_cache
	find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
	cd paper && rm -f *.aux *.log *.bbl *.blg *.out
