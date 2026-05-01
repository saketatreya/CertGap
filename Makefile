# CertGap Reproduction Makefile

.PHONY: help install figures clean verify paper

help:
	@echo "CertGap Reproduction Commands:"
	@echo "  install   : Set up the environment and dependencies"
	@echo "  figures   : Regenerate all paper figures and tables from results"
	@echo "  paper     : Compile the final PDF manuscript"
	@echo "  verify    : Run smoke tests and tabular identity check"
	@echo "  clean     : Remove build artifacts and temporary files"

install:
	python -m venv .venv
	.venv/bin/python -m pip install -e .
	.venv/bin/python -m pip install "gymnasium[mujoco]"

figures:
	export PYTHONPATH=.:$$PYTHONPATH; 	mkdir -p figures/out; 	.venv/bin/python figures/fig1_powerhouse.py; 	.venv/bin/python figures/fig1_auroc_bars.py; 	.venv/bin/python figures/fig2_calibration.py; 	.venv/bin/python figures/fig3_factorial.py; 	.venv/bin/python scripts/update_audit_table.py > figures/out/table1.tex; 	.venv/bin/python scripts/compare_intervention.py; 	.venv/bin/python scripts/analyze_ak_correlation.py

paper: figures
	pdflatex paper.tex
	bibtex paper
	pdflatex paper.tex
	pdflatex paper.tex
	rm -f paper.aux paper.bbl paper.blg paper.log paper.out

verify:
	.venv/bin/pytest tests/
	.venv/bin/python figures/appendix/figA1_tabular_identity.py

clean:
	rm -rf .pytest_cache
	rm -rf figures/out
	rm -f paper.aux paper.bbl paper.blg paper.log paper.out paper.pdf
