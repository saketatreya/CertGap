# CertGap reproduction Makefile.
#
# Common targets:
#   make install     # set up the local virtualenv and install certgap + deps
#   make experiments # run every training run from scratch (skip-if-exists)
#   make figures     # regenerate every paper figure and table from results/
#   make paper       # rebuild paper/paper.pdf
#   make verify      # run the unit tests + tabular-identity sanity check
#   make clean       # remove build artifacts (PDFs, latex aux, pyc, figures/out)

.PHONY: help install experiments figures paper verify clean

PY := .venv/bin/python
PYTEST := .venv/bin/pytest
WORKERS ?= 6

help:
	@echo "CertGap reproduction commands:"
	@echo "  make install     -- create .venv and install certgap (editable) + deps"
	@echo "  make experiments -- run every training run (~13h on 8-core CPU; idempotent)"
	@echo "  make figures     -- regenerate every figure and table from results/"
	@echo "  make paper       -- rebuild paper/paper.pdf (requires figures + table1)"
	@echo "  make verify      -- pytest + tabular-identity sanity check"
	@echo "  make clean       -- remove generated PDFs, LaTeX aux files, __pycache__"

install:
	python -m venv .venv
	$(PY) -m pip install -e .
	$(PY) -m pip install "gymnasium[mujoco]"

# Single-command full reproduction. Idempotent: skips runs whose pickle is on
# disk. Pass WORKERS=N to set parallelism (default 6).
experiments:
	$(PY) scripts/run_all.py --workers $(WORKERS) --skip-if-exists --include-trpo

# Regenerate every figure (main + appendix) and the headline table from the
# pickles in results/.
figures:
	mkdir -p figures/out figures/out/appendix
	PYTHONPATH=. $(PY) figures/fig1_powerhouse.py
	PYTHONPATH=. $(PY) figures/fig2_mechanism.py
	PYTHONPATH=. $(PY) figures/fig3_factorial.py
	PYTHONPATH=. $(PY) figures/fig_hyperparam_ranking.py
	PYTHONPATH=. $(PY) figures/appendix/figA1_dumbbell.py
	PYTHONPATH=. $(PY) figures/appendix/figA1_tabular_identity.py
	PYTHONPATH=. $(PY) figures/appendix/figA3_eps_u_variants.py
	PYTHONPATH=. $(PY) scripts/build_table1.py

paper: figures
	cd paper && pdflatex -interaction=nonstopmode paper.tex
	cd paper && bibtex paper
	cd paper && pdflatex -interaction=nonstopmode paper.tex
	cd paper && pdflatex -interaction=nonstopmode paper.tex
	cd paper && rm -f paper.aux paper.bbl paper.blg paper.log paper.out paper.toc

verify:
	$(PYTEST) tests/
	PYTHONPATH=. $(PY) figures/appendix/figA1_tabular_identity.py

clean:
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +
	rm -rf .pytest_cache figures/out paper/paper.pdf
	cd paper && rm -f paper.aux paper.bbl paper.blg paper.log paper.out paper.toc
