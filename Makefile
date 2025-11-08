# Makefile for STAT 159/259 HW3
ENV_NAME ?= stat159-hw3

.PHONY: env html clean

env:
	@if command -v mamba >/dev/null 2>&1; then \
		mamba env update -n $(ENV_NAME) -f environment.yml --prune || mamba env create -n $(ENV_NAME) -f environment.yml; \
	else \
		conda env update -n $(ENV_NAME) -f environment.yml --prune || conda env create -n $(ENV_NAME) -f environment.yml; \
	fi
	@echo "Environment ready: $(ENV_NAME). Activate with: conda activate $(ENV_NAME)"

html:
	myst build --html
	@echo "Open _build/html/index.html to view locally."

clean:
	-rm -rf _build
	-rm -f figures/* audio/*
	@echo "Cleaned _build, figures/, audio/."
