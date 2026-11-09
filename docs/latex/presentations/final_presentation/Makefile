# Makefile for TaxonVision Final MLOps Presentation

all: presentation.pdf

presentation.pdf: presentation.tex
	pdflatex -interaction=nonstopmode presentation.tex
	pdflatex -interaction=nonstopmode presentation.tex

clean:
	rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb

.PHONY: all clean
