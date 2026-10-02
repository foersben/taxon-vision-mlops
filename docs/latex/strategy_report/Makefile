# Makefile for LaTeX document compilation with BibTeX support

PROJECT ?= strategy_report

# Default rule
all: $(PROJECT).pdf

pdf: $(PROJECT).pdf

export TEXMF_OUTPUT_DIRECTORY ?= .

# Rule to build the PDF with BibTeX reference resolution
$(PROJECT).pdf: $(PROJECT).tex references.bib $(wildcard chapters/*.tex)
	pdflatex -shell-escape -interaction=nonstopmode -halt-on-error $(PROJECT).tex
	bibtex $(PROJECT)
	pdflatex -shell-escape -interaction=nonstopmode -halt-on-error $(PROJECT).tex
	pdflatex -shell-escape -interaction=nonstopmode -halt-on-error $(PROJECT).tex

# Clean rule to remove temporary, auxiliary, and target PDF files
clean:
	rm -rf *.aux *.log *.out *.toc *.nav *.snm *.vrb *.pdf *.bbl *.blg *.synctex.gz *.fls *.fdb_latexmk *.minted _minted*

# Clean auxiliary files only, preserving the compiled PDF
clean-aux:
	rm -rf *.aux *.log *.out *.toc *.nav *.snm *.vrb *.bbl *.blg *.synctex.gz *.fls *.fdb_latexmk *.minted _minted*

.PHONY: all pdf clean clean-aux
