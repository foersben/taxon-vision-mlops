# Makefile for LaTeX document compilation with BibTeX support

PROJECT ?= strategy_report

# Prevent parallel execution of targets (e.g. 'make clean all' running rm and pdflatex concurrently)
.NOTPARALLEL:

# Default rule
all: $(PROJECT).pdf

pdf: $(PROJECT).pdf

export TEXMF_OUTPUT_DIRECTORY ?= .

# Rule to build the PDF with automated dependency tracking and BibTeX resolution via latexmk
$(PROJECT).pdf: $(PROJECT).tex references.bib $(wildcard chapters/*.tex)
	latexmk -pdf -outdir=. -shell-escape -interaction=nonstopmode -halt-on-error $(PROJECT).tex

# Clean rule to remove temporary, auxiliary, and target PDF files
clean:
	latexmk -C $(PROJECT).tex 2>/dev/null || true
	rm -rf *.aux *.log *.out *.toc *.nav *.snm *.vrb *.pdf *.bbl *.blg *.synctex.gz *.fls *.fdb_latexmk *.minted _minted* *.run.xml *.bcf

# Clean auxiliary files only, preserving the compiled PDF
clean-aux:
	latexmk -c $(PROJECT).tex 2>/dev/null || true
	rm -rf *.aux *.log *.out *.toc *.nav *.snm *.vrb *.bbl *.blg *.synctex.gz *.fls *.fdb_latexmk *.minted _minted* *.run.xml *.bcf

.PHONY: all pdf clean clean-aux
