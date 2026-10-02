# Makefile for LaTeX document compilation with BibTeX support

PROJECT ?= concept_design

# Default rule
all: $(PROJECT).pdf

pdf: $(PROJECT).pdf

# Rule to build the PDF with BibTeX reference resolution
$(PROJECT).pdf: $(PROJECT).tex references.bib
	pdflatex -interaction=nonstopmode -halt-on-error $(PROJECT).tex
	bibtex $(PROJECT)
	pdflatex -interaction=nonstopmode -halt-on-error $(PROJECT).tex
	pdflatex -interaction=nonstopmode -halt-on-error $(PROJECT).tex

# Clean rule to remove temporary, auxiliary, and target PDF files
clean:
	rm -f *.aux *.log *.out *.toc *.nav *.snm *.vrb *.pdf *.bbl *.blg *.synctex.gz

# Clean auxiliary files only, preserving the compiled PDF
clean-aux:
	rm -f *.aux *.log *.out *.toc *.nav *.snm *.vrb *.bbl *.blg *.synctex.gz

.PHONY: all pdf clean clean-aux
