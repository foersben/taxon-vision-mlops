# Makefile for TaxonVision-MLOps Final Presentation (v2)

all: presentation.pdf

presentation.pdf: presentation.tex
	pdflatex -interaction=nonstopmode presentation.tex
	pdflatex -interaction=nonstopmode presentation.tex

notes: presentation.tex
	sed 's/\\setbeameroption{hide notes}/\\setbeameroption{show notes on second screen=right}/' presentation.tex > presentation_notes.tex
	pdflatex -interaction=nonstopmode presentation_notes.tex
	pdflatex -interaction=nonstopmode presentation_notes.tex
	rm -f presentation_notes.tex presentation_notes.aux presentation_notes.log presentation_notes.nav presentation_notes.out presentation_notes.snm presentation_notes.toc presentation_notes.vrb

clean:
	rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb presentation_notes.tex

.PHONY: all notes clean
