TEX = thesis_proposal.tex
PDF = thesis_proposal.pdf

LATEXMK = latexmk

.PHONY: all clean view
all: $(PDF)

$(PDF): $(TEX) $(shell find chapters -name '*.tex') references.bib
	$(LATEXMK) -pdf -interaction=nonstopmode $(TEX)

clean:
	$(LATEXMK) -C $(TEX) || true

tidy: clean
	@rm -f $(PDF)

view: $(PDF)
	@xdg-open $(PDF) >/dev/null 2>&1 || true
