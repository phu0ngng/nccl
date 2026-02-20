BUILDDIR?= $(abspath ../../../../build/plc/items/$(ITEM))

VERBOSE ?= 0
ifeq ($(VERBOSE), 0)
.SILENT:
endif

CHROME?=chromium-browser

REQ=NCCL-Req-$(ITEM)
DESIGN=NCCL-Design-$(ITEM)
CODING=NCCL-Coding-$(ITEM)
TESTING=NCCL-Testing-$(ITEM)
HTML=$(BUILDDIR)/index.html $(BUILDDIR)/$(REQ).html $(BUILDDIR)/$(DESIGN).html $(BUILDDIR)/$(CODING).html $(BUILDDIR)/$(TESTING).html
PDF=$(BUILDDIR)/$(REQ).pdf $(BUILDDIR)/$(DESIGN).pdf $(BUILDDIR)/$(CODING).pdf $(BUILDDIR)/$(TESTING).pdf

all: pdf html
pdf: $(PDF)
html: $(HTML)

$(BUILDDIR)/%.pdf: $(BUILDDIR)/%.html
	@printf "Generating %-35s > %s\n" $< $@
	mkdir -p $(BUILDDIR)
	../../html/tools/html_to_pdf.sh $(CHROME) $< $@
	chmod 644 $@

$(BUILDDIR)/index.html: index.html
	cp $< $@

$(BUILDDIR)/$(REQ).html: req.html
	@printf "Building   %-35s > %s\n" $< $@
	mkdir -p $(BUILDDIR)
	rsync -a images $(BUILDDIR)
	mkdir -p $(BUILDDIR)/../../html
	rsync -a ../../html/css $(BUILDDIR)/../../html/
	cp $< $@.tmp
	../../html/tools/git_history.sh $< >> $@.tmp
	../../html/tools/index_titles.py $@.tmp > $@
	rm $@.tmp

$(BUILDDIR)/$(DESIGN).html: design.html
	@printf "Building   %-35s > %s\n" $< $@
	mkdir -p $(BUILDDIR)
	rsync -a images $(BUILDDIR)
	mkdir -p $(BUILDDIR)/../../html
	rsync -a ../../html/css $(BUILDDIR)/../../html/
	cp $< $@.tmp
	../../html/tools/git_history.sh $< >> $@.tmp
	../../html/tools/index_titles.py $@.tmp > $@
	rm $@.tmp

$(BUILDDIR)/$(TESTING).html: testing.html
	@printf "Building   %-35s > %s\n" $< $@
	mkdir -p $(BUILDDIR)
	cp $< $@.tmp
	../../html/tools/git_history.sh $< >> $@.tmp
	../../html/tools/index_titles.py $@.tmp > $@
	rm $@.tmp

$(BUILDDIR)/$(CODING).html: coding.html coding.list
	@printf "Building   %-35s > %s\n" $< $@
	mkdir -p $(BUILDDIR)
	cp $< $@
	../../html/tools/git_to_html.sh coding.list >> $@

clean:
	rm -f $(HTML) $(PDF)
