BUILDDIR?= $(abspath ../../../build/plc/items/$(ITEM))

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
	rsync -a images $(BUILDDIR)
	mkdir -p $(BUILDDIR)/../../html
	rsync -a ../../html/css $(BUILDDIR)/../../html/

$(BUILDDIR)/%.pdf: $(BUILDDIR)/%.html
	mkdir -p $(BUILDDIR)
	$(CHROME) --headless --print-to-pdf="$@" file://$<
	chmod 644 $@

$(BUILDDIR)/index.html: index.html
	cp $< $@

$(BUILDDIR)/$(REQ).html: req.html
	mkdir -p $(BUILDDIR)
	cp $< $@.tmp
	../../html/tools/git_history.sh $< >> $@.tmp
	../../html/tools/index_titles.py $@.tmp > $@
	rm $@.tmp

$(BUILDDIR)/$(DESIGN).html: design.html
	mkdir -p $(BUILDDIR)
	cp $< $@.tmp
	../../html/tools/git_history.sh $< >> $@.tmp
	../../html/tools/index_titles.py $@.tmp > $@
	rm $@.tmp

$(BUILDDIR)/$(TESTING).html: testing.html
	mkdir -p $(BUILDDIR)
	cp $< $@.tmp
	../../html/tools/git_history.sh $< >> $@.tmp
	../../html/tools/index_titles.py $@.tmp > $@
	rm $@.tmp

$(BUILDDIR)/$(CODING).html: coding.html coding.list
	mkdir -p $(BUILDDIR)
	cp $< $@
	../../html/tools/git_to_html.sh coding.list >> $@

clean:
	rm -f $(HTML) $(PDF)
