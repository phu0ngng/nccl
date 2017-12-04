ITEM=1
include ../common/common.mk

REQ=NCCL-Req-$(ITEM)
DESIGN=NCCL-Design-$(ITEM)
CODING=NCCL-Coding-$(ITEM)
HTML=$(BUILDDIR)/$(REQ).html $(BUILDDIR)/$(DESIGN).html $(BUILDDIR)/$(CODING).html
PDF=$(BUILDDIR)/$(REQ).pdf $(BUILDDIR)/$(DESIGN).pdf $(BUILDDIR)/$(CODING).pdf

all: $(PDF)
pdf: $(PDF)
html: $(HTML)

$(BUILDDIR)/%.pdf: $(BUILDDIR)/%.html
	mkdir -p $(BUILDDIR)
	$(CHROME) --headless --print-to-pdf="$@" file://$<
	chmod 644 $@

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

$(BUILDDIR)/$(CODING).html: coding.html coding.list
	mkdir -p $(BUILDDIR)
	cp $< $@
	../../html/tools/git_to_html.sh coding.list >> $@

clean:
	rm -f $(HTML) $(PDF)
