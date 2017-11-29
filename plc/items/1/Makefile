ITEM=1
CHROME=chromium-browser

REQ=NCCL-Req-$(ITEM)
DESIGN=NCCL-Design-$(ITEM)

all: $(REQ).pdf $(DESIGN).pdf

%.pdf: %.html
	$(CHROME) --headless --print-to-pdf="$@" file://$(PWD)/$<

$(REQ).html: req.html
	cp $< $@.tmp
	../../html/tools/git_to_html.sh $< >> $@.tmp
	../../html/tools/index_titles.py $@.tmp > $@
	rm $@.tmp

$(DESIGN).html: design.html
	cp $< $@.tmp
	../../html/tools/git_to_html.sh $< >> $@.tmp
	../../html/tools/index_titles.py $@.tmp > $@
	rm $@.tmp

clean:
	rm $(REQ).pdf $(DESIGN).pdf $(REQ).html $(DESIGN).html
