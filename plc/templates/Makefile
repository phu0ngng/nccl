ITEM=${plc:item}
CHROME=chromium-browser

REQ=NCCL-Req-$(ITEM)
DESIGN=NCCL-Design-$(ITEM)

all: $(REQ).pdf $(DESIGN).pdf

%.pdf: %.html
	$(CHROME) --headless --print-to-pdf="$@" file://$(PWD)/$<

$(REQ).html: req.html
	../../../doc/index_titles.py $< > $@

$(DESIGN).html: design.html
	../../../doc/index_titles.py $< > $@

clean:
	rm $(REQ).pdf $(DESIGN).pdf $(REQ).html $(DESIGN).html
