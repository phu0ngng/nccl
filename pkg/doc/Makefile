#
# Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#

include ../../makefiles/common.mk
include ../../makefiles/version.mk
BUILDDIR ?= $(abspath ../../build)
DOCSRC  := $(abspath ../../doc/)
DOCPREPDIR  := $(BUILDDIR)/doc
PKGDIR  := $(BUILDDIR)/pkg/doc

DOCGEN_IN  := $(shell (cd $(DOCSRC); find . -name "*.in" -type f -print -o -path './.venv' -prune))
DOCGEN     := $(DOCGEN_IN:.in=)
DOCCOPY    := $(shell (cd $(DOCSRC); find . -not -name "*.in" -type f -print -o -path './.venv' -prune))
DOCTARGETS := $(patsubst ./%, $(DOCPREPDIR)/%, $(DOCGEN)) $(patsubst ./%, $(DOCPREPDIR)/%, $(DOCCOPY))

PKG_ARCH   := $(shell uname -m)

prep: $(DOCTARGETS)

build: prep
	@printf "Building doc package\n"
	make -C $(DOCPREPDIR) html
	rm -Rf $(DOCPREPDIR)/html
	mv $(DOCPREPDIR)/../html $(DOCPREPDIR)
	mkdir -p $(PKGDIR)
	(cd $(DOCPREPDIR); zip -r $(PKGDIR)/nccl-doc_$(NCCL_MAJOR).$(NCCL_MINOR).$(NCCL_PATCH)$(NCCL_SUFFIX)-$(PKG_REVISION).zip html)

clean:
	rm -Rf $(DOCPREPDIR) $(PKGDIR)

$(DOCPREPDIR)/% : $(DOCSRC)/%
	@printf "Copying %-35s > %s\n" $< $@
	mkdir -p `dirname $@`
	cp $< $@

$(DOCPREPDIR)/% : $(DOCSRC)/%.in
	@printf "Generating %-35s > %s\n" $< $@
	mkdir -p `dirname $@`
	sed -e "s/\$${nccl:Major}/$(NCCL_MAJOR)/g" \
	    -e "s/\$${nccl:Minor}/$(NCCL_MINOR)/g" \
	    -e "s/\$${nccl:Patch}/$(NCCL_PATCH)/g" \
	    -e "s/\$${nccl:Suffix}/$(NCCL_SUFFIX)/g" \
	    -e "s/\$${cuda:Major}/$(CUDA_MAJOR)/g" \
	    -e "s/\$${cuda:Minor}/$(CUDA_MINOR)/g" \
	    -e "s/\$${pkg:Revision}/$(PKG_REVISION)/g" \
	    -e "s/\$${pkg:Arch}/$(PKG_ARCH)/g" \
	    $< > $@
