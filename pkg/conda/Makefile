#
# SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#

include ../../makefiles/common.mk
include ../../makefiles/version.mk
BUILDDIR ?= $(abspath ../../build)
TGZPREPDIR  := $(BUILDDIR)/conda
PKGDIR  := $(BUILDDIR)/pkg/conda/

TGZGEN_IN  := $(wildcard *.in)
TGZGEN     := $(TGZGEN_IN:.in=)
TGZTARGETS := $(patsubst %, $(TGZPREPDIR)/%, $(TGZGEN))

PKG_ARCH   := $(shell uname -m)

prep: $(TGZTARGETS)
	$(MAKE) -C ../.. lic BUILDDIR=$(BUILDDIR)

build: prep
	$(MAKE) -C ../.. src.build BUILDDIR=$(BUILDDIR)
	@printf "Building tar.gz package\n"
	(cd $(BUILDDIR); bash conda/create_tarball.sh)
	mkdir -p $(PKGDIR)
	mv $(BUILDDIR)/../nccl*.tar.gz $(PKGDIR)
	@printf "Tarball generated in $(PKGDIR)\n"

clean:
	rm -Rf $(TGZPREPDIR) $(PKGDIR)

$(TGZPREPDIR)/% : %.in
	@printf "Generating %-35s > %s\n" $< $@
	mkdir -p $(TGZPREPDIR)
	sed -e "s/\$${nccl:Major}/$(NCCL_MAJOR)/g" \
	    -e "s/\$${nccl:Minor}/$(NCCL_MINOR)/g" \
	    -e "s/\$${nccl:Patch}/$(NCCL_PATCH)/g" \
	    -e "s/\$${pkg:Arch}/$(PKG_ARCH)/g" \
	    $< > $@
