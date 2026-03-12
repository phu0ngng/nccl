#
# SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#

include ../../makefiles/common.mk
include ../../makefiles/version.mk
BUILDDIR ?= $(abspath ../../build)
DOCSRC  := $(abspath ../../docs/userguide/)
PKGDIR  := $(BUILDDIR)/pkg/doc

.PHONY: build
build:
	@printf "Building docs\n"
	make -C $(DOCSRC) html
	@printf "Packaging docs\n"
	mkdir -p $(PKGDIR)
	(cd $(BUILDDIR)/doc; zip -r html.zip html)
	cp $(BUILDDIR)/doc/html.zip $(PKGDIR)/nccl-doc_$(NCCL_MAJOR).$(NCCL_MINOR).$(NCCL_PATCH)$(NCCL_SUFFIX)-$(PKG_REVISION).zip
	@echo "Created $(PKGDIR)/nccl-doc_$(NCCL_MAJOR).$(NCCL_MINOR).$(NCCL_PATCH)$(NCCL_SUFFIX)-$(PKG_REVISION).zip"

clean:
	rm -Rf $(PKGDIR) $(BUILDDIR)/doc
