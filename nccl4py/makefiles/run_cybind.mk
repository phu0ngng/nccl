# GitLab cybind source
GITLAB_HOST    := gitlab-master.nvidia.com
CYBIND_PROJ_ID := 106982
CYBIND_COMMIT  := 01e0f5b44e168578e87eb1773199fcba9daa664f
ARCHIVE_URL    := https://$(GITLAB_HOST)/api/v4/projects/$(CYBIND_PROJ_ID)/repository/archive.tar.gz?sha=$(CYBIND_COMMIT)

NCCL4PY_ASSETS_DIR   := $(NCCL4PY_DIR)/build_assets

# NCCL generated header path from src/Makefile
NCCL_INCDIR        := $(BUILDDIR)/include
NCCL_HEADER        := $(NCCL_INCDIR)/nccl.h

CYBIND_DIR             := $(EXTERNALS_DIR)/cybind
CYBIND_OUTPUT_DIR      := $(EXTERNALS_DIR)/cybind/out
CYBIND_DIR_STAMP       := $(CYBIND_DIR)/makefile.stamp
CYBIND_OUTPUT_STAMP    := $(CYBIND_OUTPUT_DIR)/makefile.stamp

# Auth header value for GitLab API (no -H and no nested quotes)
TOKEN_HEADER = $(strip $(if $(GITLAB_TOKEN),PRIVATE-TOKEN: $(GITLAB_TOKEN),$(if $(CI_JOB_TOKEN),JOB-TOKEN: $(CI_JOB_TOKEN),)))

.PHONY: generate_bindings

# Bridge rule: ensure the generated NCCL header exists by invoking src/Makefile
$(NCCL_HEADER):
	$(MAKE) -C $(NCCL_DIR) $(NCCL_HEADER)

$(CYBIND_DIR_STAMP):
	@if [ -z "$(TOKEN_HEADER)" ]; then echo "ERROR: Please export GITLAB_TOKEN or CI_JOB_TOKEN for $(GITLAB_HOST)"; exit 1; fi
	@rm -rf "$(CYBIND_DIR)"; mkdir -p "$(CYBIND_DIR)"
	@echo "Downloading and extracting cybind archive..."
	@mkdir -p "$(CYBIND_DIR)"
	@curl -fsSL -H "$(TOKEN_HEADER)" "$(ARCHIVE_URL)" | tar -xz -C "$(CYBIND_DIR)" --strip-components=1
	@touch "$(CYBIND_DIR_STAMP)"

$(CYBIND_OUTPUT_STAMP): $(CYBIND_DIR_STAMP) $(shell find $(NCCL4PY_ASSETS_DIR)/cybind -type f -name '*.py' -o -name '*.pyx' -o -name '*.pxd') $(NCCL_HEADER)
	@cp -f "$(NCCL4PY_ASSETS_DIR)/cybind/config_nccl.py" "$(CYBIND_DIR)/assets/configs/config_nccl.py"
	@mkdir -p "$(CYBIND_DIR)/assets/templates/nccl4py/bindings/"
	@cp -rf $(NCCL4PY_ASSETS_DIR)/cybind/templates/* "$(CYBIND_DIR)/assets/templates/nccl4py/bindings/"
	@echo "Generating cybind bindings..."
	@cd "$(CYBIND_DIR)" && CUDA_PATH=$(CUDA_HOME) $(UV) run --isolated --with . -p 3.13 -m cybind --generate nccl --input-dir "$(NCCL_INCDIR)" --output "$(CYBIND_OUTPUT_DIR)/"
	@touch "$(CYBIND_OUTPUT_STAMP)"

$(NCCL4PY_BINDINGS_STAMP): $(CYBIND_OUTPUT_STAMP) $(NCCL4PY_ASSETS_DIR)/cybind/templates/_internal/utils.*
	@echo "Copying cybind bindings..."
	@mkdir -p "$(NCCL4PY_BINDINGS_DIR)/_internal"
	@cp -a $(CYBIND_OUTPUT_DIR)/nccl4py/bindings/* "$(NCCL4PY_BINDINGS_DIR)"
	@cp $(NCCL4PY_ASSETS_DIR)/cybind/templates/_internal/utils.* "$(NCCL4PY_BINDINGS_DIR)/_internal/"
	@echo > "$(NCCL4PY_BINDINGS_DIR)/_internal/__init__.py"
	@echo "from .nccl import *" > "$(NCCL4PY_BINDINGS_DIR)/__init__.py"
	@touch "$(NCCL4PY_BINDINGS_STAMP)"
