UV            ?= uv

# Root Makefile passes BUILDDIR; default to ../build if not set
MKFILE_DIR           := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
BUILDDIR             ?= $(abspath $(MKFILE_DIR)/../build)
NCCL_DIR             := $(abspath $(MKFILE_DIR)/../src)
NCCL4PY_DIR          := $(abspath $(MKFILE_DIR))
NCCL4PY_BINDINGS_DIR := $(NCCL4PY_DIR)/nccl/bindings

EXTERNALS_DIR      := $(BUILDDIR)/externals
EXTERNALS_VENV_DIR := $(EXTERNALS_DIR)/venvs
DIST_DIR           := $(BUILDDIR)/dist

# CUDA home is only used to provide headers for cython
CUDA_HOME       ?= /usr/local/cuda
CUDA_VARIANTS   := 11 12 13
PYTHON_VERSIONS := 3.10 3.11 3.12 3.13

# manylinux repair configuration
AUDITWHEEL_PLAT ?= manylinux_2_34_$(shell uname -m)

.DEFAULT_GOAL := build
.PHONY: all build dev clean

all: build

NCCL4PY_BINDINGS_STAMP := $(NCCL4PY_BINDINGS_DIR)/makefile.stamp
-include $(NCCL4PY_DIR)/makefiles/run_cybind.mk

dev: $(NCCL4PY_BINDINGS_STAMP)
	@cd $(NCCL4PY_DIR) && $(UV) sync --extra dev && $(UV) pip install -e .[dev]


NCCL4PY_FILES := \
	nccl/__init__.py \
	nccl/core/*.py \
	nccl/bindings/*.pyx \
	nccl/bindings/*.pxd \
	nccl/bindings/*.py \
	nccl/bindings/_internal/*.pyx \
	nccl/bindings/_internal/*.pxd \
	nccl/bindings/_internal/*.py

BUILD_DIR = $(BUILDDIR)/nccl4py_cu$*
OUT_DIR = $(DIST_DIR)/nccl4py_cu$*

$(CUDA_VARIANTS:%=$(DIST_DIR)/nccl4py_cu%/makefile.stamp): $(DIST_DIR)/nccl4py_cu%/makefile.stamp : $(NCCL4PY_BINDINGS_STAMP) $(wildcard $(NCCL4PY_DIR)/nccl/core/*.py)
	mkdir -p "$(BUILD_DIR)"
	cp "$(NCCL4PY_DIR)/nccl4py_cu$*/pyproject.toml" "$(NCCL4PY_DIR)/nccl4py_cu$*/setup.py" "$(NCCL4PY_DIR)/MANIFEST.in" "$(NCCL4PY_DIR)/README.md" "$(BUILD_DIR)/"
	cd "$(NCCL4PY_DIR)"; cp --parents $(NCCL4PY_FILES) "$(BUILD_DIR)/"
	CUDA_HOME="$(CUDA_HOME)" $(UV) build --sdist -p "cpython@3.13" --out-dir "$(OUT_DIR)" "$(BUILD_DIR)"
	for v in $(PYTHON_VERSIONS); do \
	  	CUDA_HOME="$(CUDA_HOME)" $(UV) build --wheel -p "cpython@$${v}" --out-dir "$(OUT_DIR)" "$(OUT_DIR)"/nccl4py_cu$*-*.tar.gz; \
	done;
	find "$(OUT_DIR)" -maxdepth 1 -type f -name '*.whl' ! -name '*manylinux*' -print0 | \
		xargs -0 -r $(UV) run --isolated --no-project -p "cpython@3.13" --with auditwheel --with patchelf \
		-m auditwheel repair --plat "$(AUDITWHEEL_PLAT)" -w "$(OUT_DIR)"; \
	touch "$@"

$(CUDA_VARIANTS:%=nccl4py_cu%): nccl4py_cu%: $(DIST_DIR)/nccl4py_cu%/makefile.stamp

build: $(CUDA_VARIANTS:%=nccl4py_cu%)

clean:
	@rm -rf $(EXTERNALS_DIR) \
	        $(DIST_DIR) \
			$(BUILDDIR)/nccl4py_cu* \
			$(NCCL4PY_BINDINGS_DIR) \
			$(NCCL4PY_DIR)/*.egg-info \
			$(NCCL4PY_DIR)/build \
			$(NCCL4PY_DIR)/nccl/_version.py

