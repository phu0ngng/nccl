include ../../makefiles/common.mk

.PHONY: all clean

BUILDDIR := $(abspath ../../build)
NCCLDIR := $(BUILDDIR)
NVCUFLAGS += -I$(NCCLDIR)/include/ -I../include
NVCUFLAGS += -I$(MPI_HOME)/include
MPI_NVLDFLAGS = -L$(MPI_HOME)/lib -lmpi
NCCL_NVLDFLAGS = -L$(NCCLDIR)/lib -lnccl

DST_DIR := $(BUILDDIR)/test/replay

all: $(DST_DIR)/replay

clean:
	rm -rf $(DST_DIR)

NCCL_TOP := ../..

TEST_VERIFIABLE_SRCDIR := ../verifiable
TEST_VERIFIABLE_BUILDDIR := $(BUILDDIR)/test/verifiable
include ../verifiable/verifiable.mk

${DST_DIR}/replay: replay.cu $(TEST_VERIFIABLE_HDRS) $(TEST_VERIFIABLE_LIBSO)
	@printf "Linking  %s\n" $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) replay.cu $(NVLDFLAGS) $(NCCL_NVLDFLAGS) $(MPI_NVLDFLAGS) -L$(TEST_VERIFIABLE_BUILDDIR) -lverifiable -Xlinker -rpath='$$ORIGIN:$$ORIGIN/../verifiable'
