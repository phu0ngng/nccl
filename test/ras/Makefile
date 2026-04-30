include ../../makefiles/common.mk

.PHONY: all clean os.build

BUILDDIR := $(abspath ../../build)
NCCLDIR := $(BUILDDIR)
NVCUFLAGS += -I$(NCCLDIR)/include/ -I../include
NVCUFLAGS += -I$(MPI_HOME)/include
MPI_NVLDFLAGS = -L$(MPI_HOME)/lib -lmpi
NCCL_NVLDFLAGS = -L$(NCCLDIR)/lib -lnccl

TEST_OS_SRCDIR := ../os
TEST_OS_BUILDDIR := $(BUILDDIR)/test/os
TEST_OS_LIB := $(TEST_OS_BUILDDIR)/libnccl_test_os.a
NVCUFLAGS += -I$(TEST_OS_SRCDIR)
NVLDFLAGS += -L$(TEST_OS_BUILDDIR) -lnccl_test_os

os.build:
	$(MAKE) -C $(TEST_OS_SRCDIR)

$(TEST_OS_LIB): os.build
	@:

DST_DIR := $(BUILDDIR)/test/ras

RAS_TESTS := $(DST_DIR)/single_comm_idle $(DST_DIR)/triple_comm_idle \
     $(DST_DIR)/triple_comm_longidle $(DST_DIR)/single_comm_1exit \
     $(DST_DIR)/triple_comm_latecoll $(DST_DIR)/single_comm_mixedcoll

# Suspend tests use kill()/SIGSTOP with no Windows equivalent
ifeq ($(NCCL_OS_LINUX), 1)
RAS_TESTS += $(DST_DIR)/single_comm_1suspend $(DST_DIR)/single_comm_2suspend \
     $(DST_DIR)/single_comm_4suspend
endif

all: os.build $(RAS_TESTS)

clean:
	rm -rf $(DST_DIR)

NCCL_TOP := ../..

${DST_DIR}/single_comm_idle: single_comm_idle.cu $(TEST_OS_LIB)
	@printf "Linking  %s\n" $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) $< $(NVLDFLAGS) $(NCCL_NVLDFLAGS) $(MPI_NVLDFLAGS)

${DST_DIR}/triple_comm_idle: triple_comm_idle.cu $(TEST_OS_LIB)
	@printf "Linking  %s\n" $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) $< $(NVLDFLAGS) $(NCCL_NVLDFLAGS) $(MPI_NVLDFLAGS)

${DST_DIR}/triple_comm_longidle: triple_comm_longidle.cu $(TEST_OS_LIB)
	@printf "Linking  %s\n" $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) $< $(NVLDFLAGS) $(NCCL_NVLDFLAGS) $(MPI_NVLDFLAGS)

${DST_DIR}/single_comm_1exit: single_comm_1exit.cu $(TEST_OS_LIB)
	@printf "Linking  %s\n" $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) $< $(NVLDFLAGS) $(NCCL_NVLDFLAGS) $(MPI_NVLDFLAGS)

${DST_DIR}/single_comm_1suspend: single_comm_1suspend.cu $(TEST_OS_LIB)
	@printf "Linking  %s\n" $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) $< $(NVLDFLAGS) $(NCCL_NVLDFLAGS) $(MPI_NVLDFLAGS)

${DST_DIR}/single_comm_2suspend: single_comm_2suspend.cu $(TEST_OS_LIB)
	@printf "Linking  %s\n" $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) $< $(NVLDFLAGS) $(NCCL_NVLDFLAGS) $(MPI_NVLDFLAGS)

${DST_DIR}/single_comm_4suspend: single_comm_4suspend.cu $(TEST_OS_LIB)
	@printf "Linking  %s\n" $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) $< $(NVLDFLAGS) $(NCCL_NVLDFLAGS) $(MPI_NVLDFLAGS)

${DST_DIR}/triple_comm_latecoll: triple_comm_latecoll.cu $(TEST_OS_LIB)
	@printf "Linking  %s\n" $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) $< $(NVLDFLAGS) $(NCCL_NVLDFLAGS) $(MPI_NVLDFLAGS)

${DST_DIR}/single_comm_mixedcoll: single_comm_mixedcoll.cu $(TEST_OS_LIB)
	@printf "Linking  %s\n" $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) $< $(NVLDFLAGS) $(NCCL_NVLDFLAGS) $(MPI_NVLDFLAGS)
