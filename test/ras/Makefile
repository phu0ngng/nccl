include ../../makefiles/common.mk

.PHONY: all clean

BUILDDIR := $(abspath ../../build)
NCCLDIR := $(BUILDDIR)
NVCUFLAGS += -I$(NCCLDIR)/include/ -I../include
NVCUFLAGS += -I$(MPI_HOME)/include
MPI_NVLDFLAGS = -L$(MPI_HOME)/lib -lmpi
NCCL_NVLDFLAGS = -L$(NCCLDIR)/lib -lnccl

DST_DIR := $(BUILDDIR)/test/ras

all: $(DST_DIR)/single_comm_idle $(DST_DIR)/triple_comm_idle \
     $(DST_DIR)/triple_comm_longidle $(DST_DIR)/single_comm_1exit \
     $(DST_DIR)/single_comm_1suspend $(DST_DIR)/single_comm_2suspend \
     $(DST_DIR)/single_comm_4suspend $(DST_DIR)/triple_comm_latecoll

clean:
	rm -rf $(DST_DIR)

NCCL_TOP := ../..

${DST_DIR}/single_comm_idle: single_comm_idle.cu
	@printf "Linking  %s\n" $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) $< $(NVLDFLAGS) $(NCCL_NVLDFLAGS) $(MPI_NVLDFLAGS)

${DST_DIR}/triple_comm_idle: triple_comm_idle.cu
	@printf "Linking  %s\n" $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) $< $(NVLDFLAGS) $(NCCL_NVLDFLAGS) $(MPI_NVLDFLAGS)

${DST_DIR}/triple_comm_longidle: triple_comm_longidle.cu
	@printf "Linking  %s\n" $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) $< $(NVLDFLAGS) $(NCCL_NVLDFLAGS) $(MPI_NVLDFLAGS)

${DST_DIR}/single_comm_1exit: single_comm_1exit.cu
	@printf "Linking  %s\n" $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) $< $(NVLDFLAGS) $(NCCL_NVLDFLAGS) $(MPI_NVLDFLAGS)

${DST_DIR}/single_comm_1suspend: single_comm_1suspend.cu
	@printf "Linking  %s\n" $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) $< $(NVLDFLAGS) $(NCCL_NVLDFLAGS) $(MPI_NVLDFLAGS)

${DST_DIR}/single_comm_2suspend: single_comm_2suspend.cu
	@printf "Linking  %s\n" $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) $< $(NVLDFLAGS) $(NCCL_NVLDFLAGS) $(MPI_NVLDFLAGS)

${DST_DIR}/single_comm_4suspend: single_comm_4suspend.cu
	@printf "Linking  %s\n" $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) $< $(NVLDFLAGS) $(NCCL_NVLDFLAGS) $(MPI_NVLDFLAGS)

${DST_DIR}/triple_comm_latecoll: triple_comm_latecoll.cu
	@printf "Linking  %s\n" $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) $< $(NVLDFLAGS) $(NCCL_NVLDFLAGS) $(MPI_NVLDFLAGS)
