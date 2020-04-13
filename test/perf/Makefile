include ../../makefiles/common.mk

.PHONY: all clean

BUILDDIR := $(abspath ../../build)
NCCLDIR := $(BUILDDIR)
NVCUFLAGS += -I$(NCCLDIR)/include/ -I../include
ifeq ($(MPI), 1)
NVCUFLAGS += -DMPI_SUPPORT -I$(MPI_HOME)/include
NVLDFLAGS += -L$(MPI_HOME)/lib -lmpi
endif
ifeq ($(MPI_IBM),1)
NVCUFLAGS += -DMPI_SUPPORT
NVLDFLAGS += -lmpi_ibm
endif
LIBRARIES += nccl
NVLDFLAGS += -L$(NCCLDIR)/lib $(LIBRARIES:%=-l%)

# Setting MEMLEAK=1 enables simply memory leak detection using clang++
ifeq ($(MEMLEAK),1)
CXX=clang++
NVLDFLAGS += --compiler-options="-fsanitize=leak"
endif

DST_DIR := $(BUILDDIR)/test/perf
SRC_FILES := $(wildcard *.cu)
OBJ_FILES := $(SRC_FILES:%.cu=${DST_DIR}/%.o)
COLLS := all_reduce all_gather broadcast reduce_scatter reduce alltoall gather scatter
EXTRA_COLLS := all_gatherv reduce_scatterv
COLL_OBJS := $(COLLS:%=${DST_DIR}/%.o)
BIN_FILES := $(COLLS:%=${DST_DIR}/%_perf) $(EXTRA_COLLS:%=${DST_DIR}/%_perf) ${DST_DIR}/replay
ifeq ($(MPI_COLLNET), 1)
BIN_FILES += $(COLLS:%=${DST_DIR}/%_perf_mpi_collnet)
endif
DEP_FILES := $(OBJ_FILES:%.o=%.dep)

all: ${BIN_FILES}

clean:
	rm -rf ${DST_DIR}

ifneq (clean,${MAKECMDGOALS})
-include $(DEP_FILES)
${DST_DIR}/%.dep: %.cu
	@mkdir -p ${DST_DIR}
	$(NVCC) --generate-dependencies $(NVCUFLAGS) $< \
	| sed "0,/^$*.o :/s//$(subst /,\/,${DST_DIR}/$*.o) :/" \
	> $@
endif

${DST_DIR}/%.o: %.cu
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) -c $<

${DST_DIR}/common_mpi_collnet.o:common.cu
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -DMPI_COLLNET_SUPPORT -o $@ $(NVCUFLAGS) -c $<

${DST_DIR}/nccl_mpi.o: ../../share/nccl_mpi.c
	@printf "Compiling   %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	mpicc -std=c99 $(CXXFLAGS) -I$(NCCLDIR)/include/ -o $@ -c $<

${DST_DIR}/nccl_mpi_collnet.o: ../../share/nccl_mpi_collnet.c
	@printf "Compiling   %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	mpicc -std=c99 $(CXXFLAGS) -I$(NCCLDIR)/include/ -o $@ -c $<

${DST_DIR}/%_perf:${DST_DIR}/%.o ${DST_DIR}/common.o
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) $^ ${NVLDFLAGS}

${DST_DIR}/replay: ${DST_DIR}/replay.o ${COLL_OBJS} ${DST_DIR}/common.o
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) $^ ${NVLDFLAGS}

${DST_DIR}/%_perf_mpi_collnet: ${DST_DIR}/%.o ${DST_DIR}/common_mpi_collnet.o ${DST_DIR}/nccl_mpi_collnet.o
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) $^ ${NVLDFLAGS}

