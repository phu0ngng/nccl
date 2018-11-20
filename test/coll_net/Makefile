include ../../makefiles/common.mk

.PHONY: all clean

BUILDDIR := $(abspath ../../build)
NVCUFLAGS += -I$(BUILDDIR)/include/ -I../include
ifeq ($(MPI), 1)
NVCUFLAGS += -DMPI_SUPPORT -I$(MPI_HOME)/include
NVLDFLAGS += -L$(MPI_HOME)/lib -lmpi
endif
ifeq ($(MPI_IBM),1)
NVCUFLAGS += -DMPI_SUPPORT
NVLDFLAGS += -lmpi_ibm
endif
LIBRARIES += curand nccl
NVLDFLAGS += -L$(BUILDDIR)/lib $(LIBRARIES:%=-l%)

DST_DIR := $(BUILDDIR)/test/coll_net
SRC_FILES := $(wildcard *.cu)
OBJ_FILES := $(SRC_FILES:%.cu=${DST_DIR}/%.o)
BIN_FILES_LIST := multiproc_mpi_collnet
BIN_FILES := $(BIN_FILES_LIST:%=${DST_DIR}/%)
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

${DST_DIR}/nccl_mpi_collnet.o: ../../share/nccl_mpi_collnet.c
	@printf "Compiling   %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	mpicc -std=c99 $(CXXFLAGS) -I$(BUILDDIR)/include/ -I../../src/include/ -o $@ -c $<

${DST_DIR}/%: ${DST_DIR}/%.o ${DST_DIR}/nccl_mpi_collnet.o
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) $^ ${NVLDFLAGS}

