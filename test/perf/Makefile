include ../../makefiles/common.mk

.PHONY: all clean

BUILDDIR := $(abspath ../../build)
NCCLDIR := $(BUILDDIR)
NVCUFLAGS += -I$(NCCLDIR)/include/ -I../include -g
ifeq ($(MPI), 1)
NVCUFLAGS += -DMPI_SUPPORT -I$(MPI_HOME)/include
NVLDFLAGS += -L$(MPI_HOME)/lib -lmpi
endif
ifeq ($(MPI_IBM),1)
NVCUFLAGS += -DMPI_SUPPORT
NVLDFLAGS += -lmpi_ibm
endif
LIBRARIES += nccl
NVLDFLAGS += -L$(NCCLDIR)/lib $(LIBRARIES:%=-l%) -lcuda
RPATHS = $$ORIGIN

# Setting MEMLEAK=1 enables simply memory leak detection using clang++
ifeq ($(MEMLEAK),1)
CXX=clang++
NVLDFLAGS += --compiler-options="-fsanitize=leak"
endif

DST_DIR := $(BUILDDIR)/test/perf
SRC_FILES := $(wildcard *.cu)
OBJ_FILES := $(SRC_FILES:%.cu=${DST_DIR}/%.o)
COLLS := all_reduce all_gather broadcast reduce_scatter reduce alltoall gather scatter sendrecv hypercube
EXTRA_COLLS := all_gatherv reduce_scatterv
COLL_OBJS := $(COLLS:%=${DST_DIR}/%.o)
BIN_FILES := $(COLLS:%=${DST_DIR}/%_perf) $(EXTRA_COLLS:%=${DST_DIR}/%_perf) ${DST_DIR}/replay-coll
ifeq ($(MPI_COLLNET), 1)
BIN_FILES += $(COLLS:%=${DST_DIR}/%_perf_mpi_collnet)
endif
DEP_FILES := $(OBJ_FILES:%.o=%.dep)

all: ${BIN_FILES}

clean:
	rm -rf ${DST_DIR}

TEST_VERIFIABLE_SRCDIR := ../verifiable
TEST_VERIFIABLE_BUILDDIR := $(BUILDDIR)/test/verifiable
include ../verifiable/verifiable.mk

COMMON_OBJS := ${DST_DIR}/common.o ${DST_DIR}/timer.o ${DST_DIR}/ft_test.o

NVLDFLAGS += -L$(TEST_VERIFIABLE_BUILDDIR) -lverifiable
RPATHS := $(RPATHS):$$ORIGIN/../verifiable

NVLDFLAGS += -Xlinker -rpath='$(RPATHS)'

ifneq (clean,${MAKECMDGOALS})
-include $(DEP_FILES)
${DST_DIR}/%.dep: %.cu
	@mkdir -p ${DST_DIR}
	$(NVCC) --generate-dependencies $(NVCUFLAGS) $< \
	| sed "0,/^$*.o :/s//$(subst /,\/,${DST_DIR}/$*.o) :/" \
	> $@
endif

${DST_DIR}/%.o: %.cu $(TEST_VERIFIABLE_HDRS)
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) -c $<

${DST_DIR}/timer.o: timer.cc timer.h
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(CXX) $(CXXFLAGS) -o $@ -c timer.cc

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

$(DST_DIR)/libverifiable.so: $(TEST_VERIFIABLE_LIBSO)
	ln -sfn ../verifiable/libverifiable.so $@

${DST_DIR}/%_perf:${DST_DIR}/%.o $(COMMON_OBJS) $(DST_DIR)/libverifiable.so
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	@ln -sfn $(DST_DIR)/../verifiable/libverifiable.so $(DST_DIR)/libverifiable.so
	$(NVCC) -o $@ $(NVCUFLAGS) $^ $(NVLDFLAGS)

${DST_DIR}/replay-coll: ${DST_DIR}/replay-coll.o ${COLL_OBJS} ${COMMON_OBJS} $(TEST_VERIFIABLE_LIBSO)
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) ${DST_DIR}/replay-coll.o ${COLL_OBJS} ${COMMON_OBJS} $(NVLDFLAGS)

${DST_DIR}/%_perf_mpi_collnet: ${DST_DIR}/%.o ${DST_DIR}/common_mpi_collnet.o ${DST_DIR}/nccl_mpi_collnet.o
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) $^ $(NVLDFLAGS)
