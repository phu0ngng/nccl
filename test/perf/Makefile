include ../../makefiles/common.mk

.PHONY: all clean

# Permit unsafe device code in perf tests (explicit opt-in).
CXXFLAGS += -DNCCL_DEVICE_PERMIT_EXPERIMENTAL_CODE=1
NVCUFLAGS += -DNCCL_DEVICE_PERMIT_EXPERIMENTAL_CODE=1

# Check NVCC_GENCODE for fp8 multimem architectures that match multimem__funcs.h logic:
# __CUDA_ARCH_SPECIFIC__ == 1000 (sm_100a), 1010 (sm_101a), 1200 (sm_120a), 1210 (sm_121a)
# __CUDA_ARCH_FAMILY_SPECIFIC__ == 1000 (sm_100f+), 1010 (sm_101f+)
# Check for both compute_XXX AND sm_XXX to match gencode flags like: -gencode=arch=compute_100a,code=sm_100a
# If both are present, set compile-time macros to indicate fp8 multimem is available
# Priority: sm_100f > sm_100a > sm_101f > sm_101a > sm_120a > sm_121a
# Use findstring to check if strings are present (works on multi-word variables)
HAS_COMPUTE_100F := $(findstring compute_100f,$(NVCC_GENCODE))
HAS_SM_100F := $(findstring sm_100f,$(NVCC_GENCODE))
HAS_COMPUTE_100A := $(findstring compute_100a,$(NVCC_GENCODE))
HAS_SM_100A := $(findstring sm_100a,$(NVCC_GENCODE))
HAS_COMPUTE_101F := $(findstring compute_101f,$(NVCC_GENCODE))
HAS_SM_101F := $(findstring sm_101f,$(NVCC_GENCODE))
HAS_COMPUTE_101A := $(findstring compute_101a,$(NVCC_GENCODE))
HAS_SM_101A := $(findstring sm_101a,$(NVCC_GENCODE))
HAS_COMPUTE_120A := $(findstring compute_120a,$(NVCC_GENCODE))
HAS_SM_120A := $(findstring sm_120a,$(NVCC_GENCODE))
HAS_COMPUTE_121A := $(findstring compute_121a,$(NVCC_GENCODE))
HAS_SM_121A := $(findstring sm_121a,$(NVCC_GENCODE))

# Check each pair independently - set macro if both compute_XXX and sm_XXX are present
# Check in priority order, but each check is independent
ifneq ($(HAS_COMPUTE_100F),)
  ifneq ($(HAS_SM_100F),)
    CXXFLAGS += -DNCCL_BUILD_SM100F=1
    NVCUFLAGS += -DNCCL_BUILD_SM100F=1
  endif
endif
ifneq ($(HAS_COMPUTE_100A),)
  ifneq ($(HAS_SM_100A),)
    CXXFLAGS += -DNCCL_BUILD_SM100A=1
    NVCUFLAGS += -DNCCL_BUILD_SM100A=1
  endif
endif
ifneq ($(HAS_COMPUTE_101F),)
  ifneq ($(HAS_SM_101F),)
    CXXFLAGS += -DNCCL_BUILD_SM101F=1
    NVCUFLAGS += -DNCCL_BUILD_SM101F=1
  endif
endif
ifneq ($(HAS_COMPUTE_101A),)
  ifneq ($(HAS_SM_101A),)
    CXXFLAGS += -DNCCL_BUILD_SM101A=1
    NVCUFLAGS += -DNCCL_BUILD_SM101A=1
  endif
endif
ifneq ($(HAS_COMPUTE_120A),)
  ifneq ($(HAS_SM_120A),)
    CXXFLAGS += -DNCCL_BUILD_SM120A=1
    NVCUFLAGS += -DNCCL_BUILD_SM120A=1
  endif
endif
ifneq ($(HAS_COMPUTE_121A),)
  ifneq ($(HAS_SM_121A),)
    CXXFLAGS += -DNCCL_BUILD_SM121A=1
    NVCUFLAGS += -DNCCL_BUILD_SM121A=1
  endif
endif

PROFDIR := $(abspath ../../plugins/profiler/example)
BUILDDIR := $(abspath ../../build)
NCCLDIR := $(BUILDDIR)
NVCUFLAGS += -I$(NCCLDIR)/include/ -I../include -g -I$(PROFDIR)/nccl -I$(PROFDIR)
ifeq ($(MPI), 1)
NVCUFLAGS += -DMPI_SUPPORT -I$(MPI_HOME)/include
NVLDFLAGS += -L$(MPI_HOME)/lib -lmpi
endif
ifeq ($(MPI_IBM),1)
NVCUFLAGS += -DMPI_SUPPORT
NVLDFLAGS += -lmpi_ibm
endif
LIBRARIES += nccl m stdc++
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

all: profiler.build ${BIN_FILES}

clean:
	rm -rf ${DST_DIR}

TEST_VERIFIABLE_SRCDIR := ../verifiable
TEST_VERIFIABLE_BUILDDIR := $(BUILDDIR)/test/verifiable
include ../verifiable/verifiable.mk

COMMON_OBJS := ${DST_DIR}/common.o ${DST_DIR}/util.o ${DST_DIR}/timer.o ${DST_DIR}/ft_test.o

NVLDFLAGS += -L$(TEST_VERIFIABLE_BUILDDIR) -lverifiable
RPATHS := $(RPATHS):$$ORIGIN/../verifiable

NVLDFLAGS += -Xlinker -rpath='$(RPATHS)' -lm -lstdc++

ifneq (clean,${MAKECMDGOALS})
-include $(DEP_FILES)
${DST_DIR}/%.dep: %.cu
	@mkdir -p ${DST_DIR}
	$(NVCC) --generate-dependencies $(NVCUFLAGS) $< \
	| sed "0,/^$*.o :/s//$(subst /,\/,${DST_DIR}/$*.o) :/" \
	> $@
endif

profiler.%:
	${MAKE} -C $(PROFDIR) $* BUILDDIR=$(BUILDDIR)/lib

${DST_DIR}/%.o: %.cu common.h util.h $(TEST_VERIFIABLE_HDRS)
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

${DST_DIR}/%_perf:${DST_DIR}/%.o $(COMMON_OBJS) $(TEST_VERIFIABLE_LIBSO)
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) $< $(COMMON_OBJS) $(NVLDFLAGS)

${DST_DIR}/replay-coll: ${DST_DIR}/replay-coll.o ${COLL_OBJS} ${COMMON_OBJS} $(TEST_VERIFIABLE_LIBSO)
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) ${DST_DIR}/replay-coll.o ${COLL_OBJS} ${COMMON_OBJS} $(NVLDFLAGS)

${DST_DIR}/%_perf_mpi_collnet: ${DST_DIR}/%.o ${DST_DIR}/common_mpi_collnet.o ${DST_DIR}/nccl_mpi_collnet.o
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) $^ $(NVLDFLAGS)
