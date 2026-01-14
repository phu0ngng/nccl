.DEFAULT_GOAL=all
include ../../../makefiles/common.mk
SRCDIR   ?= $(abspath ../../..)
BUILDDIR ?= $(abspath ../../../build)
NCCLDIR := $(BUILDDIR)
GTEST_DIR ?= $(abspath ../../googletest/googletest)
include ../../gtest.mk

### files
SRC_FILES   := $(wildcard *.cu)
COMMON_SRC  := ../ncclCommon_test.cu
DST_DIR     := $(BUILDDIR)/test/apitest/device_api
OBJ_FILES   := $(SRC_FILES:%.cu=${DST_DIR}/%.o) ${DST_DIR}/ncclCommon_test.o
DEP_FILES   := $(OBJ_FILES:%.o=%.dep)

### cc
NVCUFLAGS += -I${NCCLDIR}/include/ -I$(CUDA_INC) -I.. -include nccl.h ${CPPFLAGS} ${GTEST_CPPFLAGS} -g

### ld
LIBDIR := ${NCCLDIR}/lib
NVLDFLAGS := -L${LIBDIR} ${NVLDFLAGS} ${GTEST_LIBS}

### rules
.PHONY: all clean test

ifneq (clean,${MAKECMDGOALS})
-include ${DEP_FILES}
${DST_DIR}/%.dep:%.cu
	@mkdir -p ${DST_DIR}
	$(NVCC) --generate-dependencies ${NVCUFLAGS} $< \
	| sed "0,/^$*.o :/s//$(subst /,\/,${DST_DIR}/$*.o) :/" \
	> $@

${DST_DIR}/ncclCommon_test.dep: ${COMMON_SRC}
	@mkdir -p ${DST_DIR}
	$(NVCC) --generate-dependencies ${NVCUFLAGS} $< \
	| sed "0,/^ncclCommon_test.o :/s//$(subst /,\/,${DST_DIR}/ncclCommon_test.o) :/" \
	> $@
endif

all: ${DST_DIR}/device_api_test

clean: gtest.clean
	rm -rf ${DST_DIR}

# Link all object files into single binary
${DST_DIR}/device_api_test: ${GTEST_LIBS} ${OBJ_FILES}
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	${NVCC} -o $@ ${NVCUFLAGS} ${NVLDFLAGS} $^ -lnccl

# Compile .cu to .o
${DST_DIR}/%.o: %.cu
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ ${NVCUFLAGS} --compile $<

# Compile ncclCommon_test.cu from parent directory
${DST_DIR}/ncclCommon_test.o: ${COMMON_SRC}
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ ${NVCUFLAGS} --compile $<

# Run tests
test: ${DST_DIR}/device_api_test
	LD_LIBRARY_PATH+=:$(LIBDIR) $^


