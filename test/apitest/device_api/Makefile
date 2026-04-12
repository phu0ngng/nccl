.DEFAULT_GOAL=all
include ../../../makefiles/common.mk
SRCDIR   ?= $(abspath ../../..)
BUILDDIR ?= $(abspath ../../../build)
NCCLDIR := $(BUILDDIR)
GTEST_DIR ?= $(abspath ../../googletest/googletest)
include ../../gtest.mk

### files
REDUCECOPY_GEN_DIR := $(BUILDDIR)/test/apitest/device_api/reduceCopy/generated
REDUCECOPY_GEN_SCRIPT := reduceCopy/generate_tests.py
REDUCECOPY_TYPES_ENV_STAMP := $(REDUCECOPY_GEN_DIR)/.types_env.stamp
REDUCECOPY_GEN_FILES_ABS := $(shell NCCL_TEST_REDUCE_COPY_TYPES="$(NCCL_TEST_REDUCE_COPY_TYPES)" python3 $(REDUCECOPY_GEN_SCRIPT) --output-dir $(REDUCECOPY_GEN_DIR) --list-files)
REDUCECOPY_GEN_FILES := $(patsubst $(BUILDDIR)/test/apitest/device_api/%,%,$(REDUCECOPY_GEN_FILES_ABS))

DST_DIR     := $(BUILDDIR)/test/apitest/device_api
VPATH += $(DST_DIR)

SRC_FILES   := $(wildcard *.cu) $(wildcard reduceCopy/*.cu) $(REDUCECOPY_GEN_FILES)
COMMON_SRC  := ../ncclCommon_test.cu
OBJ_FILES   := $(patsubst %.cu,${DST_DIR}/%.o,$(SRC_FILES)) ${DST_DIR}/ncclCommon_test.o
DEP_FILES   := $(OBJ_FILES:%.o=%.dep)

### cc
NVCUFLAGS += -I${NCCLDIR}/include/ -I$(CUDA_INC) -I.. -IreduceCopy -include nccl.h ${CPPFLAGS} ${GTEST_CPPFLAGS} -g
NVCUFLAGS += -DNCCL_DEVICE_PERMIT_EXPERIMENTAL_CODE=1
# Allow constexpr host functions (e.g. std::array::operator[]) to be called from
# __host__ __device__ constexpr contexts (needed for kApiFunctionTraitsTable access
# in hasMultimemSourceConstexpr / hasMultimemDestinationConstexpr used in kernels).
NVCUFLAGS += --expt-relaxed-constexpr

# fp8 multimem arch macros (match apitest)
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
ifneq ($(HAS_COMPUTE_100F),)
  ifneq ($(HAS_SM_100F),)
    NVCUFLAGS += -DNCCL_BUILD_SM100F=1
  endif
endif
ifneq ($(HAS_COMPUTE_100A),)
  ifneq ($(HAS_SM_100A),)
    NVCUFLAGS += -DNCCL_BUILD_SM100A=1
  endif
endif
ifneq ($(HAS_COMPUTE_101F),)
  ifneq ($(HAS_SM_101F),)
    NVCUFLAGS += -DNCCL_BUILD_SM101F=1
  endif
endif
ifneq ($(HAS_COMPUTE_101A),)
  ifneq ($(HAS_SM_101A),)
    NVCUFLAGS += -DNCCL_BUILD_SM101A=1
  endif
endif
ifneq ($(HAS_COMPUTE_120A),)
  ifneq ($(HAS_SM_120A),)
    NVCUFLAGS += -DNCCL_BUILD_SM120A=1
  endif
endif
ifneq ($(HAS_COMPUTE_121A),)
  ifneq ($(HAS_SM_121A),)
    NVCUFLAGS += -DNCCL_BUILD_SM121A=1
  endif
endif

### ld
LIBDIR := ${NCCLDIR}/lib
NVLDFLAGS := -L${LIBDIR} ${NVLDFLAGS} ${GTEST_LIBS}

### rules
.PHONY: all clean test

ifneq (clean,${MAKECMDGOALS})
-include ${DEP_FILES}
${DST_DIR}/%.dep: %.cu
	@mkdir -p $(dir $@)
	@OBJ_BASENAME=$(notdir $*.o); OBJ_FULLPATH=$(subst /,\/,${DST_DIR}/$*.o); \
	$(NVCC) --generate-dependencies ${NVCUFLAGS} $< 2>/dev/null | \
	sed -E "1s|^[^:]*$$OBJ_BASENAME([^:]*):|$$OBJ_FULLPATH\1:|" > $@ 2>/dev/null || \
	$(NVCC) --generate-dependencies ${NVCUFLAGS} $< | sed "0,/^$*.o :/s//$(subst /,\/,${DST_DIR}/$*.o) :/" > $@

${DST_DIR}/ncclCommon_test.dep: ${COMMON_SRC}
	@mkdir -p ${DST_DIR}
	$(NVCC) --generate-dependencies ${NVCUFLAGS} $< \
	| sed "0,/^ncclCommon_test.o :/s//$(subst /,\/,${DST_DIR}/ncclCommon_test.o) :/" \
	> $@
endif

reducecopy_gen: $(REDUCECOPY_GEN_DIR)/.stamp

$(REDUCECOPY_GEN_DIR)/.stamp: $(REDUCECOPY_GEN_SCRIPT) reduceCopy/test_reduce_copy.h reduceCopy/config.h $(REDUCECOPY_TYPES_ENV_STAMP)
	@mkdir -p $(REDUCECOPY_GEN_DIR)
	@NCCL_TEST_REDUCE_COPY_TYPES="$(NCCL_TEST_REDUCE_COPY_TYPES)" python3 $(REDUCECOPY_GEN_SCRIPT) --output-dir $(REDUCECOPY_GEN_DIR)
	@touch $@

$(REDUCECOPY_TYPES_ENV_STAMP):
	@mkdir -p $(REDUCECOPY_GEN_DIR)
	@tmpfile="$@.tmp"; \
	printf "%s\n" "$(NCCL_TEST_REDUCE_COPY_TYPES)" > "$$tmpfile"; \
	if [ -f "$@" ] && cmp -s "$@" "$$tmpfile"; then rm -f "$$tmpfile"; else mv -f "$$tmpfile" "$@"; fi

# Generated .cu files are produced by the stamp; make must build the stamp before compiling them
$(REDUCECOPY_GEN_FILES_ABS): $(REDUCECOPY_GEN_DIR)/.stamp
	@:

all: ${DST_DIR}/device_api_test

clean: gtest.clean
	rm -rf ${DST_DIR}

# Link all object files into single binary
${DST_DIR}/device_api_test: ${GTEST_LIBS} ${OBJ_FILES}
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	${NVCC} -o $@ ${NVCUFLAGS} ${NVLDFLAGS} $^ -lcuda -lnccl

# Compile generated .cu (in build tree) to .o — use full path so nvcc finds the file
${DST_DIR}/reduceCopy/generated/%.o: $(REDUCECOPY_GEN_DIR)/%.cu | reducecopy_gen
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p $(dir $@)
	$(NVCC) -o $@ ${NVCUFLAGS} --compile $<

# Compile .cu to .o (including reduceCopy; generated files use the rule above)
${DST_DIR}/%.o: %.cu
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p $(dir $@)
	$(NVCC) -o $@ ${NVCUFLAGS} --compile $<

# Compile ncclCommon_test.cu from parent directory
${DST_DIR}/ncclCommon_test.o: ${COMMON_SRC}
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ ${NVCUFLAGS} --compile $<

# Run tests
test: ${DST_DIR}/device_api_test
	LD_LIBRARY_PATH+=:$(LIBDIR) $^


