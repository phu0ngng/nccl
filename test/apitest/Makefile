.DEFAULT_GOAL=all
include ../../makefiles/common.mk
SRCDIR   ?= $(abspath ../..)
BUILDDIR ?= $(abspath ../../build)
NCCLDIR := $(BUILDDIR)
GTEST_DIR ?= $(abspath ../googletest/googletest)
include ../gtest.mk
### files
DST_DIR		:= $(BUILDDIR)/test/apitest
VPATH += $(DST_DIR)

SRC_FILES	:= $(wildcard *.cu)
OBJ_FILES	:= $(patsubst %.cu,${DST_DIR}/%.o,$(SRC_FILES))
DEP_FILES	:= $(OBJ_FILES:%.o=%.dep)

### plugin libraries
PLUGIN_SRC:= $(basename $(wildcard plugin/*.cc))
PLUGIN_OBJ:= $(PLUGIN_SRC:%.cc=${DST_DIR}/plugin/libnccl-%.so)

### cc
# Permit unsafe device code in apitest (explicit opt-in).
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

NVCUFLAGS += -I${NCCLDIR}/include/ -I$(CUDA_INC) -I. -I../include -include nccl.h ${CPPFLAGS} ${GTEST_CPPFLAGS} -g
### ld
LIBDIR := ${NCCLDIR}/lib
NVLDFLAGS := -L${LIBDIR} ${NVLDFLAGS} ${GTEST_LIBS}
### rules
.PHONY: all clean test device_api

ifneq (clean,${MAKECMDGOALS})
-include ${DEP_FILES}
${DST_DIR}/%.dep: %.cu
	@mkdir -p $(dir ${DST_DIR}/$*.dep)
	@OBJ_BASENAME=$(notdir $*.o); \
	OBJ_FULLPATH=$(subst /,\/,${DST_DIR}/$*.o); \
	$(NVCC) --generate-dependencies ${NVCUFLAGS} $< 2>/dev/null | \
	sed -E "1s|^[^:]*$$OBJ_BASENAME([^:]*):|$$OBJ_FULLPATH\1:|" \
	> ${DST_DIR}/$*.dep 2>/dev/null || \
	$(NVCC) --generate-dependencies ${NVCUFLAGS} $< > ${DST_DIR}/$*.dep
endif
all:${DST_DIR}/apitest ${DST_DIR}/apitest_static \
    ${DST_DIR}/plugin/libnccl-net-plugin_v6.so   \
    ${DST_DIR}/plugin/libnccl-net-plugin_v7.so   \
    ${DST_DIR}/plugin/libnccl-net-plugin_v8.so   \
    ${DST_DIR}/plugin/libnccl-net-plugin_v9.so   \
    ${DST_DIR}/plugin/libnccl-net-plugin_v10.so   \
    ${DST_DIR}/plugin/libnccl-net-plugin_v11.so   \
    ${DST_DIR}/plugin/libnccl-net-plugin_v12.so   \
    ${DST_DIR}/plugin/libnccl-net-plugin_nodev_v6.so   \
    ${DST_DIR}/plugin/libnccl-net-plugin_nodev_v7.so   \
    ${DST_DIR}/plugin/libnccl-net-plugin_nodev_v8.so   \
    ${DST_DIR}/plugin/libnccl-net-plugin_nodev_v9.so   \
    ${DST_DIR}/plugin/libnccl-net-plugin_nodev_v10.so   \
    ${DST_DIR}/plugin/libnccl-net-plugin_nodev_v11.so   \
    ${DST_DIR}/plugin/libnccl-net-plugin_nodev_v12.so   \
    ${DST_DIR}/plugin/libnccl-shared-plugins.so \
    ${DST_DIR}/plugin/libnccl-env-plugin.so \
    ${DST_DIR}/plugin/libnccl-net-plugin-init-once.so \
    ${DST_DIR}/plugin/libnccl-gin-plugin_v11.so \
    device_api

device_api:
	${MAKE} -C device_api BUILDDIR=${BUILDDIR}

env.%:
	${MAKE} -C $(SRCDIR)/plugins/env/example $* BUILDDIR=$(DST_DIR)/plugin/libnccl-env-plugin.so

clean:gtest.clean
	rm -rf ${DST_DIR}
	${MAKE} -C device_api clean

test: ${DST_DIR}/apitest
	/bin/bash -c "LD_LIBRARY_PATH+=:$(LIBDIR):${NCCLDIR}/test/apitest/plugin $^"

${DST_DIR}/apitest:${GTEST_LIBS} ${OBJ_FILES}
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	${NVCC} -o $@ ${NVCUFLAGS} ${NVLDFLAGS} $^ -lnccl

${DST_DIR}/apitest_static:${GTEST_LIBS} ${OBJ_FILES}
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	${NVCC} -o $@ ${NVCUFLAGS} ${NVLDFLAGS} $^ -lnccl_static

${DST_DIR}/%.o: %.cu
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p $(dir $@)
	$(NVCC) -o $@ ${NVCUFLAGS} --compile $<

${DST_DIR}/plugin/libnccl-%.so: plugin/%.cc
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}/plugin
	${CXX} -I${SRCDIR}/plugins/net/example/nccl -I${SRCDIR}/plugins/profiler/example/nccl -I${SRCDIR}/plugins/tuner/example/nccl -I$(CUDA_INC) -fPIC -fvisibility=default -shared -o $@ $^

${DST_DIR}/plugin/libnccl-env-plugin.so: env.build
	@echo "Environment plugin built via env.build target"
