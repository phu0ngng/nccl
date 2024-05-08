.DEFAULT_GOAL=all
include ../../makefiles/common.mk
BUILDDIR ?= $(abspath ../../build)
NCCLDIR := $(BUILDDIR)
GTEST_DIR ?= $(abspath ../googletest/googletest)
include ../gtest.mk
### files
SRC_FILES	:= $(wildcard *.cu)
DST_DIR		:= $(BUILDDIR)/test/apitest
OBJ_FILES	:= $(SRC_FILES:%.cu=${DST_DIR}/%.o)
DEP_FILES	:= $(OBJ_FILES:%.o=%.dep)
### cc
NVCUFLAGS += -I${NCCLDIR}/include/ -include nccl.h ${CPPFLAGS} ${GTEST_CPPFLAGS} -g
### ld
LIBDIR := ${NCCLDIR}/lib
NVLDFLAGS := -L${LIBDIR} ${NVLDFLAGS} ${GTEST_LIBS}
### rules
.PHONY:

ifneq (clean,${MAKECMDGOALS})
-include ${DEP_FILES}
${DST_DIR}/%.dep:%.cu
	@mkdir -p ${DST_DIR}
	$(NVCC) --generate-dependencies ${NVCUFLAGS} $< \
	| sed "0,/^$*.o :/s//$(subst /,\/,${DST_DIR}/$*.o) :/" \
	> $@
endif
all:${DST_DIR}/apitest ${DST_DIR}/apitest_static

clean:
	rm -rf ${DST_DIR}

test: ${DST_DIR}/apitest
	/bin/bash -c "LD_LIBRARY_PATH+=:$(LIBDIR) $^"

${DST_DIR}/apitest:${OBJ_FILES}
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	${NVCC} -o $@ ${NVCUFLAGS} ${NVLDFLAGS} $^ -lnccl

${DST_DIR}/apitest_static:${OBJ_FILES}
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	${NVCC} -o $@ ${NVCUFLAGS} ${NVLDFLAGS} $^ -lnccl_static

${DST_DIR}/%.o: %.cu
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ ${NVCUFLAGS} --compile $<
