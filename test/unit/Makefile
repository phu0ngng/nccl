include ../../makefiles/common.mk
##
BUILDDIR ?= $(abspath ../../build)
### mpi lib
ifeq ($(MPI),1)
MPI_HOME ?= $(realpath $(dir $(or \
	$(realpath /usr/include/mpi.h),\
	$(realpath /usr/lib/openmpi/include/mpi.h)))..)
NVLDFLAGS += -L$(MPI_HOME)/lib -lmpi
NVCUFLAGS += -g -I$(MPI_HOME)/include
endif
ifeq ($(MPI_IBM),1)
NVLDFLAGS += -lmpi_ibm
endif
# We need private functions. Include and link with src
SRCINC = -I$(BUILDDIR)/include/ -I../include -I../../src/include -I ../../src/device/ -I../../src/graph
# Use static library since all the internal symbols are there
DLIB = ${BUILDDIR}/lib/libnccl.so
DLIBFLAGS = $(NVLDFLAGS) -L$(BUILDDIR)/lib -lnccl -lcudart
LIBRARIES += nccl_static
STATICLIB = ${BUILDDIR}/lib/libnccl_static.a
NVLDFLAGS += -L$(BUILDDIR)/lib $(LIBRARIES:%=-l%)
##
DST_DIR := $(abspath ${BUILDDIR}/test/unit)
CU_FILES := align_test.cu ft_test.cu #kernel_test.cu
CC_FILES += graph_test.cc model.cc data.cc xml_test.cc comm_leak_test.cc channel_test.cc
KCU_FILES := align_test.cu #kernel_test.cu
ifeq ($(MPI),1)
#CU_FILES += ncclNet_test.cu ncclNet_read_write_stress_test.cu
CU_FILES += comm_leak_test_multi.cu coll_leak_test_multi.cu
endif
OBJ_FILES:= $(CU_FILES:%.cu=${DST_DIR}/%.o) $(CC_FILES:%.cc=${DST_DIR}/%.o)
EXE_FILES:= $(CU_FILES:%.cu=${DST_DIR}/%) $(CC_FILES:%.cc=${DST_DIR}/%)
KEXE_FILES:= $(KCU_FILES:%.cu=${DST_DIR}/%)
DEP_FILES:= $(OBJ_FILES:%.o=${DST_DIR}/%.d)

LSB_RELEASE=$(shell lsb_release -is)
#
all:${EXE_FILES}
	@printf "Syncing  %-35s > %s\n" topo ${DST_DIR}
	rsync -q -av topo ${DST_DIR}

kernel: ${KEXE_FILES}

clean:
	rm -rf ${DST_DIR}
#
ifneq (clean,${MAKECMDGOALS})
-include ${DEP_FILES}
endif
#
${DST_DIR}/%.o:%.cc
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(CXX) $(SRCINC) $(CXXFLAGS) -c $< -o $@
	@$(CXX) $(SRCINC) $(CXXFLAGS) -M $< > $(@:%.o=%.d.tmp)
	@sed "0,/^.*:/s//$(subst /,\/,$@):/" $(@:%.o=%.d.tmp) > $(@:%.o=%.d)
	@sed -e 's/.*://' -e 's/\\$$//' < $(@:%.o=%.d.tmp) | fmt -1 | \
                sed -e 's/^ *//' -e 's/$$/:/' >> $(@:%.o=%.d)
	@rm -f $(@:%.o=%.d.tmp)
#
${DST_DIR}/%.o:%.cu
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) --compile -o $@ $(SRCINC) $(NVCUFLAGS) $<
#
${DST_DIR}/kernel_test: ${DST_DIR}/kernel_test.o
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) ${NVLDFLAGS} $^ -lcudart
#
${DST_DIR}/align_test: ${DST_DIR}/align_test.o
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) ${NVLDFLAGS} $^ -lcudart
#
#${DST_DIR}/nccl_mpi.o: ../../share/nccl_mpi.c
#	@printf "Compiling   %-35s > %s\n" $< $@
#	@mkdir -p ${DST_DIR}
#	mpicc -std=c99 $(CXXFLAGS) -I$(BUILDDIR)/include/ -o $@ -c $<
#
${DST_DIR}/ft_test: ${DST_DIR}/ft_test.o ${DLIB}
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) $^ ${DLIBFLAGS}
#
${DST_DIR}/graph_test: ${DST_DIR}/graph_test.o ${STATICLIB}
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) $^ ${NVLDFLAGS}
#
${DST_DIR}/model: ${DST_DIR}/model.o ${STATICLIB}
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) $^ ${NVLDFLAGS}
#
${DST_DIR}/xml_test: ${DST_DIR}/xml_test.o ${STATICLIB}
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) $^ ${NVLDFLAGS}
#
${DST_DIR}/ncclNet_test: ${DST_DIR}/ncclNet_test.o ${LIBOBJS}
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) ${NVLDFLAGS} $^
#
${DST_DIR}/ncclNet_read_write_stress_test: ${DST_DIR}/ncclNet_read_write_stress_test.o ${STATICLIB}
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) ${NVLDFLAGS} $^

${DST_DIR}/comm_leak_test: ${DST_DIR}/comm_leak_test.o
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	@if [ "$(LSB_RELEASE)" = "UbuntuXXXX" ]; then\
		$(NVCC) -o $@ $(NVCUFLAGS) --compiler-options="-fsanitize=address" $^ ${NVLDFLAGS};\
	else\
		$(NVCC) -o $@ $(NVCUFLAGS) $^ ${NVLDFLAGS};\
	fi

${DST_DIR}/comm_leak_test_multi: ${DST_DIR}/comm_leak_test_multi.o
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) ${NVLDFLAGS} $^

${DST_DIR}/coll_leak_test_multi: ${DST_DIR}/coll_leak_test_multi.o
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) ${NVLDFLAGS} $^

${DST_DIR}/channel_test: ${DST_DIR}/channel_test.o
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) ${NVLDFLAGS} $^ -ldl
