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
SRCINC = -I$(BUILDDIR)/include/ -I../include -I../../src/include -I ../../src/device/ -I../../src/graph -I../../src/include/plugin -I../../src/include/os
# Use static library since all the internal symbols are there
DLIB = ${BUILDDIR}/lib/libnccl.so
DLIBFLAGS = $(NVLDFLAGS) -L$(BUILDDIR)/lib -lnccl -lcudart
LIBRARIES += nccl_static
STATICLIB = ${BUILDDIR}/lib/libnccl_static.a
NVLDFLAGS += -L$(BUILDDIR)/lib $(LIBRARIES:%=-l%)
##
DST_DIR := $(abspath ${BUILDDIR}/test/unit)
CU_FILES := align_test.cu  #kernel_test.cu
#CC_FILES += graph_test.cc model.cc data.cc xml_test.cc xml_fuse_test.cc comm_leak_test.cc channel_test.cc enqueue_test.cc log_algo.cc ft_test.cc overlap_test.cc plugin_load.cc device_id_test.cc
CC_FILES += graph_test.cc model.cc data.cc xml_test.cc xml_fuse_test.cc comm_leak_test.cc enqueue_test.cc log_algo.cc ft_test.cc overlap_test.cc plugin_load.cc device_id_test.cc intrusive_map_test.cc
KCU_FILES := align_test.cu #kernel_test.cu
CU_FILES += devapi_window_stress.cu
ifeq ($(MPI),1)
#CU_FILES += ncclNet_test.cu ncclNet_read_write_stress_test.cu

CU_FILES += comm_leak_test_multi.cu coll_leak_test_multi.cu ft_abort_rank0.cu put_signal_ping_pong_gin.cu devapi_uts.cu devapi_signal_ring.cu devapi_barrier.cu devapi_data_ring.cu devapi_data_ring2.cu gin_data_ring.cu put_gin_alltoall.cu lsa_pointer_test.cu lsa_multimem_pointer_test.cu
endif
CU_FILES += register_memcpyTest.cu
OBJ_FILES:= $(CU_FILES:%.cu=${DST_DIR}/%.o) $(CC_FILES:%.cc=${DST_DIR}/%.o)
EXE_FILES:= $(CU_FILES:%.cu=${DST_DIR}/%) $(CC_FILES:%.cc=${DST_DIR}/%)
KEXE_FILES:= $(KCU_FILES:%.cu=${DST_DIR}/%)
DEP_FILES:= $(OBJ_FILES:%.o=%.d)

LSB_RELEASE=$(shell lsb_release -is)
#
all:${EXE_FILES} net.build tuner.build profiler.build env.build
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
net.%:
	${MAKE} -C $(BUILDDIR)/../ext-net/example $* BUILDDIR=$(BUILDDIR)/test/unit/plugins

tuner.%:
	${MAKE} -C $(BUILDDIR)/../ext-tuner/example $* BUILDDIR=$(BUILDDIR)/test/unit/plugins

profiler.%:
	${MAKE} -C $(BUILDDIR)/../ext-profiler/example $* BUILDDIR=$(BUILDDIR)/test/unit/plugins

env.%:
	${MAKE} -C $(BUILDDIR)/../ext-env/example $* BUILDDIR=$(BUILDDIR)/test/unit/plugins

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

${DST_DIR}/register_memcpyTest: ${DST_DIR}/register_memcpyTest.o
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) $^ ${NVLDFLAGS} ${GTEST_LIBS}

#
#${DST_DIR}/nccl_mpi.o: ../../share/nccl_mpi.c
#	@printf "Compiling   %-35s > %s\n" $< $@
#	@mkdir -p ${DST_DIR}
#	mpicc -std=c99 $(CXXFLAGS) -I$(BUILDDIR)/include/ -o $@ -c $<
#
${DST_DIR}/ft_abort_rank0: ${DST_DIR}/ft_abort_rank0.o ${DLIB}
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) $^ ${DLIBFLAGS}

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
${DST_DIR}/data: ${DST_DIR}/data.o ${STATICLIB}
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) $^ ${NVLDFLAGS}
#
${DST_DIR}/xml_test: ${DST_DIR}/xml_test.o ${STATICLIB}
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) $^ ${NVLDFLAGS}
#
${DST_DIR}/xml_fuse_test: ${DST_DIR}/xml_fuse_test.o ${STATICLIB}
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
#
${DST_DIR}/log_algo: ${DST_DIR}/log_algo.o
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

#${DST_DIR}/channel_test: ${DST_DIR}/channel_test.o
#	@printf "Linking  %-35s > %s\n" $< $@
#	@mkdir -p ${DST_DIR}
#	$(NVCC) -o $@ $(NVCUFLAGS) ${NVLDFLAGS} $^ -ldl

${DST_DIR}/enqueue_test: ${DST_DIR}/enqueue_test.o
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) ${NVLDFLAGS} $^ -lcudart

${DST_DIR}/overlap_test: ${DST_DIR}/overlap_test.o
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) ${NVLDFLAGS} $^ -lcudart

${DST_DIR}/plugin_load: ${DST_DIR}/plugin_load.o
	@printf "Linking %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) ${NVLDFLAGS} $^ -ldl

${DST_DIR}/device_id_test: ${DST_DIR}/device_id_test.o ${DLIB}
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) $^ ${DLIBFLAGS}

${DST_DIR}/intrusive_map_test: ${DST_DIR}/intrusive_map_test.o ${STATICLIB}
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) $^ ${NVLDFLAGS}

${DST_DIR}/lsa_pointer_test: ${DST_DIR}/lsa_pointer_test.o ${DLIB}
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) $^ ${DLIBFLAGS}

${DST_DIR}/lsa_multimem_pointer_test: ${DST_DIR}/lsa_multimem_pointer_test.o ${DLIB}
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) $^ ${DLIBFLAGS}

${DST_DIR}/devapi_uts: ${DST_DIR}/devapi_uts.o ${DLIB}
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) $^ ${DLIBFLAGS}

${DST_DIR}/devapi_signal_ring: ${DST_DIR}/devapi_signal_ring.o ${DLIB}
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) $^ ${DLIBFLAGS}

${DST_DIR}/devapi_barrier: ${DST_DIR}/devapi_barrier.o ${DLIB}
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) $^ ${DLIBFLAGS}

${DST_DIR}/devapi_data_ring: ${DST_DIR}/devapi_data_ring.o ${DLIB}
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) $^ ${DLIBFLAGS}

${DST_DIR}/devapi_data_ring2: ${DST_DIR}/devapi_data_ring2.o ${DLIB}
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) $^ ${DLIBFLAGS}

${DST_DIR}/gin_data_ring: ${DST_DIR}/gin_data_ring.o ${DLIB}
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) $^ ${DLIBFLAGS}

${DST_DIR}/put_signal_ping_pong_gin: ${DST_DIR}/put_signal_ping_pong_gin.o ${DLIB}
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVLDFLAGS) $^ ${DLIBFLAGS}

${DST_DIR}/put_gin_alltoall: ${DST_DIR}/put_gin_alltoall.o ${DLIB}
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) $^ ${DLIBFLAGS}

${DST_DIR}/devapi_window_stress: ${DST_DIR}/devapi_window_stress.o ${DLIB}
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) $^ ${DLIBFLAGS}
