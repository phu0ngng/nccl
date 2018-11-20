.DEFAULT_GOAL: build
DIRS := apitest single unit perf examples
ifeq ($(MPI),1)
DIRS += mpi coll_net
endif
build: ${DIRS:%=%.build}
clean: ${DIRS:%=%.clean} mpi.clean gtest.clean

%.build:gtest
	${MAKE} -C $* BUILDDIR=${BUILDDIR}

%.clean:
	${MAKE} -C $* clean

##
include gtest.mk
