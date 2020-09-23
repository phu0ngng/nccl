.DEFAULT_GOAL: build
DIRS := apitest single unit perf examples binary stress
ifeq ($(MPI),1)
DIRS += mpi
endif
build: ${DIRS:%=%.build}
clean: ${DIRS:%=%.clean} mpi.clean gtest.clean

%.build:gtest
	${MAKE} -C $* BUILDDIR=${BUILDDIR}

%.clean:
	${MAKE} -C $* clean

##
include gtest.mk
