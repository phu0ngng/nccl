.DEFAULT_GOAL: build
DIRS := apitest single unit perf examples binary stress
ifeq ($(MPI),1)
DIRS += mpi replay
endif
build: ${DIRS:%=%.build}
clean: ${DIRS:%=%.clean} mpi.clean gtest.clean verifiable.clean

%.build:gtest
	${MAKE} -C $* BUILDDIR=${BUILDDIR}

verifiable_self_test:
	${MAKE} -C verifiable BUILDDIR=${BUILDDIR} verifiable_self_test

%.clean:
	${MAKE} -C $* clean

##
include gtest.mk
