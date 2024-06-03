.DEFAULT_GOAL: build
DIRS := apitest single unit perf examples binary stress verifiable
ifeq ($(MPI),1)
DIRS += mpi replay ras
endif
build: ${DIRS:%=%.build}
clean: ${DIRS:%=%.clean} mpi.clean gtest.clean

%.build:gtest
	${MAKE} -C $* BUILDDIR=${BUILDDIR}

perf.build replay.build:verifiable.build

%.clean:
	${MAKE} -C $* clean

##
include gtest.mk
