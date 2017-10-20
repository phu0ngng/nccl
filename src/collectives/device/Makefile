include ../../../makefiles/common.mk
include ../../../makefiles/version.mk

BUILDDIR ?= $(abspath ../../../build)
OBJDIR := $(BUILDDIR)/obj/collectives/device

LIBSRCFILES := all_reduce.cu broadcast.cu reduce.cu all_gather.cu reduce_scatter.cu

LIBOBJ     := $(patsubst %.cu,$(OBJDIR)/%_sum.o, $(LIBSRCFILES)) \
              $(patsubst %.cu,$(OBJDIR)/%_prod.o, $(LIBSRCFILES)) \
              $(patsubst %.cu,$(OBJDIR)/%_min.o, $(LIBSRCFILES)) \
              $(patsubst %.cu,$(OBJDIR)/%_max.o, $(LIBSRCFILES)) \
              $(OBJDIR)/kernels.o
STATICLIB  := $(OBJDIR)/colldevice.a
DEVOBJ     := $(OBJDIR)/devlink.o

NVCUFLAGS  += -I. -I.. -I../../include --compiler-options "-fPIC -fvisibility=hidden"

all: $(STATICLIB)

$(STATICLIB): $(LIBOBJ) $(DEVOBJ)
	@printf "Archiving  %-35s > %s\n" objects $@
	ar cr $@ $^

# Compiled kernels and collectives with reloctable device code ...
$(OBJDIR)/kernels.o : kernels.cu
	@printf "Compiling  %-35s > %s\n" $< $@
	mkdir -p `dirname $@`
	$(NVCC) $(NVCUFLAGS) -dc $< -o $@

$(OBJDIR)/%_sum.o : %.cu ../%.h ../ll_kernel.h
	@printf "Compiling  %-35s > %s\n" $< $@
	mkdir -p `dirname $@`
	$(NVCC) -DNCCL_OP=0 $(NVCUFLAGS) -dc $< -o $@

$(OBJDIR)/%_prod.o : %.cu ../%.h ../ll_kernel.h
	@printf "Compiling  %-35s > %s\n" $< $@
	mkdir -p `dirname $@`
	$(NVCC) -DNCCL_OP=1 $(NVCUFLAGS) -dc $< -o $@

$(OBJDIR)/%_min.o : %.cu ../%.h ../ll_kernel.h
	@printf "Compiling  %-35s > %s\n" $< $@
	mkdir -p `dirname $@`
	$(NVCC) -DNCCL_OP=2 $(NVCUFLAGS) -dc $< -o $@

$(OBJDIR)/%_max.o : %.cu ../%.h ../ll_kernel.h
	@printf "Compiling  %-35s > %s\n" $< $@
	mkdir -p `dirname $@`
	$(NVCC) -DNCCL_OP=3 $(NVCUFLAGS) -dc $< -o $@

# ... and create the device-side linked object with all those.
$(DEVOBJ) : $(LIBOBJ)
	$(NVCC) $(NVCUFLAGS) -dlink $^ -o $@

clean:
	rm -f $(LIBOBJ) $(STATICLIB) test
