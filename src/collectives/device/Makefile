include ../../../makefiles/common.mk
include ../../../makefiles/version.mk

BUILDDIR ?= $(abspath ../../../build)
OBJDIR := $(BUILDDIR)/obj/collectives/device

LIBSRCFILES := all_reduce.cu broadcast.cu reduce.cu all_gather.cu reduce_scatter.cu

LIBOBJ     := $(patsubst %.cu,$(OBJDIR)/%_sum.o, $(LIBSRCFILES)) \
              $(patsubst %.cu,$(OBJDIR)/%_prod.o, $(LIBSRCFILES)) \
              $(patsubst %.cu,$(OBJDIR)/%_min.o, $(LIBSRCFILES)) \
              $(patsubst %.cu,$(OBJDIR)/%_max.o, $(LIBSRCFILES)) \
              $(OBJDIR)/multiop.o

LIBSRCFILES += multiop.cu

DEPFILES   := $(patsubst %.cu, $(OBJDIR)/%.d, $(LIBSRCFILES))
STATICLIB  := $(OBJDIR)/colldevice.a
DEVOBJ     := $(OBJDIR)/devlink.o

NVCUFLAGS  += -I. -I.. -I../.. -I../../include --compiler-options "-fPIC -fvisibility=hidden"


all: $(STATICLIB)

-include $(DEPFILES)

$(STATICLIB): $(LIBOBJ) $(DEVOBJ)
	@printf "Archiving  %-35s > %s\n" objects $@
	ar cr $@ $^

$(OBJDIR)/%.d : %.cu
	@mkdir -p $(OBJDIR)
	@$(NVCC) $(NVCUFLAGS) -M $< -o $@.tmp
	@sed "0,/^.*:/s//$(subst /,\/,$@):/" $@.tmp > $@
	@sed -e 's/.*://' -e 's/\\$$//' < $@.tmp | fmt -1 | \
                sed -e 's/^ *//' -e 's/$$/:/' >> $@
	@rm -f $@.tmp

# Compiled kernels and collectives with reloctable device code ...
$(OBJDIR)/multiop.o : multiop.cu $(OBJDIR)/multiop.d
	@printf "Compiling  %-35s > %s\n" $< $@
	mkdir -p `dirname $@`
	$(NVCC) $(NVCUFLAGS) -dc $< -o $@

$(OBJDIR)/%_sum.o : %.cu $(OBJDIR)/%.d
	@printf "Compiling  %-35s > %s\n" $< $@
	mkdir -p `dirname $@`
	$(NVCC) -DNCCL_OP=0 $(NVCUFLAGS) -dc $< -o $@

$(OBJDIR)/%_prod.o : %.cu $(OBJDIR)/%.d
	@printf "Compiling  %-35s > %s\n" $< $@
	mkdir -p `dirname $@`
	$(NVCC) -DNCCL_OP=1 $(NVCUFLAGS) -dc $< -o $@

$(OBJDIR)/%_min.o : %.cu $(OBJDIR)/%.d
	@printf "Compiling  %-35s > %s\n" $< $@
	mkdir -p `dirname $@`
	$(NVCC) -DNCCL_OP=2 $(NVCUFLAGS) -dc $< -o $@

$(OBJDIR)/%_max.o : %.cu $(OBJDIR)/%.d
	@printf "Compiling  %-35s > %s\n" $< $@
	mkdir -p `dirname $@`
	$(NVCC) -DNCCL_OP=3 $(NVCUFLAGS) -dc $< -o $@

# ... and create the device-side linked object with all those.
$(DEVOBJ) : $(LIBOBJ)
	$(NVCC) $(NVCUFLAGS) -dlink $^ -o $@

clean:
	rm -f $(LIBOBJ) $(STATICLIB) test
