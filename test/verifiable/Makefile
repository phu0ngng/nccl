include ../../makefiles/common.mk

.PHONY: all clean

BUILDDIR := $(abspath ../../build)
NCCLDIR := $(BUILDDIR)
NVCUFLAGS += -I$(NCCLDIR)/include/ -I../include
DST_DIR := $(BUILDDIR)/test/verifiable

all: verifiable_self_test

clean:
	rm -rf $(DST_DIR)

verifiable_self_test: $(DST_DIR)/verifiable_self_test

TEST_VERIFIABLE_SRCDIR := .
TEST_VERIFIABLE_BUILDDIR := $(DST_DIR)
include verifiable.mk

$(TEST_VERIFIABLE_BUILDDIR)/verifiable.o: $(TEST_VERIFIABLE_SRCDIR)/verifiable.cu $(TEST_VERIFIABLE_HDRS)
	@printf "Compiling %s\n" $@
	@mkdir -p $(TEST_VERIFIABLE_BUILDDIR)
	$(NVCC) -Xcompiler -fPIC -o $@ $(NVCUFLAGS) -c $(TEST_VERIFIABLE_SRCDIR)/verifiable.cu

$(TEST_VERIFIABLE_LIBSO): $(TEST_VERIFIABLE_BUILDDIR)/verifiable.o
	@printf "Creating %s\n" $@
	@mkdir -p $(TEST_VERIFIABLE_BUILDDIR)
	$(CXX) -shared -o $@.0 $^ -Wl,-soname,$(notdir $@).0
	ln -sf $(notdir $@).0 $@

$(DST_DIR)/verifiable_self_test: self_test.cu $(TEST_VERIFIABLE_LIBSO)
	@printf "Linking %s\n" $@
	@mkdir -p $(DST_DIR)
	$(NVCC) -o $@ $(NVCUFLAGS) $< $(NVLDFLAGS) -L$(TEST_VERIFIABLE_BUILDDIR) -lverifiable -Xlinker -rpath='$$ORIGIN'
