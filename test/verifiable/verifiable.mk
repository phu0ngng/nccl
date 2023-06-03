# We require both of the following paths to be set upon including this makefile
# TEST_VERIFIABLE_SRCDIR = <points to this directory>
# TEST_VERIFIABLE_BUILDDIR = <points to destination directory for libverifiable.so>

TEST_VERIFIABLE_HDRS = $(TEST_VERIFIABLE_SRCDIR)/verifiable.h
TEST_VERIFIABLE_LIBSO = $(TEST_VERIFIABLE_BUILDDIR)/libverifiable.so

$(TEST_VERIFIABLE_BUILDDIR)/verifiable.o: $(TEST_VERIFIABLE_SRCDIR)/verifiable.cu $(TEST_VERIFIABLE_HDRS)
	@printf "Compiling %s\n" $@
	@mkdir -p $(TEST_VERIFIABLE_BUILDDIR)
	$(NVCC) -Xcompiler -fPIC -o $@ $(NVCUFLAGS) -c $(TEST_VERIFIABLE_SRCDIR)/verifiable.cu

$(TEST_VERIFIABLE_LIBSO): $(TEST_VERIFIABLE_BUILDDIR)/verifiable.o
	@printf "Creating %s\n" $@
	@mkdir -p $(TEST_VERIFIABLE_BUILDDIR)
	$(CXX) -shared -o $@.0 $^ -Wl,-soname,$(notdir $@).0
	ln -sf $(notdir $@).0 $@
