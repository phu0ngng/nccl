include ../../makefiles/common.mk

.PHONY: all clean

BUILDDIR := $(abspath ../../build)
DST_DIR := $(BUILDDIR)/test/os

# Select platform-specific source file
ifeq ($(OS),Windows_NT)
    OS_SRC := windows.cc
else
    OS_SRC := linux.cc
endif

OBJ_FILES := ${DST_DIR}/$(OS_SRC:%.cc=%.o)

all build: ${DST_DIR}/libnccl_test_os.a

clean:
	rm -rf ${DST_DIR}

${DST_DIR}/%.o: %.cc os.h
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(CXX) $(CXXFLAGS) -I$(BUILDDIR)/include -fPIC -o $@ -c $<

${DST_DIR}/libnccl_test_os.a: $(OBJ_FILES)
	@printf "Creating archive %-35s > %s\n" $^ $@
	@mkdir -p ${DST_DIR}
	ar rcs $@ $^
