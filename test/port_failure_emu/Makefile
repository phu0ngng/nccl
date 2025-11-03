include ../../makefiles/common.mk

.PHONY: all clean

BUILDDIR := $(abspath ../../build)
DST_DIR := $(BUILDDIR)/test/port_failure_emu

# Port Failure Emulator variables
PFE_SRCDIR := src
PFE_INCDIR := include
PFE_CLI_DIR := cli
PFE_EXAMPLES_DIR := examples
PFE_BUILDDIR := $(DST_DIR)
PFE_LIBSO := $(PFE_BUILDDIR)/libportfailureemu.so
PFE_CLI_EXE := $(PFE_BUILDDIR)/portFailureEmu
PFE_EXAMPLE_EXE := $(PFE_BUILDDIR)/portFailureEmuExample

# Source files
PFE_LIB_SRCS := $(wildcard $(PFE_SRCDIR)/*.c)
PFE_LIB_OBJS := $(patsubst $(PFE_SRCDIR)/%.c, $(PFE_BUILDDIR)/%.o, $(PFE_LIB_SRCS))

# Compiler flags
PFE_CFLAGS := -I$(PFE_INCDIR) -fPIC -Wall -Wextra -ggdb
PFE_LDFLAGS := -lmlx5 -libverbs

# Check for RDMA libraries
HAVE_MLX5 := $(shell ldconfig -p 2>/dev/null | grep -q 'libmlx5\.so' && echo 1 || echo 0)
HAVE_VERBS := $(shell ldconfig -p 2>/dev/null | grep -q 'libibverbs\.so' && echo 1 || echo 0)

ifeq ($(HAVE_MLX5),0)
$(info RDMA libraries not found (libmlx5 not available))
$(info Skipping port-failure-emu build (requires RDMA hardware support))
all:
	@true
else ifeq ($(HAVE_VERBS),0)
$(info RDMA libraries not found (libibverbs not available))
$(info Skipping port-failure-emu build (requires RDMA hardware support))
all:
	@true
else
$(info Building port-failure-emu with RDMA support)

all: $(PFE_LIBSO) $(PFE_CLI_EXE) $(PFE_EXAMPLE_EXE)

clean:
	rm -rf $(DST_DIR)

# Library build
$(PFE_BUILDDIR)/%.o: $(PFE_SRCDIR)/%.c
	@printf "Compiling %s\n" $@
	@mkdir -p $(PFE_BUILDDIR)
	$(CC) $(PFE_CFLAGS) -c $< -o $@

$(PFE_LIBSO): $(PFE_LIB_OBJS)
	@printf "Creating %s\n" $@
	@mkdir -p $(PFE_BUILDDIR)
	$(CC) -shared -o $@.0 $^ $(PFE_LDFLAGS) -Wl,-soname,$(notdir $@).0
	ln -sf $(notdir $@).0 $@

# CLI tool build
$(PFE_CLI_EXE): $(PFE_CLI_DIR)/main.c $(PFE_LIBSO)
	@printf "Linking %s\n" $@
	@mkdir -p $(PFE_BUILDDIR)
	$(CC) $(PFE_CFLAGS) -o $@ $< -L$(PFE_BUILDDIR) -lportfailureemu -Wl,-rpath='$$ORIGIN' $(PFE_LDFLAGS)

# Example build
$(PFE_EXAMPLE_EXE): $(PFE_EXAMPLES_DIR)/port_failure_emu_example.c $(PFE_LIBSO)
	@printf "Linking %s\n" $@
	@mkdir -p $(PFE_BUILDDIR)
	$(CC) $(PFE_CFLAGS) -o $@ $< -L$(PFE_BUILDDIR) -lportfailureemu -Wl,-rpath='$$ORIGIN' $(PFE_LDFLAGS)

endif
