include ../../makefiles/common.mk

.PHONY: all clean

BUILDDIR := $(abspath ../../build)
NCCLDIR := $(BUILDDIR)

DST_DIR := $(BUILDDIR)/test/binary
SRC_FILES := test_lib.sh test_symbols.sh
SH_FILES := $(SRC_FILES:%=$(DST_DIR)/%)

all: ${SH_FILES}

clean:
	rm -rf ${DST_DIR}

${DST_DIR}/%.sh: %.sh
	@printf "Grabbing  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	install -m 755 $< $@
