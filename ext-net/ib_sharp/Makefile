#
# Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#
.PHONY : test build
default: build

BUILDDIR=build
CUDA_HOME ?= /usr/local/cuda
SHARP_HOME ?= /opt/hpcx/v2.4.0_update/sharp
INC:= -I$(CUDA_HOME)/include -I$(SHARP_HOME)/include
CFLAGS:=-fPIC $(INC) -std=c99  -D_GNU_SOURCE -I. -g -O0
PLUGIN_SO:=$(BUILDDIR)/lib/libnccl-net.so
FILES:=plugin.c ibvwrap.c utils.c
OBJS:=$(FILES:%.c=$(BUILDDIR)/obj/%.o)
TEST:=$(BUILDDIR)/bin/test

$(BUILDDIR)/obj/%.o: %.c
	mkdir -p $(BUILDDIR)/obj
	$(CC) $(CFLAGS) -c $< -o $@

$(PLUGIN_SO): $(OBJS)
	mkdir -p $(BUILDDIR)/lib
	$(CC) -shared -o $@ -Wl,-soname,$(PLUGIN_SO) $(OBJS) -libverbs -L$(CUDA_HOME)/lib64 -lcudart -L$(SHARP_HOME)/lib -lsharp_coll

build: $(PLUGIN_SO)
test:  $(TEST)

$(TEST): test.c $(PLUGIN_SO)
	mkdir -p $(BUILDDIR)/bin
	LD_LIBRARY_PATH=$(SHARP_HOME)/lib:$LD_LIBRARY_PATH $(CC) $(CFLAGS) test.c -o $@ -L$(BUILDDIR)/lib -lnccl-net
	LD_LIBRARY_PATH=$(SHARP_HOME)/lib:$LD_LIBRARY_PATH $(TEST)

clean:
	rm -f $(PLUGIN_SO) $(OBJS) $(TEST)
