#
# Copyright (c) 2015-2023, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#

PLCITEMS := $(shell find ./ -maxdepth 1 -type d -regex "./[0-9]+")

all: item

item: $(PLCITEMS)
	@for dir in $(PLCITEMS); do \
		$(MAKE) -C $${dir}; 			\
	done
