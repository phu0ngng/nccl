#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# See LICENCE.txt for license information
#

ASTYLE_FORMAT_OPTS=-Qv --style=java --indent-after-parens --indent-modifiers --indent-switches --indent-continuation=2 --keep-one-line-blocks --keep-one-line-statements --indent=spaces=2 --lineend=linux --suffix=none
FORMATFILES := $(shell find src \( -name "*.cu" -o -name "*.h" \) -print | grep -v -E 'ibvwrap.h|nvmlwrap.h')

ifdef NOFORMATCHECK
$(info Skipping code formatting checks on user request.)
else
ifeq "$(findstring $(MAKECMDGOALS),format)" ""
ASTYLE_PRESENT := $(shell which astyle>/dev/null; echo $$?)
ifeq ($(ASTYLE_PRESENT), 1)
$(warning Astyle could not be found in the system. Please install astyle to enable code formatting checks.)
else
CODE_FORMATTED := $(shell astyle --dry-run $(ASTYLE_FORMAT_OPTS) $(FORMATFILES) | grep -q "0 formatted"; echo $$?)
ifneq ($(CODE_FORMATTED), 0)
$(error Code is not formatted properly. Please run make format in the src directory.)
endif # code formatted properly
endif # astyle present in the system
endif # "format" not a goal in this invocation
endif # def NOFORMATCHECK

.PHONY : format
format :
	@astyle $(ASTYLE_FORMAT_OPTS) $(FORMATFILES)
