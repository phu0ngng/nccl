#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# See LICENCE.txt for license information
#

# Prerequisite: $(FILESTOFORMAT) contains the list of files of interest for formatting
# As this file defines a new target (format), it should be included at least after the definition of the
# default target.

ASTYLE_FORMAT_OPTS=-Qv --style=java --indent-after-parens --indent-modifiers --indent-switches --indent-continuation=2 --keep-one-line-blocks --keep-one-line-statements --indent=spaces=2 --lineend=linux --suffix=none

.PHONY : format
format :
	@astyle $(ASTYLE_FORMAT_OPTS) $(FILESTOFORMAT)
