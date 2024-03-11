# We require both of the following paths to be set upon including this makefile
# TEST_VERIFIABLE_SRCDIR = <points to this directory>
# TEST_VERIFIABLE_BUILDDIR = <points to destination directory for libverifiable.so>

TEST_VERIFIABLE_HDRS = $(TEST_VERIFIABLE_SRCDIR)/verifiable.h
TEST_VERIFIABLE_LIBSO = $(TEST_VERIFIABLE_BUILDDIR)/libverifiable.so
