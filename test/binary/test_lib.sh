#!/bin/bash

# WARNING!!!
#
# Any change to the expected library dependencies needs to be discussed by the NCCL dev team, and
# specifically to be cleared by Sylvain.
#
# Changes to the libraries could cause incompatibility between NCCL versions for existing
# customers.
#
# Do not modify this file to allow additional libraries to be used without first getting
# that approval!

lib=$1
if [ "$lib" == "" ]; then
  lib=`dirname $0`/../../lib/libnccl.so
fi
if [ ! -f $lib ]; then
  echo -e "\e[31m\e[1m [FAILED]\e[0m : Library not found."
  exit 1
fi

errors=""

stripldd() {
  while read line; do
    basename `echo $line | cut -d "=" -f 1`
  done
}

libs="linux-vdso.so libstdc++.so libm.so libgcc_s.so libc.so ld-linux-x86-64.so"
# Glibc 2.34 now integrates libpthread, librt and libdl. Do not check them on recent distros.
glibc_version=""
if command -v objdump >/dev/null 2>&1; then
  glibc_version=$(objdump -T "$lib" 2>/dev/null | grep -oE 'GLIBC_[0-9][0-9.]*' | sed 's/^GLIBC_//' | sort -Vu | tail -1)
fi
if [ -z "$glibc_version" ]; then
  glibc_version=$(ldd --version | head -1 | tr -s ' ' '\n' | tail -1)
fi
glibc_version_major=$(echo "$glibc_version" | cut -d '.' -f 1)
glibc_version_minor=$(echo "$glibc_version" | cut -d '.' -f 2)
if [ "$glibc_version_major" -lt "2" ] || [ "$glibc_version_minor" -lt "34" ]; then
  libs="$libs librt.so libdl.so libpthread.so"
fi

lddlibs=`ldd $lib | stripldd`

for lddlib in $lddlibs; do
  found=0
  otherlibs=""
  for lib in $libs; do
    if [[ "$lddlib" == "$lib"* ]]; then
      found=1
    else
      otherlibs="$otherlibs $lib"
    fi
  done
  if [ $found == 1 ]; then
    echo -ne "\e[32m\e[1m  [OK]\e[0m\t"
  else
    echo -ne "\e[31m\e[1m [FAIL]\e[0m\t"
    errors="$errors $lddlib"
  fi
  echo $lddlib
  libs=$otherlibs
done

echo -n "Test result : "
if [ "$errors" != "" ]; then
  echo -e "\e[31m\e[1m [FAILED]\e[0m : extra libs :"
  for el in $errors; do echo "    $el"; done
  exit 1
fi

if [ "$otherlibs" != "" ]; then
  echo -e "\e[31m\e[1m [FAILED]\e[0m : libs not found :"
  for el in $otherlibs; do echo "    $el"; done
  exit 1
fi

echo -e "\e[32m\e[1m [OK]\e[0m\t"
exit 0
