#!/bin/bash

lib=$1
if [ "$lib" == "" ]; then
  lib=../../build/lib/libnccl.so
fi

errors=""

stripldd() {
  while read line; do
    basename `echo $line | cut -d "=" -f 1`
  done
}

libs="linux-vdso.so librt.so libstdc++.so libm.so libgcc_s.so libc.so ld-linux-x86-64.so libpthread.so"
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
