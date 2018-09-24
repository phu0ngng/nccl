#!/bin/bash

lib=$1
if [ "$lib" == "" ]; then
  lib=../../build/lib/libnccl.so
fi

exportedsymbols=`nm --dynamic --defined-only $lib | cut -c 20-`
errors=""

sym_ok() {
s=$1
  othersymbols="_init _fini __bss_start _end _edata"
  for sym in $othersymbols; do
    if [ "$s" == "$sym" ]; then
      return
    fi
  done
  echo $symbol | grep -v "^nccl"
}


for symbol in $exportedsymbols; do
  sym=`sym_ok $symbol`
  if [ "$sym" == "" ]; then
    echo -e "\e[32m\e[1m  [OK]\e[0m\t $symbol"
  else
    echo -e "\e[31m\e[1m [FAIL]\e[0m\t $symbol"
    errors="$errors $symbol"
  fi
done

if [ "$errors" != "" ]; then
  echo -e "Test result : \e[31m\e[1m [FAILED]\e[0m"
  exit 1
else
  echo -e "Test result : \e[32m\e[1m [OK]\e[0m\t"
  exit 0
fi
