#!/bin/bash

lib=$1
if [ "$lib" == "" ]; then
  lib=`dirname $0`/../../lib/libnccl.so
fi

ncclsymbols=`nm --dynamic --defined-only $lib | cut -c 20- | grep "^nccl"`
pncclsymbols=`nm --dynamic --defined-only $lib | cut -c 20- | grep "^pnccl"`
othersymbols=`nm --dynamic --defined-only $lib | cut -c 20- | grep -v "^nccl" | grep -v "^pnccl"`
errors=""

sym_ok() {
  s=$1
  othersymbols="_init _fini __bss_start _end _edata"
  for sym in $othersymbols; do
    if [ "$s" == "$sym" ]; then
      return
    fi
  done
  echo $symbol
}

while [ "$ncclsymbols" != "" ]; do
  set -- $ncclsymbols
  ncclsym=$1
  shift
  ncclsymbols=$@
  set -- $pncclsymbols
  found=0
  for pncclsym in $pncclsymbols; do
    if [ "`echo $pncclsym | cut -c 2-`" == "$ncclsym" ]; then
      found=1
      break
    fi
  done
  if [ "$found" == "1" ]; then
    echo -e "\e[32m\e[1m  [OK]\e[0m\t $ncclsym p$ncclsym"
  else
    echo -e "\e[31m\e[1m [FAIL]\e[0m\t $ncclsym"
  fi
done

for symbol in $othersymbols; do
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
