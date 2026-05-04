#!/bin/bash

# WARNING!!!
#
# Any change to the symbols which are exported needs to be discussed by the NCCL dev team, and
# specifically to be cleared by Sylvain.
#
# Changes to the exported symbols could cause incompatibility between NCCL versions for existing
# customers.
#
# Do not modify this file to allow additional symbols to be exported without first getting
# that approval!

lib=$1
if [ "$lib" == "" ]; then
  lib=`dirname $0`/../../lib/libnccl.so
fi

# check if lib exists
if [ ! -f $lib ]; then
  echo -e "\e[31m\e[1m [FAILED]\e[0m : Library not found.\e[0m"
  exit 1
fi

# For ncclsymbols and pncclsymbols, use --demangle option to avoid C++ symbol messing up the $othersymbols list
ncclsymbols=`nm --demangle --dynamic --defined-only $lib | cut -c 20- | grep "^nccl"`
pncclsymbols=`nm --demangle --dynamic --defined-only $lib | cut -c 20- | grep "^pnccl"`
othersymbols=`nm --demangle --dynamic --defined-only $lib | cut -c 20- | grep -v "^nccl" | grep -v "^pnccl"`
# ncclparamsymbols are symbols from NCCL_PARAM framework
ncclparamsymbols=`nm --demangle --dynamic --defined-only $lib | cut -c 20- | grep "^ncclParam"`
no_pncclsymbol="ncclResetDebugInitInternal"
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
  for no_p in $no_pncclsymbol; do
    if [ "$no_p" == "$ncclsym" ]; then
       found=1
       break
    fi
  done
  if [ "$found" == "1" ]; then
    echo -e "\e[32m\e[1m  [OK]\e[0m\t $ncclsym [profiling symbol waived]"
    continue;
  fi

  found=0
  for param in $ncclparamsymbols; do
    if [ "$param" = "$ncclsym" ]; then
      found=1
      break
    fi
  done
  if [ "$found" == "1" ]; then
    # echo "skip param symbol $ncclsym"
    continue
  fi

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
    errors="$errors $ncclsym"
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
