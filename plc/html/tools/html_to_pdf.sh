#!/bin/bash

if [ "$3" == "" ]; then
  echo "Usage : $0 <chrome-binary> <html> <pdf>"
  exit 1
fi

chrome=$1
html=$2
pdf=$3

$chrome --headless --print-to-pdf="$pdf" file://$html 2>&1 \
 | grep -v "MessageAttachmentSet destroyed with unconsumed attachments" \
 | grep -v "Written to file"

if [ -f $pdf ]; then
  exit 0
else
  exit 1
fi
