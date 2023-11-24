#!/bin/bash

if [ "$1" == "" ]; then
  echo "Usage : $0 Title of the feature"
  exit 1
fi

title="$@"

item=1
while [ -d items/$item ]; do item=`expr $item + 1`; done

mkdir -p items/$item/images
# create dummy file so that git can add it.
# you can delete it when you have real images stored in the images dir
touch items/$item/images/dummy

for template in templates/*; do
  sed -e "s/\${plc:item}/$item/g" \
      -e "s/\${plc:title}/$title/g" \
  $template > items/$item/`basename $template`
done

echo "Created item $item ($PWD/items/$item)"
