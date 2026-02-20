#!/bin/bash

if [ "$1" == "" ]; then
  echo "Usage : $0 Title of the feature"
  exit 1
fi

# to change to fit your system
machine=$(uname -a)
MYSED=$(which sed)
MYGREP=$(which grep)
if ${MYGREP} -q "Darwin" <<< "${machine}"; then
  # echo "trying to load Darwin specific sed and grep..."
  MYSED=$(which gsed)
  MYGREP=$(which ggrep)
fi
if [ -z "${MYSED}" ]; then echo "ERROR: GNU compatible sed not found. If on mac, consider using `brew install gnu-sed`"; fi
if [ -z "${MYGREP}" ]; then echo "ERROR: GNU compatible grep not found. If on mac, consider using `brew install grep`"; fi


echo "INFO: using sed: ${MYSED} and grep: ${MYGREP} on machine = ${machine}"

# clean the title: remove the "/"
title=$(echo "$@" | ${MYSED} 's/\// /')
# get the filename from the title
title_name=${title// /_}

# search for the next available item
uuid=$(uuidgen)
item=${uuid:0:8}
folder=items/id_${item}
while [ -d ${folder} ]; do
  uuid=$(uuidgen)
  item=${uuid:0:8}
  folder=items/id_${item}
done

# get the folder name and the filename
output=${folder}/${title_name}.md
output_grep="items\/id_${item}\/${title_name}.md"

# get the version number
re='^[0-9]+$'
vnum="none"
while ! [[ ${vnum} =~ $re ]] ; do
  read -p "Which version is your feature for? version 2."
  vnum=${REPLY}
done
vkey="<!-- V2_${vnum}_DO_NOT_MOVE -->"

# identify if the index already has the version subsection, abort if not
# we ask the user to add the section to preverve the ordering
test_v=$(${MYGREP} "## version 2.${vnum}" index.md)
echo "test_v ${test_v}"
if [ -z "$test_v" ]; then
  echo "ERROR: version 2.${vnum} header NOT found"
  echo "TO DO: place the following code in index.md at the right location and try again"
  echo "<!------------------------------------->"
  echo "## version 2.${vnum}"
  echo " "
  echo "${vkey}"
  echo " "
  echo "<!------------------------------------->"
  exit 1
else
  echo "version 2.${vnum} header found";
fi

# make sure we don't do stupid things
read -p "You are about to create a new documentation file (${output}) for version 2.${vnum}. Do you want to proceed? [Yy/Nn] " -n 1 -r
echo    # (optional) move to a new line
if [[ $REPLY =~ ^[Yy]$ ]]
then
  # create the folder structure
  mkdir -p ${folder}
  # get the new title as the header of the file
  echo "# ${title}" > ${output}
  cat $(pwd)/templates/template.md >> ${output}
  # create dummy file so that git can add it.
  # you can delete it when you have real images stored in the images dir
  mkdir -p ${folder}/images
  touch ${folder}/images/dummy

  # reference the file in the main index, find the right location and add the feature at the bottom of the list
  root=$(pwd)/"index.md"
  ${MYSED} -i "s/${vkey}/- \[${title}\]\(${output_grep}\)\n${vkey}/" ${root}

  # it's done!
  echo "Done! You have created new feature documentation for ${title}"
  echo "Now go to ${output} to start editing"
else
  echo "Aborted"
fi
