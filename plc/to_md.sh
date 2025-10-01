#!/bin/bash

#-------------------------------------
function to_md_pandoc(){
  echo "<details>" >> ${output}
  echo "<summary><h2>$1</h2></summary>" >> ${output}
  echo " " >> ${output}
  cat $2 | sed 's/<title>.*//' | sed 's/<center><h1>.*//' | sed 's/\$CONTENTS//' | pandoc -f html -t gfm --shift-heading-level-by=1 >> ${output}
  if ! [ -z $3 ]; then
    cat $3 | sed 's/^/- nccl@/' >> ${output}
  fi
  echo "</details>" >> ${output}
  echo " " >> ${output}
}
#-------------------------------------
root=$(pwd)/"index.md"
echo "# NCCL - PLC" > ${root}
echo " " >> ${root}
#-------------------------------------

item=1
# for each item folder
while [ -d items/$item ]; do
  #-------------------------------------
  folder=items/${item}
  # get the title and remove the '/' if any to avoid issues with folder names
  title=$(ggrep -o -P '(?<=\: ).*(?=</title>)' ${folder}/index.html | sed 's/\// /')
  title_name=${title// /_}
  output=${folder}/${title_name}.md

  echo "new markdown file: ${output} for feature: ${title}"
  # generate the main title
  echo "# ${title}" > ${output}
  # get the requirements, design, coding, and testing
  to_md_pandoc "Requirements" ${folder}/req.html
  to_md_pandoc "Design" ${folder}/design.html
  to_md_pandoc "Coding" ${folder}/coding.html ${folder}/coding.list
  to_md_pandoc "Testing" ${folder}/testing.html

  echo "- [${title}](${output})" >> ${root}

  #-------------------------------------
  item=`expr $item + 1`;
done

