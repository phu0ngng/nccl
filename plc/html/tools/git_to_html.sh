#!/bin/bash

write_revisions() {
  first=1
  echo "<table>"
  echo "<tr><th width="70">Commit</th><th width="170">Author</th><th width="90">Date</th><th>Description</th></tr>"
  while read line; do
    set -- $line
    if [ "$1" == "commit" ]; then
      if [ "$first" != "1" ]; then
        echo "</pre></td></tr>"
      fi
      echo "<tr><td>$2</td>"
      first=0
    elif [ "$1" == "Author:" ]; then
      shift
      echo "<td>"
      # Replace the email address < and > by &lt; and &gt;
      echo "$@" | sed s/'<'/'\&lt;'/g | sed s/'>'/'\&gt;'/g
      echo "</td>"
    elif [ "$1" == "Date:" ]; then
      shift
      echo "<td>$@</td><td><pre>"
      read line # Empty line
    else
      echo $line
    fi
  done
  echo "</td></tr>"
  echo "</table>"
}

echo "<h2>List of commits</h2>"

for commit in `cat $1`; do
 git show --abbrev-commit --date=short $commit -s
done | write_revisions

echo "</body></html>"
