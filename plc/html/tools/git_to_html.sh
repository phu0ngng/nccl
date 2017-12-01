#!/bin/bash

write_revisions() {
  first=1
  echo "<table>"
  echo "<tr><th>Commit</th><th>Author</th><th>Date</th><th>Description</th></tr>"
  while read line; do
    set -- $line
    if [ "$1" == "commit" ]; then
      if [ "$first" != "1" ]; then
        echo "</td></tr>"
      fi
      echo "<tr><td>$2</td>"
    elif [ "$1" == "Author:" ]; then
      shift
      echo "<td>"
      # Replace the email address < and > by &lt; and &gt;
      echo "$@" | sed s/'<'/'\&lt;'/g | sed s/'>'/'\&gt;'/g
      echo"</td>"
    elif [ "$1" == "Date:" ]; then
      shift
      echo "<td>$@</td><td>"
      read line # Empty line
    else
      echo $line
    fi
  done
  echo "</td></tr>"
  echo "</table>"
}

echo "<h2>Document history</h2>"

git log --abbrev-commit --date=short $1 | write_revisions

echo "</body></html>"
