#!/bin/bash

usage() {
  echo "Usage: $0 [version] [last version]"
  exit 1
}

findversion() {
  target=$1
  branch=$2
  v=`git log $target/$branch --format=%s -n 1 | head -1 | sed -e 's/NCCL //'`
  git log $target/$branch --format=%s | while read commit; do
    v=`echo $commit | sed -e 's/NCCL //'`
    if [ "`echo $v | cut -c 1`" == "2" ]; then
      if git log v$v >/dev/null; then
        echo $v
        return
      fi
    fi
  done
}

version=$1
lastversion=$2

# check setup. We should have the following remote targets:
# origin -> gitlab
# github -> github.
set -- \
 "github" "https://github.com/NVIDIA/nccl.git" \
 "origin" "ssh://git@gitlab-master.nvidia.com:12051/nccl/nccl.git"
echo "#### Checking remotes"
while [ "$1" != "" ]; do
  if [ `git remote get-url $1` != "$2" ]; then
    echo "Error: need the '$1' target pointing to '$2'";
    exit 1
  fi
  shift 2
done
echo "#### Remotes OK"
echo

echo "#### Repo update"
git remote update
echo "#### Repo update done"
echo

# Guess origin version if not provided
if [ "$version" == "" ]; then
  echo "#### Finding lastest stable version ..."
  version=`findversion origin stable`
  if [ "$version" == "" ]; then
    echo "#### Could not find version"
    usage
  fi
  echo "#### Found $version"
  echo
fi

# Guess github version
if [ "$lastversion" == "" ]; then
  echo "#### Finding github version ..."
  lastversion=`findversion github master`
  if [ "$lastversion" == "" ]; then
    echo "#### Could not find last version"
    usage
  fi
  echo "#### Found $lastversion"
  echo
fi

echo "#### Ready to generate diff"
echo -n "Generating github diff between origin/stable (v$version) and github/master (v$lastversion). Is that correct (y/n)?"
read answer
if [ "$answer" != "y" ]; then
  exit 1
fi
echo

echo "#### Checking for modifications in manually-merged directories..."
manualmodlist=`cat github/sync-modif.list`
git diff -r v$lastversion -r v$version $manualmodlist > github-manual-$version.diff
if [ "`cat github-manual-$version.diff`" != "" ]; then
  echo "#### MODIFICATIONS FOUND"
  echo "#### WARNING: please check github-manual-$version.diff, those modifications may need a manual merge."
else
  echo "#### No modification, good."
fi
echo

echo "#### Generating diff..."
git diff -r github/master -r v$version `cat github/sync.list` > github-$version.diff
echo "#### Written github-$version.diff"
echo
