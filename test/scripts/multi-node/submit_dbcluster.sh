#!/bin/bash

set -e
#set -x

# Change directory to the location of the script
cd "$( dirname "${BASH_SOURCE[0]}" )"

POSITIONAL=()
while [[ $# -gt 0 ]]
do
    key="$1"

    case $key in
	-j|--jobdir)
	    JOBDIR="$2"
	    shift # past argument
	    shift # past value
	    ;;
	-N|--nodes)
	    NODES="$2"
	    shift # past argument
	    shift # past value
	    ;;
	-npernode|--npernode|-nppn|-ppn|-n)
	    PPN="$2"
	    shift # past argument
	    shift # past value
	    ;;
	*)    # unknown option
	    POSITIONAL+=("$1") # save it in an array for later
	    shift # past argument
	    ;;
    esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

NOW="$(date +%F-%H-%M-%S)"

: ${JOBDIR:="job-N${NODES}n${PPN}"}
NAME=$(basename ${JOBDIR})

mkdir -p "$JOBDIR"

sed -e "s/<NODES>/$NODES/g" \
    -e "s/<PROCS>/$PPN/g" \
    -e "s/<NAME>/$NAME/g" \
    template.sh >"${JOBDIR}/job.sh"

pushd "$JOBDIR"

# dgx1-prd-05 is broken (as of 26 Nov 2017)
sbatch -N ${NODES} -p dgx1 -x dgx1-prd-05 job.sh $*

popd

