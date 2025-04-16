#!/bin/bash

# Utility script for housing cluster-related functions

function source_cluster_config() {
    config_file="docker/$1-config.sh"

    if [ ! -f "$config_file" ]; then
        echo "ERROR: Config file $config_file is not found."
        usage
    fi

    source $config_file
}

function identify_cluster() {
    hostname="$(hostname)"

    if [[ "$hostname" =~ ^gc[01][0-9]$ ]]; then
        echo "gc"
	return
    fi

    if [[ "$hostname" =~ eos.clusters.nvidia.com$ ]]; then
        echo "eos"
	return
    fi

    if [[ "$hostname" =~ draco-rno-login- ]]; then
	echo "draco-rno"
	return
    fi

    if [[ "$hostname" =~ draco-oci-login- ]]; then
        echo "draco-oci"
	return
    fi

    if [[ "$hostname" =~ .*prenyx.*clusters\.nvidia\.com$ ]]; then
        echo "pre-nyx"
	return
    fi

    if [[ "$hostname" =~ .*ptyche.*$ ]]; then
        echo "pre-tyche"
	return
    fi

    # copy the gc setup in pdx
    if [[ "$(hostname -f)" =~ pdx02.us.nvidia.com$ ]]; then
        echo "gc"
        return
    fi

    echo "UNKNOWN"
}
