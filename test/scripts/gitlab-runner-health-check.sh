#!/bin/bash
hostname=$(hostname)

persistence_on=$(nvidia-smi --format=csv,noheader --query-gpu=persistence_mode | grep Enabled)
if [ -z "$persistence_on" ]; then
    pm="OFF"
else
    pm="ON"
fi

ib_active=$(ibv_devinfo | grep ACTIVE)
if [ -z "$ib_active" ]; then
    ib_status="FAIL"
else
    ib_status="OK"
fi


printf "$hostname: GPU persistence mode $pm, IB interfaces $ib_status\n"

if [ "$pm" = "OFF" -o "$ib_status" = "FAIL" ]; then
	exit 1
fi


