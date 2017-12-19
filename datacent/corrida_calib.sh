#!/bin/bash
BASE_FOLDER="/usr/local/data"
FILE_NAME="br_calib_orden"
FILE="${BASE_FOLDER}/${FILE_NAME}"
FILE_BASE="br_calib_"

HV_BASE=32525
HV_STEP=55

cd /usr/local/bin
for i in $(seq 0 44);
do
	echo "br_calib_$i	=>	$(($HV_BASE - $i*$HV_STEP ))" >> "${FILE}"
	HV_Control -h HST -a 0 -b 1272 -c 47063 -d 0 -e 1272 -f $(($HV_BASE - $i*$HV_STEP )) -p /dev/ttyS1
	echo HV_Control -h HST -a 0 -b 1272 -c 47063 -d 0 -e 1272 -f $(($HV_BASE - $i*$HV_STEP )) -p /dev/ttyS1 >> "${FILE}"
	HV_Control -h HPO -p /dev/ttyS1 >> "${FILE}"
	./cal >> "${BASE_FOLDER}/${FILE_BASE}$i"
	HV_Control -h HPO -p /dev/ttyS1 >> "${FILE}"
done
