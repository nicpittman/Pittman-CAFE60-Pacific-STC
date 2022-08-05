#!/bin/bash

pds=( 0 0.001 0.0083 0.05 0.0675 0.0833 0.125 0.15 0.2 0.25 0.3 0.35 0.5 )
zd=( 0 0.085 0.113 0.17 0.34 0.68 1.02 1.36 1.7 )

k=0
for (( i=1; i<=12; ++i)) ; do
for (( j=1; j<=8; ++j)) ; do
	k=$(( k + 1 ))
	# echo $i,$j
	memstr=`printf "mem%03d" $k`
	echo  ${memstr}, ${pds[$i]},  ${zd[$j]}
#	ncap2 -O -s  "muezbio = muezbio*0 +${zd[$j]}/86400" -s "muepsbio=muepsbio*0 + ${pds[$i]}/86400" new.nc $memstr.nc 
done
done

for (( mem = 1; mem <= 96; ++mem )); do
	memstr=`printf "mem%03d" $mem`
	echo copy $memstr
#	rm ../enkf-bgc5/$memstr/INPUT/bgc_param.nc
#	cp $memstr.nc ../enkf-bgc5/$memstr/INPUT/bgc_param.nc 
 done
 
#
# change the parameter value for zoo and phyto plankton mortality

# first make muepbio=0
# ncap2 -O -s "muepbio=muepbio*0" bgc_param.nc new.nc 
