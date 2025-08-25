#!/bin/bash


groups=(1 2 3)
n_old_groups=10


FASTQDIR='/home/andrea/Documents/Immunology/HealthyBCells/Phad_data/fastq/'
cd $FASTQDIR


while read -r line
do
	group=$(echo "$line" | awk -F'\t' '{printf "%s", $38}' | cut -d" " -f2)
	#echo $group
	
	for g in "${groups[@]}"
	do
		if [ $g == $group ]
		then		
			# Downloading
			address=$(echo "$line" | awk -F'\t' '{printf "%s", $44}')
			wget $address
			
			# Renaming the sample
			name=$(echo "$line" | awk -F'\t' '{printf "%s", $42}')
			id=$(echo "$name" | cut -d'_' -f1)
			S=$(echo "$name" | cut -d'_' -f2)
			S=${S:0:2} # Dropping other letters after S, redundand after specifying the group in the name
			g=$(($g + $n_old_groups))
			new_name=g"$g"_"$id"_"$S"_$(echo "$name" | cut -d'_' -f3-5)
			mv $name $new_name
		fi
	done	
	
done < ../metadata/metadata_fastq2.tsv
