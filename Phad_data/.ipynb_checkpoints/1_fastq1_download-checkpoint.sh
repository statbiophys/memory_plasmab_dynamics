#!/bin/bash


# The dataset is divided in two parts, this scripts download the first part,
# the other script ending with D1pc the second smaller part composed on plasmablasts
# of the first donor


groups=(1 2 3 4 5 6 7 8 9 10)


FASTQDIR='/home/andrea/Documents/Immunology/HealthyBCells/Phad_data/fastq/'
cd $FASTQDIR


while read -r line
do
	group=$(echo "$line" | awk -F'\t' '{printf "%s", $39}' | cut -d" " -f2)

	for g in "${groups[@]}"
	do
		if [ $g == $group ]
		then		
			# Downloading
			address=$(echo "$line" | awk -F'\t' '{printf "%s", $45}')
			wget $address
			
			# Renaming the sample
			name=$(echo "$line" | awk -F'\t' '{printf "%s", $43}')
			id=$(echo "$name" | cut -d'_' -f1)
			S=$(echo "$name" | cut -d'_' -f2)
			S=${S:0:2} # Dropping other letters after S, redundand after specifying the group in the name
			new_name=g"$g"_"$id"_"$S"_$(echo "$name" | cut -d'_' -f3-5)
			mv $name $new_name
		fi
	done	
	
done < ../metadata/metadata_fastq1.tsv
