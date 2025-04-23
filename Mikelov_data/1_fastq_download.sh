#!/bin/bash


FASTQDIR='/home/andrea/Documents/Immunology/HealthyBCells/Mikelov_data/fastq/'
cd $FASTQDIR

metadata_path="../metadata/metadata_samples.tsv" # new metadata
echo -e "sample_n\tpatient\tage\ttime\treplicate\tcell" > "$metadata_path"

while read -r line
do

    cell=$(echo "$line" | awk -F'\t' '{printf "%s", $14}')
    if [ "$cell" == "memory B cell" ]
        then
        cell="mem"
    elif [ "$cell" == "plasmablast" ]
        then
        cell="pb"
    else
        continue
    fi
        
    sample_n=$(echo "$line" | awk -F'\t' '{printf "%s", $1}' | cut -d" " -f2)    
    pat=$(echo "$line" | awk -F'\t' '{printf "%s", $4}')
    age=$(echo "$line" | awk -F'\t' '{printf "%s", $7}')
    time=$(echo "$line" | awk -F'\t' '{printf "%s", $15}')
    time=${time:2:1}
    repl=$(echo "$line" | awk -F'\t' '{printf "%s", $16}')
    
    if [ "$pair" == "1" ]
        then
        echo -e "$sample_n""\t""$pat""\t""$age""\t""$time""\t""$repl""\t""$cell" >> "$metadata_path"
    fi
    
    pair=$(echo "$line" | awk -F'\t' '{printf "%s", $36}' | cut -d"_" -f2 | cut -d"." -f1)
    address=$(echo "$line" | awk -F'\t' '{printf "%s", $36}')
    old_name=$(echo "$line" | awk -F'\t' '{printf "%s", $36}' | cut -d"/" -f9)
    name="$sample_n""_""$pair"".fastq.gz"
    
    
    if [ -f "$name" ]
    then
        echo "File '$name' already exists. It won't be downloaded."
        continue
    fi
    
    #wget $address
    #mv "$old_name" "$name"

	
done < ../metadata/meta_database.tsv # metadata from ArrayExpress
