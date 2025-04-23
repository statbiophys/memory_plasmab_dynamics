#!/bin/bash


# Exectute the cell ranger analysis for the samples contined in the samples_fastq1.txt or samples_fastq2.txt list.


FASTQDIR='/home/andrea/Documents/Immunology/HealthyBCells/Phad_data/fastq/'
CELLRANGERDIR='/home/andrea/Documents/Immunology/HealthyBCells/Phad_data/cellranger_out/'
CELLRANGERREF='/home/andrea/.local/cellranger-7.1.0/refdata-cellranger-vdj-GRCh38-alts-ensembl-5.0.0'

cd $CELLRANGERDIR

while read -r sample
do	
	if [[ ${sample:0:1} == "#" ]]
		then continue
	fi
	
	cellranger vdj --id=$sample \
                 --reference=$CELLRANGERREF \
                 --fastqs=$FASTQDIR \
                 --sample=$sample \
                 --localcores=6 \
                 --localmem=8
                 
done < ../metadata/samples_fastq1.txt
