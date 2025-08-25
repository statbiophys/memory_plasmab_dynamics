#!/bin/bash

# Exectute the igBlast alignment for the samples contined in the samples_fastq1.txt or samples_fastq2.txt list.


CLONEDIR='/home/andrea/Documents/Immunology/HealthyBCells/Phad_data/changeO_out/'
CELLRANGERDIR='/home/andrea/Documents/Immunology/HealthyBCells/Phad_data/cellranger_out/'


while read -r sample
do	
	if [[ ${sample:0:1} == "#" ]]
		then continue
	fi
	
	if [ ! -f $CELLRANGERDIR$sample"/outs/filtered_contig.fasta" ]
	then
		echo "Cellranger out of "$sample" does not exists"
		continue
	fi
	
	if [ -f $CLONEDIR$sample"/filtered_contig_igblast_db-pass.tsv" ]
	then
		echo $sample" already processed"
		continue
	fi
		
	
	# Removing unnecessary files in cellranger out
	if [ -d $CELLRANGERDIR$sample"/SC_VDJ_ASSEMBLER_CS" ] 
	then
		cd $CELLRANGERDIR$sample
		mv outs/filtered_contig.fasta filtered_contig.fasta
		mv outs/filtered_contig_annotations.csv filtered_contig_annotations.csv
		rm -r outs/*
		mv filtered_contig.fasta outs/filtered_contig.fasta
		mv filtered_contig_annotations.csv outs/filtered_contig_annotations.csv
		
		rm -r SC_VDJ_ASSEMBLER_CS/*
		rm -d SC_VDJ_ASSEMBLER_CS
		rm ./*
		cd ../..
	fi
	
	
	mkdir -p $CLONEDIR/$sample/
	
	AssignGenes.py igblast \
		-s $CELLRANGERDIR$sample/outs/filtered_contig.fasta \
		-b /home/andrea/.local/share/igblast \
		-o $CLONEDIR/$sample/filtered_contig_igblast.fmt7
		--organism human --loci ig --format blast
		
	MakeDb.py igblast \
		-i $CLONEDIR/$sample/filtered_contig_igblast.fmt7 \
		-s "$CELLRANGERDIR""$sample"/outs/filtered_contig.fasta \
		-r /home/andrea/.local/share/germlines/imgt/human/vdj \
		--10x $CELLRANGERDIR$sample/outs/filtered_contig_annotations.csv \
		--extended --failed

	
done < metadata/samples_fastq2.txt

