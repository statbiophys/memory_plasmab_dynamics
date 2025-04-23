#!/bin/bash


CLONEDIR='/home/andrea/Documents/Immunology/HealthyBCells/Mikelov_data/changeO_out/'
MIXCRDIR='/home/andrea/Documents/Immunology/HealthyBCells/Mikelov_data/mixcr_out/'


while read -r line
do
    
    sample=$(echo "$line" | awk -F'\t' '{printf "%s", $1}')
    
    if [ "$sample" == "sample_n" ]
        then
        continue
    fi
    
    if [ ! -f "${MIXCRDIR}""${sample}""_seqs.fasta" ]
    then
        echo "Clone file '$sample' already exists. It won't be processed."
        continue
    fi
    
    if [ -f "${CLONEDIR}""${sample}""_igblast_db-pass.tsv" ]
    then
        echo "ChangeO file '$sample' already exists. It won't be processed."
        continue
    fi
    
    echo $sample
    
    
    AssignGenes.py igblast \
        -s $MIXCRDIR"$sample"_seqs.fasta \
        -b /home/andrea/.local/share/igblast \
        -o $CLONEDIR/"$sample"_igblast.fmt7 \
        --organism human --loci ig --format blast --nproc 4

    MakeDb.py igblast \
        -i $CLONEDIR/"$sample"_igblast.fmt7 \
        -s $MIXCRDIR"$sample"_seqs.fasta \
        -r /home/andrea/.local/share/germlines/imgt/human/vdj
      
    rm -f $CLONEDIR/"$sample"_igblast.fmt7
    
done < metadata/metadata.tsv
        