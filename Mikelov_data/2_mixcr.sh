#!/bin/bash


fastq_dir="fastq/"
out_dir="mixcr_out/"
n_proc="3"

chunks=("72_chunk_1" "72_chunk_2" "72_chunk_3")
#while read -r line   
for sample in ${chunks[@]}
do
    
    #sample=$(echo "$line" | awk -F'\t' '{printf "%s", $1}')
    
    if [ "$sample" == "sample_n" ]
        then
        continue
    fi
    
    if [ "$sample" == "72" ]
        then
        continue
    fi
    
    if [ -f "${out_dir}""${sample}"".clones_IGH.tsv" ]
    then
        echo "Clone file '$sample' already exists. It won't be processed."
        continue
    fi
    
    mixcr analyze mikelov-et-al-2021 --threads $n_proc -Xmx8g --no-reports --no-json-reports \
          ${fastq_dir}${sample}_1.fastq \
          ${fastq_dir}${sample}_2.fastq \
          ${out_dir}${sample}

    rm -f ${out_dir}${sample}.alignments.vdjca
    rm -f ${out_dir}${sample}.clns
    rm -f ${out_dir}${sample}.refined.vdjca
    rm -f ${out_dir}${sample}.qc.json
    rm -f ${out_dir}${sample}.qc.txt
      
#done < metadata/metadata.tsv
done