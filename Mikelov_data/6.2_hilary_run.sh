#!bin/bash

# Must be run in the hilary conda environment

sample_names=("MT.tsv" "MRK.tsv" "IZ.tsv" "IM.tsv" "AT.tsv" "D01.tsv")

for sample in ${sample_names[@]}
do
    infer-lineages full-method lineages/seqs_in/$sample --result-folder lineages/hilary_out/ 
    echo $sample
done
