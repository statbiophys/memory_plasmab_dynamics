#!bin/bash

# Must be run in the hilary conda environment

sample_name="pat2_hilary_heavy.tsv"

infer-lineages full-method lineages/src_data/$sample_name --result-folder lineages/hilary_out/ 

sample_name_h="pat2_hilary_pairs_h.tsv"
sample_name_l="pat2_hilary_pairs_l.tsv"
infer-lineages full-method lineages/src_data/$sample_name_h --kappa-file lineages/src_data/$sample_name_l --result-folder lineages/hilary_out/

