#!/bin/bash


LINEAGEDIR='/home/andrea/Documents/Immunology/HealthyBCells/Phad_data/lineages/'

DefineClones.py -d $LINEAGEDIR"in_data/pat1_changeO.tsv" --act set --model ham --norm len --dist 0.15 \ 
    --outdir $LINEAGEDIR"changeO_out/"

DefineClones.py -d $LINEAGEDIR"in_data/pat2_changeO.tsv" --act set --model ham --norm len --dist 0.15 \
    --outdir $LINEAGEDIR"changeO_out/"