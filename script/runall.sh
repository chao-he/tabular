export LD_LIBRARY_PATH=$HOME/miniconda3/envs/cxxdev/lib/

doi=acs.jmedchem.0c00224
doi=acs.jmedchem.1c01682
doi=jm401642q
rm -rf images/$doi/*
python ./p01-pdf2table.py $doi
python ./p02-segment-text.py $doi
python ./p03-segment-molecular.py $doi
python ./p04-mol2smiles.py $doi
