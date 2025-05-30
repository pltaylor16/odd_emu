#!/bin/bash
python driver.py
python merge_batches.py
compress_params.py
python get_crit.py
source /srv/scratch1/taylor.4264/anaconda3/etc/profile.d/conda.sh
conda activate odd_emu
python model_driver.py
