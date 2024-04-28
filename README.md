# parallel-implicit-crime


This repository contains code to reproduce experiments described in the abstract "Parallel Imaging Reconstruction in Public Datasets Biases Downstream Analysis in Retrospective Sampling Studies", which will be presented at ISMRM 2024 on Tuesday, 07 May 2024.


Dependencies:

- BART toolbox. See <https://github.com/mrirecon/bart>
- Python3 and toolboxes: numpy, pandas, multiprocessing, h5py, cfl
- fastMRI data as .h5 files, to be placed in /data/fastMRI, with required file names listed in /data/fastMRI/req_fastMRI.txt


Contents:

- /scripts/sense_experiment.py: functions for data synthesizing, processing and reconstruction
- /data/in_vivo/: data for in vivo experiment
- /data/fastMRI/: data for fastMRI experiment to be placed here
- /figure3/figure3.py: recreates figure 3 plot
- /figure4/figure4.py: recreates figure 4 plot
- /figure5/figure5.py: recreates figure 5 plot 
