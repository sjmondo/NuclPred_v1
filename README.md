NuclPred v1.0
================

**Author**: Stephen J. Mondo, \<<sjmondo@lbl.gov>\>

**Copyright**:

NuclPred Copyright (c) 2024, The Regents of the University of California, through Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from the U.S. Dept. of Energy). All rights reserved.

If you have questions about your rights to use or distribute this software, please contact Berkeley Lab's Intellectual Property Office at IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department of Energy and the U.S. Government consequently retains certain rights.  As such, the U.S. Government has been granted for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software to reproduce, distribute copies to the public, prepare derivative works, and perform publicly and display publicly, and to permit others to do so.

## Overview

**NuclPred** is a **Nucl**eosome **Pred**iction tool. Trained using *in vitro* nucleosome data from fungi, it aims to calculate the physical preference of histones for certain DNA profiles, creating a histone-DNA favorability map. It is written in Python and utilizes a hybrid deep learning architecture comprised of multiple convolutional and recurrent neural network layers, which takes in DNA and shape parameters (calculated using DNAshapeR; Chiu et al., 2016, doi:10.1093/bioinformatics/btv735), then predicts occupancy. 

NuclPred was trained using ~28 million sites from 4 fungi distributed across the Kingdom (*Meredithblackwellia eburnea*, *Usnea florida*, *Rhodosporidium toruloides* and *Catenaria anguillulae*) and trained to predict *in vitro* nucleosome occupancy.


## Installation

**NuclPred** requires specific versions of dependencies. We recommend using Anaconda to install and store dependencies required to run **NuclPred** within a single conda environment. For install instructions, see https://docs.anaconda.com/anaconda/install/. 

Once installed, navigate to the folder you unpackaged NuclPred into (e.g. NuclPred_v1), then you can create a conda environment containing all dependencies as below: 

```bash
cd NuclPred_v1
conda env create -f ./nucleosome_prediction_env.yml
```

You can then activate the environment by typing: 

```bash
conda activate nucleosome_prediction_env 
```

You will also need an R package, DNAshapeR (Chiu et al., 2016, doi:10.1093/bioinformatics/btv735), to generate input shape data for **NuclPred**. You can download and install DNAshapeR following the instructions here: https://www.bioconductor.org/packages/release/bioc/html/DNAshapeR.html

## NuclPred dependencies

* `_libgcc_mutex=0.1`
* `_openmp_mutex=4.5`
* `_tflow_select=2.3.0`
* `absl-py=0.15.0`
* `astor=0.8.1`
* `biopython=1.79`
* `blas=1.0`
* `c-ares=1.19.1`
* `ca-certificates=2024.9.24`
* `certifi=2021.5.30`
* `cudatoolkit=11.8.0`
* `dataclasses=0.8`
* `gast=0.5.3`
* `google-pasta=0.2.0`
* `grpcio=1.31.0`
* `h5py=2.10.0`
* `hdf5=1.10.6`
* `intel-openmp=2022.1.0`
* `joblib=1.0.1`
* `keras=2.2.4`
* `keras-applications=1.0.8`
* `keras-base=2.2.4`
* `keras-preprocessing=1.1.2`
* `ld_impl_linux-64=2.40`
* `libffi=3.3`
* `libgcc=14.2.0`
* `libgcc-ng=14.2.`
* `libgfortran-ng=7.5.0`
* `libgfortran4=7.5.0`
* `libgomp=14.2.0`
* `libprotobuf=3.17.2`
* `libstdcxx=14.2.0`
* `libstdcxx-ng=14.2.0`
* `markdown=3.1.1`
* `mkl=2020.2`
* `mkl-service=2.3.0`
* `mkl_fft=1.3.0`
* `mkl_random=1.1.1`
* `ncurses=6.4`
* `numpy=1.19.2`
* `numpy-base=1.19.2`
* `openssl=1.1.1w`
* `pandas=1.1.5`
* `pip=21.2.2`
* `protobuf=3.17.2`
* `python=3.6.13`
* `python-dateutil=2.8.2`
* `python_abi=3.6`
* `pytz=2021.3`
* `pyyaml=5.4.1`
* `readline=8.2`
* `scikit-learn=0.24.2`
* `scipy=1.5.2`
* `setuptools=58.0.4`
* `six=1.16.0`
* `sqlite=3.45.3`
* `tensorboard=1.14.0`
* `tensorflow=1.14.0`
* `tensorflow-base=1.14.0`
* `tensorflow-estimator=1.14.0`
* `termcolor=1.1.0=`
* `threadpoolctl=2.2.0`
* `tk=8.6.14`
* `werkzeug=2.0.3`
* `wheel=0.37.1`
* `wrapt=1.12.1`
* `xz=5.4.6`
* `yaml=0.2.5`
* `zlib=1.2.13`

## Input data

NuclPred requires input data from two sources:

1) a genome assembly, fasta format (an example fasta is provided in the examples folder)
2) a folder containing DNA shape parameters. To calculate shape parameters, install DNAshapeR and using R, run:

```
library(DNAshapeR)
pred <- getShape("YOUR_ASSEMBLY.fasta")
```

By default, shape files will be stored in the same folder as your assembly with extensions for the different types.

## Running NuclPred

One input data are generated, you can predict nucleosome occupancy. For all steps, you will need the `nucleosome_prediction_env` conda environment loaded:

```bash
conda activate nucleosome_prediction_env 
```

After the environment is loaded, format the data matrix for prediction:
```bash
./generate_matrix.py --assembly in.fasta --data_dir shapes_directory_in --organism_scaffold_label label > nucleosome_prediction_input.matrix
```

Organism labels are attached to scaffolds to allow multiple lineages to be analyzed simultaneously. Matrices from several taxa can be concatenated like so:

```bash
cat organism1_nucleosome_prediction_input.matrix organism2_nucleosome_prediction_input.matrix ... organismN_nucleosome_prediction_input.matrix > all_organisms_merged.matrix
```

Next, run nucleosome prediction. This can be done on CPU but is faster on GPU:

```bash
./predict_nucleosomes.py --input nucleosome_prediction_input.matrix > predictions.wig
```

Note that this step can take quite some time, especially if you are running a large dataset on CPU. If a high performance computing center is available to you, one option would be splitting sequences by scaffold or sets of scaffolds and farming jobs out in an array.

### Example (using files available in examples folder):

```bash
conda activate nucleosome_prediction_env
./generate_matrix.py --assembly examples/Mereb1.fasta --data_dir examples/ --organism_scaffold_label Mereb1 > Mereb1_input.matrix #this will generate a full dataset for Meredithblackwellia eburnea. Best to use the test file available in examples folder for a quick test.
./predict_nucleosomes.py --input examples/Mereb1_example_nucleosome_prediction_input.matrix > Mereb1_nucleosome_predictions.wig
```

## Outputs:

`generate_matrix.py` produces a data matrix with the following columns, in order:

* `organism_name.scaffold` organism_name is input using --organism_scaffold_label
* `nucleotide` 0 = A, 1 = T, 2 = C, 3 = G, 4 = N
* `average GC +/- 75bp` Average GC +/- 75 basepairs surrounding each position in the assembly is calculated. The first and last 75 bp of each scaffold are set to 0.
* `electrostatic potential` Vector of electrostatic potential, calculated using DNAshapeR
* `helical twist` Vector of helical twist, calculated using DNAshapeR
* `propellar twist` Vector of propellar twist, calculated using DNAshapeR
* `roll` Vector of roll, calculated using DNAshapeR
* `minor groove width` Vector of minor groove width, calculated using DNAshapeR

`predict_nucleosomes.py` produces a wig file (start, step and span=1) with z-score normalized occupancy predictions. As this model uses a lookback of 80bp for occupancy prediction, the first 80bp of each scaffold are sacrificed and therefore set to 0 in the resulting wig file. An example output is available in the examples folder. See: `examples/Mereb1_example_nucleosome_prediction_output.wig`. This file can then be parsed, loaded into a genome viewer, etc.
