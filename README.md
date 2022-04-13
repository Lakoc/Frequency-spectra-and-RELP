## Frequency spectra and RELP - Speech Signal Processing (ZRE) project
#### Author: Alexander Polok ([xpolok03@fit.vutbr.cz](mailto:xpolok03@fit.vutbr.cz))
#### Date: 12.4.2022

## Installation
Change the current working directory to the root of the project.
```bash
cd __PROJECT_ROOT__
```

### Configure anaconda environment
Create new environment with [anaconda distribution](https://www.anaconda.com/) and activate it.
```bash
conda create -n ZRE python=3.9 --yes
conda activate ZRE
```

### Install required packages
```bash
pip install -r requirements.txt
```


## How to run
Start up jupyter notebook and open `ZRE_proj.ipynb`.
```bash
jupyter notebook
```

> **_NOTE:_** Project outputs could be also visualized directly in the [Google collab](https://colab.research.google.com/github/Lakoc/ZRE/blob/main/ZRE_proj.ipynb).

## Cleanup
```bash
conda deactivate
conda remove --name ZRE --all --yes
```
