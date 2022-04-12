## Frequency spectra and RELP - Speech Signal Processing (ZRE) project
#### Author: Alexander Polok ([xpolok03@fit.vutbr.cz](mailto:xpolok03@fit.vutbr.cz))
#### Date: 12.4.2022

## Installation
Change directory to the root of the project.
```bash
cd __PROJECT_ROOT__
```

### Configure anaconda enviroment
Create new environment with [anaconda distribution](https://www.anaconda.com/) and activate it.
```bash
conda create -n ZRE python=3.9 --yes
conda activate ZRE
```
Install required packages
```bash
pip install -r requirements.txt
```


## Start up jupyter notebook.

```bash
jupyter notebook
```

## Cleanup environment
```bash
conda deactivate
conda remove --name ZRE --all --yes
```
