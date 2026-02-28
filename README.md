# Urban Sinkhole SPI Framework

This repository contains the Python scripts used in the study:

"Time-Series Integration of Groundwater, Rainfall, and Satellite Soil Moisture for Urban Sinkhole Precursor Detection"

## Description

The framework integrates:
- Groundwater level
- Groundwater electrical conductivity
- Rainfall
- SMAP root-zone soil moisture

to compute a Sinkhole Potential Index (SPI) using multi-scale accumulation windows.

## Structure

- scripts/: SPI computation and analysis code
- data_sample/: example dataset
- figures/: example outputs

## Requirements

Install required packages:

pip install -r requirements.txt

## Reproducibility

All figures in the manuscript can be reproduced by running:

python scripts/spi_calculation.py
