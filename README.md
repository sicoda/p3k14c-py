# p3k14c-py

Python package for the p3k14c global archaeological radiocarbon database. (Currently in Progress, come back later!)

[View the interactive p3k14c database map](./interactive_spatial_map.html)

<img width="742" height="345" alt="image" src="https://github.com/user-attachments/assets/67ce4f41-e512-4b5f-bf08-a05a8657e4f9" />

Map showing all available datapoints in the p3k14c database

# Before Running

This Python code assumes your data has already passed through the [p3k14c-data-scrubbing](https://github.com/people3k/p3k14c-data-scrubbing) process. This scrubbing package is essential for quality control and performs the following:

1. Removes records with lab codes from unknown laboratories;
2. Standardizes coordinate formats among records with location data;
3. Handles duplicate entries;
4. Cleans anomalous data; and
5. Obfuscates precise coordinates for dates in the United States and Canada, as well as from the17 dataset, in order to protect site locations.

- For more information on the database and scrubbing methodology, see the official [Scientific Data publication](https://www.nature.com/articles/s41597-022-01118-7).

- See [Data_Prep](Data_Prep.md) for directions on how to use this package to prepare your dataset.

- See the official [p3k14c GitHub page](https://github.com/people3k/p3k14c) for the original implementation using the **R Language**.

**I've attached the cleaned, scrubbed, fuzzed, calibrated dataset on this page (see p3k14c_pristine_dates.csv). You can bypass the finnicky cleaninh and calibrating process entirely.**

# Introduction

The [PAGES People3000 Archaeological Radiocarbon Database](https://www.nature.com/articles/s41597-022-01118-7) (p3k14c) is a comprehensive, global database of archaeological radiocarbon dates. The raw data, however, is uncalibrated and can be messy. This can be daunting for researchers new to coding or who want to reproduce findings. The p3k14c project originally released code in [R](https://github.com/people3k/p3k14c) to help researchers tackle the dataset, relying heavily on great R packages like `rcarbon`.

While R is a fantastic language for statistical analysis and has been heavily adopted by archaeologists, it has limitations in areas pertaining to data engineering, complex GIS integration, machine learning scalability, and large-scale data handling.

This repository bridges that gap. `p3k14c-py` provides detailed Python scripts for the calibration, filtering, and analysis of the p3k14c dataset, enabling better integration into the broader Python data science colloquium.

# Installation and Requirements

To run the scripts in this repository, you will need Python 3.8+ and the following core libraries. You can install them via pip:

```python
pip install pandas plotly shapely scipy numpy matplotlib seaborn scikit-learn iosacal tqdm os
```

# Overview of Scripts

## Calibration

Radiocarbon ages (CRA) must be calibrated to account for historical fluctuations in atmospheric Carbon-14, especially during the Holocene.

**Tool**: We utilize `IOSACal`, an [open-source](https://c14.iosa.it/en/latest/) radiocarbon calibration library in Python.

**Functionality**: The calibration scripts automatically map each date to the correct calibration curve (`IntCal20` for the Northern Hemisphere, `SHCal20` for the Southern Hemisphere) based on the sample's latitude. It extracts key numerical boundaries, such as the median calendar age and 95% confidence intervals, formatted neatly into a `pandas` DataFrame for later analyses. This script outputs a `p3k14c_pristine_dates.csv` file, which will be used in later analyses. 

<img width="4753" height="1752" alt="image" src="https://github.com/user-attachments/assets/cff72b16-2b08-4856-bf5a-1e315571e249" />


<img width="4753" height="1752" alt="image" src="https://github.com/user-attachments/assets/433fba98-494a-42f7-bd44-19cd2d7f99eb" />

**I've attached the cleaned, scrubbed, fuzzed, calibrated dataset on this page (see p3k14c_pristine_dates.csv). You can bypass the finnicky cleaninh and calibrating process entirely.**

## Summary Statistics

A great first step to analysing any large dataset. Gives the researcher a general idea of the dataset's geometry and contents before diving into complex modeling.

**Tools**: We utilize `matplotlib`, `pandas`, and `numpy`.

**Functionality**: Generates summaries of continental and regional data, missing values, error margins, and distributions. In addition to the text report below, this code also provides a `Summary_{Global OR CounrtyName OR SiteName}.png` visual showing Top 10 Countries (or Top 10 materials if limited to specific country/site), Distribution of Error, and Geometry of Uncalibrated Ages (before implementing calibration script) OR Geometry of Calibrated Ages (after implementing the calibration script).

``` ========================================
        DATASET SUMMARY REPORT          
========================================
Total Records: 172,823

--- Continental Breakdown ---
Continent
Europe           74701
North America    63771
Asia             13725
Africa            9354
South America     6435
Name: count, dtype: int64

--- Top 5 Countries ---
Country
USA               55586
United Kingdom    24437
Canada             8185
France             6987
Norway             6475
Name: count, dtype: int64

--- Missing Values ---
Lat               3772
Long              3763
Cal_Median_Age    3772

--- Age & Error Margins ---
                   Count     Mean  Median   Min      Max
Age             172823.0  4542.57  3030.0  30.0  54940.0
Error           172823.0    89.68    55.0  15.0   9300.0
Cal_Median_Age  169051.0  5043.14  3191.0  48.0  53937.0
```

## SPD (Summed Probability Distributions)

SPD's can be used on calibrated radiocarbon data to estimate population fluctuations over time as a proxy for human activity. Essentially, SPD stacks calibrated dates and evaluates the shape they make. Note that this should only be used to visualize archaeological activity, as SPD tends to be mathematically biased due to not incorporating data error. Based on [Palmisano et al. 2021](https://www.sciencedirect.com/science/article/pii/S0277379120307010).

(Weninger et al., 2015; Bevan et al., 2017) has suggested that normalised dates create artificial peaks in the resulting SPDs due to the steepening portions of the radiocarbon calibration curve 

**Functionality**: Because Python lacks a direct equivalent to R's `rcarbon::spd()`, this module provides custom scripts to:
  - Standardize and "bin" the data (using `scikit-learn`'s agglomerative clustering) to prevent heavily-sampled sites from skewing the distribution.
  - Calculate the probability matrices for each date.
  - Aggregate these probabilities into a unified temporal time-series.
  - Perform rolling averages and confidence envelope generation using `scipy` and `numpy`.

<img width="800" height="500" alt="image" src="https://github.com/user-attachments/assets/4934d33b-647d-44f1-9fae-dadf47f06baf" />


### End-to-End Bayesian Analysis

Next, move to the **Bayesian script** for a final, detailed analysis. This analysis evaluates the simplest underlying population shape that could have produced the calibrated radiocarbon data. This analysis provides a cleaner, more realistic demographic curve that filters out the noise in radiocarbon data. Based on [Price et al. 2021](https://www.sciencedirect.com/science/article/pii/S0277379120307010).

<img width="800" height="500" alt="image" src="https://github.com/user-attachments/assets/85d036ff-4e50-413c-b7f6-6734cb33919b" />

### Comparing the Two Approaches

<img width="800" height="500" alt="image" src="https://github.com/user-attachments/assets/7cb519a0-e57d-445b-847d-7f74885de60e" />

**Why the peak shift if they are based on the same dataset?** The SPD is influenced by the median of each calibrated date, while the Bayesian model accounts for the uncertainty of every date simultaneously. Dates with large error are "weighted" differently in the Bayesian likelihood than they are when summed in an SPD, leading to shifts.

**Why the jagged SPD peaks?** SPD acts like a mirror. If the calibration curve has a plateau, many different radiocarbon dates will pile up into that same calendar window. This creates a massive, artificial spike in the SPD that looks like a population spike. Meanwhile, Bayesian uses the calibration curve as part of its analysis. It recognizes where these plateaus are and mathematically irons out the artificial spikes, resulting in the smoother trend.

**Which analysis should I use?** Use the SPD model if you want to show archaeological activity (where we have the most evidence and the most dates). Use the Bayesian model if you want to discuss general demographic trends (the most likely "true" shape of the population over time, reduced calibration noise).

<img width="800" height="500" alt="image" src="https://github.com/user-attachments/assets/f685e884-cb41-4008-9e6e-31bb5d5777f6" />

Notice that the Africa data SPD and Bayesian models align because of the massive scale of the data, which acts as a sort of manual smoother for the SPD. This is due to the random fluctuations of the calibration curve, which are overwhelmed by the volume of the data.

### Growth Rate

This analysis mathematically describes the speed of population change at a particular site/country/continent. This is calculated as the first derivative of the log-transformed Bayesian curve. A positive value = population increase, a negative value = population decrease.

<img width="800" height="500" alt="image" src="https://github.com/user-attachments/assets/586f177b-82be-4fcb-9065-fdf786db27a2" />


## Risk & Density Analysis

Replicating the spatial relative risk surfaces often generated using R's `sparr` package.

Functionality: Uses `scipy.stats.gaussian_kde` to create continuous Kernel Density Estimation (KDE) surfaces for recorded sites versus dated sites, allowing researchers to visualize spatial sampling biases across the globe.

TBD

## Data Merging

Archaeological data is inherently spatial. This module utilizes `geopandas` to merge and filter the p3k14c data against external spatial and environmental datasets.

TBD

# Example Figures

TBD





