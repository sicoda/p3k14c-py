# p3k14c-py

Python package for the p3k14c global archaeological radiocarbon database. (Currently in Progress, come back later!)

[View the interactive p3k14c database map](./interactive_spatial_map.html)

<img width="742" height="345" alt="image" src="https://github.com/user-attachments/assets/67ce4f41-e512-4b5f-bf08-a05a8657e4f9" />

Map showing all available datapoints in the p3k14c database

# Before Running

For more information on the database and scrubbing methodology, see the official [Scientific Data publication](https://www.nature.com/articles/s41597-022-01118-7).

See the official [p3k14c GitHub page](https://github.com/people3k/p3k14c) for the original implementation using the **R Language**.

**I've attached the cleaned and calibrated dataset on this page (see p3k14c_pristine_dates.csv). You can bypass the finicky cleaning and calibrating process entirely.**

# Introduction

The [PAGES People3000 Archaeological Radiocarbon Database](https://www.nature.com/articles/s41597-022-01118-7) (p3k14c) is a comprehensive, global database of archaeological radiocarbon dates. The raw data, however, is uncalibrated and can be messy. This can be daunting for researchers new to coding or who want to reproduce findings. The p3k14c project originally released code in [R](https://github.com/people3k/p3k14c) to help researchers tackle the dataset, relying heavily on great R packages like `rcarbon`.

While R is a fantastic language for statistical analysis and has been heavily adopted by archaeologists, it has limitations in areas pertaining to data engineering, complex GIS integration, machine learning scalability, and large-scale data handling.

This repository bridges that gap. `p3k14c-py` provides detailed Python scripts for the calibration, filtering, and analysis of the p3k14c dataset, enabling better integration into the broader Python data science community.

# Installation and Requirements

To run the scripts in this repository, you will need Python 3.8+ and the following core libraries. You can install them via pip:

```python
pip install pandas plotly shapely scipy numpy matplotlib seaborn scikit-learn iosacal tqdm os
```

# Overview of Scripts

## Cleaning

This cleaning script is an essential step for quality control and later computational analyses. This script is adapted from  [p3k14c-data-scrubbing](https://github.com/people3k/p3k14c-data-scrubbing) to work using Python 3.12.10 and performs the following:

1. Lab-code validation via Labs.csv (typo correction included)
2. LabID quality checks: must contain a numeral, no '?', no corrupted Unicode
3. LabID standardisation: strip punctuation, uppercase, insert dash
4. Coordinate-format conversion (deg/min/sec, Solheim Northing/Easting)
5. SiteName / SiteID whitespace stripping
6. Duplicate removal (LabID-exact; source-priority tiebreaking when DatasetFamilyTree.csv is present, first-occurrence fallback otherwise)
7. Miscellaneous scrubbing:
        - null Age / Error
        - Age and Error must be whole numbers (integers)
        - Age > 0  (future dates removed)
        - Error >= 15 BP  (impossibly small errors removed)
        - Error <= Age
        - Age <= 55,000 BP
        - "United States" normalized to "USA"
8. Hardcoded bad-coordinate patches for 6 known problem LabIDs
9. Encoding repair on text columns (ftfy when installed, stdlib fallback)
10. Column sanitisation: strips stray quotes, commas, exotic whitespace
11. Outputs: cleaned CSV, graveyard.csv, unknown_codes.csv

```
============================================================
  Input    : 173,946 records
  Output   : 172,823 records
  Removed  : 1,123 records  (0.6%)
  Cleaned  : cleaned_p3k14c.csv
  Graveyard: graveyard.csv
  Unknown  : unknown_codes.csv
============================================================
```

## Calibration

Radiocarbon ages (CRA) must be calibrated to account for historical fluctuations in atmospheric Carbon-14, especially during the Holocene.

**Tool**: We utilize `IOSACal`, an [open-source](https://c14.iosa.it/en/latest/) radiocarbon calibration library in Python.

**Functionality**: The calibration scripts automatically map each date to the correct calibration curve (`IntCal20` for the Northern Hemisphere, `SHCal20` for the Southern Hemisphere) based on the sample's latitude. It extracts key numerical boundaries, such as the median calendar age and 95% confidence intervals, formatted neatly into a `pandas` DataFrame for later analyses. This script outputs a `p3k14c_pristine_dates.csv` file, which will be used in later analyses. 

1. Curve selection: for every row, it checks the Lat column. Any sample at or above the equator gets `intcal20`; anything below gets `shcal20`. This is the standard approach for global datasets.
2. Calibration: calls R(age, error, lab_id).calibrate(curve) from `IOSACal`, exactly as the official library intends.
3. Extracting statistics: calls .quantiles() on each CalAge result, which returns the median (quants[50]) and the 1σ / 2σ confidence intervals (quants[68], quants[95]).
4. Failure handling: rows with missing Age, Error, or Lat, or any row where IOSACal throws an exception, are written to calibration_failures.csv rather than crashing the whole run.

<img width="4753" height="1752" alt="image" src="https://github.com/user-attachments/assets/cff72b16-2b08-4856-bf5a-1e315571e249" />


<img width="4753" height="1752" alt="image" src="https://github.com/user-attachments/assets/433fba98-494a-42f7-bd44-19cd2d7f99eb" />

**I've attached the cleaned and calibrated dataset on this page (see p3k14c_pristine_dates.csv). You can bypass the finicky cleaning and calibrating process entirely.**

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

SPD's can be used on calibrated radiocarbon data to estimate demographic fluctuations over time as a proxy for human activity. Essentially, the frequency of datable anthropogenic carbon recovered from archaeological contexts serves as a direct proxy for past fluctuations in human population density and associated settlement activity. [Palmisano et al. 2021](https://www.sciencedirect.com/science/article/pii/S0277379120307010).

Functionality: This script combines several methodologies from paleodemography:

1. Chronometric Hygiene: Filters out problematic materials (old-wood/marine effects), drops large-error dates.
2. Spatial-Temoral Binning: Controls oversampling biases from single archaeological phases.
3. Taphonomic Correction: Applies power-function corrections (Surovell et al. 2009 and Bluhm & Surovell 2018) to account for the natural decay and loss of organic material over time.
4. Null Hypothesis Significance Testing (NHST): Uses 5,000-iteration (adjustable) Monte Carlo simulations to test the empirical SPD against exponential and logistic null models, highlighting periods of statistically significant population deviation.
5. Continuous Piecewise Linear (CPL) Modelling: Uses differential evolution and Bayesian Information Criterion (BIC) to identify optimal "hinge points" that represent major regime shifts in population growth and decline.

Case Study: Catalhoyuk

<img width="5345" height="3640" alt="image" src="https://github.com/user-attachments/assets/ad37a6ea-1c5a-44ab-b0b4-ed6d5a170a42" />

Because the model prioritizes mathematical precision over roundness, 

By identifying hinge points, peaks of a civilization's footprint on the landscape can be elucidated. The more complex models (like the 3-hinge or 4-hinge) simply add nuance, showing us the "plateaus" and "stutters" that happened on the way up and the way down.

Significantly, the moment after this single hinge point marks the beginning of a long, slow trajectory of decline. While the population didn't vanish overnight, the architectural density began to loosen. In the final phases of Çatalhöyük East (leading up to the abandonment of the East mound and the shift to the West mound), the community started leaving more open courtyard spaces, and the sheer volume of radiocarbon-dated activity begins to taper off.

## Density Analysis

This analysis shows the geographical distribution of unweighted sites across a region (Plot a) and the weighted C14 dates at those sites (Plot b). This script allows the user to pick a specific temporal range and region (one of the seven continents) or to create a custom region (lat. & long.) to analyze. 

Functionality: Uses `scipy.stats.gaussian_kde` to create continuous Kernel Density Estimation (KDE) surfaces for recorded sites vs. dated sites. Provides a visual of spatial sampling biases across the globe. Here is an example for North America:

<img width="550" height="800" alt="image" src="https://github.com/user-attachments/assets/c61ccfc8-758f-4a68-97b3-aac4301966f0" />


## Data Merging

Archaeological data is inherently spatial. This module utilizes `geopandas` to merge and filter the p3k14c data against external spatial and environmental datasets. Essentially, this converts the archaeological data into .gpd files, commonly used file formats for storing environmental data. 

TBD

# Example Figures

TBD





