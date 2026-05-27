# p3k14c-py

Python package for the p3k14c global archaeological radiocarbon database. (Currently in Progress, come back later!)

[View the interactive p3k14c database map](./interactive_spatial_map.html)

<figure>
  <img width="830" height="450" alt="image" src="https://github.com/user-attachments/assets/67ce4f41-e512-4b5f-bf08-a05a8657e4f9" />
  <figcaption align="center"><b>Figure 1:</b> Map showing all available datapoints in the p3k14c database.</figcaption>
</figure>

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

This cleaning script is an essential step for quality control and later computational analyses. This script is adapted from  [p3k14c-data-scrubbing](https://github.com/people3k/p3k14c-data-scrubbing) to work using Python 3.12.10.

**Functionality**: The cleaning script automatically cleans the dataset via these tasks:

1. Lab-code validation via Labs.csv (typo correction included)
2. LabID quality checks: must contain a numeral, no '?', no corrupted Unicode
3. LabID standardisation: strip punctuation, uppercase, insert dash
4. Coordinate-format conversion (deg/min/sec, Solheim Northing/Easting)
5. SiteName / SiteID whitespace stripping
6. Duplicate removal (LabID-exact; first-occurrence fallback otherwise)
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

**Functionality**: The calibration script automatically maps each date to the correct calibration curve (`IntCal20` for the Northern Hemisphere, `SHCal20` for the Southern Hemisphere) based on the sample's latitude. It extracts key numerical boundaries, such as the median calendar age and 95% confidence intervals, and formats them neatly into a `pandas` DataFrame for later analysis. This script outputs a `p3k14c_pristine_dates.csv` file, which will be used in the following statistical analyses. 

1. Curve selection: for every row, it checks the Lat column. Any sample at or above the equator gets `intcal20`; anything below gets `shcal20`. This is the standard approach for global datasets.
2. Calibration: calls R(age, error, lab_id).calibrate(curve) from `IOSACal`, exactly as the official library intends.
3. Extracting statistics: calls .quantiles() on each CalAge result, which returns the median (quants[50]) and the 1σ / 2σ confidence intervals (quants[68], quants[95]).
4. Failure handling: rows with missing Age, Error, or Lat, or any row where IOSACal throws an exception, are written to calibration_failures.csv rather than crashing the whole run.

<figure>
  <img width="4753" height="1752" alt="image" src="https://github.com/user-attachments/assets/cff72b16-2b08-4856-bf5a-1e315571e249" />
  <figcaption align="center"><b>Figure 2:</b> The calibrated (blue) and uncalibrated (red) radiocarbon data in the p3k14c dataset following calibration via IOSACal.</figcaption>
</figure>
  
**Case Study**: Çatalhöyük
    
<figure>
  <img width="4753" height="1752" alt="image" src="https://github.com/user-attachments/assets/433fba98-494a-42f7-bd44-19cd2d7f99eb" />
  <figcaption align="center"><b>Figure 3:</b> The calibrated (blue) and uncalibrated (red) Çatalhöyük radiocarbon data following calibration via IOSACal.</figcaption>
</figure>

______________________________________________________
**These Plots Show:**

1. Calibration Flattens and Spreads Time
  - In Figures 2 & 3, the raw 14C ages (red) are naturally clustered and spiky due to plateaus in the calibration curve (where multiple calendar years produce the same radiocarbon age).
  - Calibration (blue) smooths these spikes out, distributing the radiocarbon data accurately across the timeline. 
2. The "Hallstatt Plateau" Effect
  - In Figure 3, note how the blue calibrated radiocarbon data is wider and lower than the red uncalibrated peak. This visually demonstrates why calibration is necessary: a single, precise radiocarbon measurement often corresponds to a broad range of true calendar years.
    
> 💡 **Skip the setup!** I've attached the final cleaned and calibrated dataset in this repository (`p3k14c_pristine_dates.csv`). You can bypass the finicky cleaning and calibrating process entirely and jump straight to the data analysis.

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

SPD's can be used on calibrated radiocarbon data to estimate demographic fluctuations over time as a proxy for human activity. Essentially, the frequency of datable anthropogenic carbon recovered from archaeological contexts serves as a direct proxy for past fluctuations in human population density and associated settlement activity.

**Functionality**: This script combines several methodologies from paleodemography:

1. Chronometric Hygiene: Filters out problematic materials (old-wood/marine effects), drops large-error dates ([Reimer et al. 2020](https://www.cambridge.org/core/journals/radiocarbon/article/intcal20-northern-hemisphere-radiocarbon-age-calibration-curve-055-cal-kbp/83257B63DC3AF9CFA6243F59D7503EFF)).
2. Spatial-Temoral Binning: Controls oversampling biases from single archaeological phases ([Timpson et al. 2014](https://www.researchgate.net/publication/265421202_Reconstructing_regional_population_fluctuations_in_the_European_Neolithic_using_radiocarbon_dates_A_new_case-study_using_an_improved_method)).
3. Taphonomic Correction: Applies power-function corrections to account for the natural decay and loss of organic material over time ([Surovell et al. 2009](https://www.sciencedirect.com/science/article/pii/S0305440309001204); [Bluhm & Surovell 2018](https://www.researchgate.net/publication/327320188_Validation_of_a_global_model_of_taphonomic_bias_using_geologic_radiocarbon_ages)).
4. Null Hypothesis Significance Testing (NHST): Uses 5,000-iteration (adjustable) Monte Carlo simulations to test the empirical SPD against exponential and logistic null models, highlighting periods of statistically significant population deviation ([Timpson et al. 2014](https://www.researchgate.net/publication/265421202_Reconstructing_regional_population_fluctuations_in_the_European_Neolithic_using_radiocarbon_dates_A_new_case-study_using_an_improved_method); [Crema et al. 2017](https://www.sciencedirect.com/science/article/pii/S0305440317301310); [Weninger et al. 2015](https://www.tandfonline.com/doi/full/10.1080/00438243.2015.1064022); [Bettinger et al. 2016](https://www.pnas.org/doi/full/10.1073/pnas.1523806113); [Shennan et al. 2013](https://www.nature.com/articles/ncomms3486)).
5. Continuous Piecewise Linear (CPL) Modelling: Uses differential evolution and Bayesian Information Criterion (BIC) to identify optimal "hinge points" that represent major regime shifts in population growth and decline ([McLaughlin 2019](https://link.springer.com/article/10.1007/s10816-018-9381-3); [Edinborough et al. 2017](https://www.pnas.org/doi/full/10.1073/pnas.1713012114)).

**Case Study**: Çatalhöyük

<figure>
  <img width="5055" height="3326" alt="image" src="https://github.com/user-attachments/assets/7e202d6b-47aa-4348-8af7-3a2a82ddc828" />
  <figcaption align="center"><b>Figure 4:</b> A multi-panel analysis of population trends using Summed Probability Distributions (SPD) of radiocarbon dates (Monte Carlo N=5000). <b>(A)</b> Comparison of the raw empirical SPD (solid blue line) and the taphonomically corrected SPD (dashed red line), which accounts for time-dependent preservation biases in the archaeological record. <b>(B)</b> Null Hypothesis Significance Testing (NHST) comparing the empirical SPD against an Exponential null growth model. The grey shaded area represents the 95% Monte Carlo confidence envelope. Periods where the empirical data significantly exceed the envelope (red) indicate population booms, while periods falling below (green) indicate significant demographic decline. <b>(C)</b> NHST comparing the empirical SPD against a Logistic null growth model; an alternative baseline for population dynamics. <b>(D)</b> A Continuous Piecewise Linear (CPL) model best fit applied to the empirical SPD, identifying  major demographic turning points (hinge points) at 9220, 8477, 8277, and 7921 Cal BP to characterize the primary phases of growth, decline, and recovery at the site.</figcaption>
  
______________________________________________________
**These Plots Show:**

1. Initial Settlement and Growth (9220 - 8477 Cal BP)
  - Plot D shows an initial hinge point at 9220 Cal BP, marking the beginning of an upward demographic trend
  - Plots B & C show this growth tracking relatively closely with expected baselines, staying within the null envelope.
  - Aligns with the settlement of the East mound.

3. Sudden Population Decline (8477 - 8277 Cal BP)
  - Plot D shows a sharp drop at 8277 Cal BP, and Plots B & C show a red "Below Null" zone shortly after this period.
  - This aligns with the 8.2 ky climate event and the "Late" phase, bringing about social fragmentation.

4. Population Boom (~8000 - 7500 Cal BP)
  - Plot D shows a hinge point at 7921 Cal BP, showing a population recovery.
  - Plot A spikes around this time.
  - Plots B & C show massive red areas indicating "Above Null".
  - This aligns with the occupation of the West mound; there may be sampling bias making this boom seem more prominent.

5. Final Decline (post 7500 Cal BP)
  - Across all plots, a sharp decline is seen after 7500 Cal BP, marking the site's final abandonment.

## Data Merging
By merging the results from the SPD script with paleoclimate data, it can be elucidated whether or not a demographic trend was influenced by an environmental change. By overlaying the demographic proxy with smoothed environmental proxies, the script highlights distinct periods of environmental stress and then evaluates whether corresponding demographic drops (identified via Monte Carlo significance testing) represent true societal collapses or resilient recoveries.

**Functionality**:
  - Automated Data Retrieval: Fetches local environmental proxy data (e.g., stable isotopes, pollen) from PANGAEA and Neotoma databases based on proximity to the archaeological site.
  - Data Harmonization: Detrends and Z-scores disparate environmental proxies, aligning them temporally with the IntCal20-calibrated demographic SPD.
  - Anomaly Detection: Identifies periods of extreme environmental stress (anomalies) using customizable thresholds.
  - Resilience & Resistance Metrics: Calculates the demographic resistance to an environmental event, and fits an exponential recovery curve to quantify subsequent human resilience.
  - Visualization: Generates a comprehensive, multi-panel dashboard comparing demographic and environmental Z-scores over time.

**Case Study**: Çatalhöyük

<figure>
  <img width="4160" height="5296" alt="image" src="https://github.com/user-attachments/assets/05f713bc-3d0e-4f38-ae72-7115d6bc4f5f" />
  <figcaption align="center"><b>Figure 5:</b> A multi-panel analysis assessing the demographic response to climate anomalies (shown in the yellow band). <b>(A)</b> Summed Probability Distribution (SPD) of radiocarbon dates serving as a proxy for archaeological activity. <b>(B)</b> Z-scored environmental proxy highlighting a severe climate anomaly (dropping below the -1.0 threshold) between ~8400 - 8150 Cal BP (yellow band). <b>(C)</b> Z-scored demographic proxy demonstrating a corresponding population decline during the climate event. <b>(D)</b> Overlay of demographic and environmental trends, illustrating the synchronized decline and the subsequent demographic boom. <b>(E)</b> A detailed view of the climate anomaly, quantifying the site's demographic resistance and exponential resilience recovery rates. <b>(F)</b> Monte Carlo significance test comparing the empirical SPD against a 95% simulated null-growth envelope; red shaded regions denote statistically significant demographic troughs, confirming the population crash during the climate event was not a random fluctuation.</figcaption>
</figure>

______________________________________________________
**These Plots Show:**

1. Environmental Shock (8400 - 8150 Cal BP)
  - The Environmental Proxy Plot (B) shows a sharp decline in the environmental Z-score reaching well below the -1.0 anomaly threshold
  - This aligns perfectly with the 8.2 ky event (abrupt global cooling and aridification)

2. Human - Environment Interaction
  - The Archaeological Activity Proxy Plot (A) and the Demographic Proxy Plot (C) both exhibit a sharp decline in synch with the environmental change.
  - The Resistance & Resilience Detail Pot (E) and the Monte Carlo Significance Test (F) confirm that this was not a random fluctuation, meaning the climate event forced a genuine demographic change.

3. Shock Aftermath (~8000 - 7500 Cal BP)
  - The Overlay Plot (D) shows that as soon as the climate recovers, the population follows almost immediately.
  - By 8000 Cal BP, the archaeological activity surpasses the previous levels, exhibiting high societal resilience (noted as 0.274/100yr in the detail plot).
  - This shows that the population at Çatalhöyük adapted, reorganized, and recovered once the climate stabilized.
  - From the Ice Core data, it is unclear whether the final abandonment of the site was climate-motivated. 


## Density Analysis

This analysis shows the geographical distribution of unweighted sites across a region (Plot a) and the weighted C14 dates at those sites (Plot b). This script allows the user to select a specific temporal range and region (one of the seven continents) or create a custom region (via latitude and longitude) for analysis. 

Functionality: Uses `scipy.stats.gaussian_kde` to create continuous Kernel Density Estimation (KDE) surfaces for recorded sites vs. dated sites. Provides a visual of spatial sampling biases across the globe. Here is an example for North America:

<figure>
  <img width="700" height="1000" alt="image" src="https://github.com/user-attachments/assets/6168e0ee-4df1-4b6f-9cd3-79fff1d4bec3" />
 <figcaption align="center"><b>Figure 5:</b> here.</figcaption>
</figure>






