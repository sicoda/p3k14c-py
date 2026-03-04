### p3k14c-py
Python package for the p3k14c global archaeological radiocarbon database
----
### Before Running

This Python code assumes your data has already passed through the p3k14c-data-scrubbing pipeline. This python code:

1. Removes records with lab codes from unknown laboratories;
2. Standardizes coordinate formats among records with location data;
3. Handles duplicate entries;
4. Cleans anomalous data; and
5. Obfuscates of precise coordinates for dates in the United States and Canada, as well as from the17 dataset, in order to protect site locations.
#








