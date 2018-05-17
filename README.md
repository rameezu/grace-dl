# grace-dl

# Overview

This project provides scripts for building Deep Learning models that attempt to correct NOAH Total Water Storage (TWS) simulation models to more closely align with
'ground truth' TWS readings dervived from GRACE satellite data.

# Workflow

1. Download relavent data
    * ndviExtract.py
    * gldasnoah_*.py
2. Run model
    * convLocal.py - This script contains the main code for loading the data (gldasLocal.py) and training the models.  It also runs predictions
    on the test data and plots results.
    

# Notes

## Additional Files

* plot*.py - Provide methods for producing various plots regarding the data inputs used in the paper.
    

## Global vs Local vs Basin

Several scripts are named with either 'global' or 'local' in there names.  
The 'global' scripts are GLDAS.
The 'local' scripts are India LDAS.
The 'basin' scripts are sub parts of India LDAS.




    