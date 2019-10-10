# MuonMVATraining
The tools to train MVA Muons identifications for CMS.

## Requirements

This toolkit requires the __xgbo package__ to be installed:
* `git clone git@github.com:guitargeek/xgbo.git`
* `cd xgbo`
* `pip install --user .`

## How to train a Muon MVA ID

### Overview

The procedure splits up in a few fundamental steps:

1. Make training ntuple with CMSSW
2. Train the MVA with XGBoost
3. Determine working points
4. Generate configuration files to integrate MVA in CMSSW
5. Make validation ntuple with CMSSW
6. Draw performance plots

Only step 1 and 4 require interaction with CMSSW, the other steps can be done offline.

### Step 0 - Clone this repository and tweak the configuration

Adapt the configuration in e.g. `config_2016.py` to your needs.

### Step 1 - Making NTuples for Training

Start by setting up the CMSSW area:

* `cmsrel CMSSW_10_3_1`
* `cd CMSSW_10_3_1/src`
* `cmsenv`

Checkout the needed packages:

* `git clone https://github.com/mkovac/MuonMVANtuplizer.git`
* `git clone https://github.com/mkovac/MuonMVAReader.git`
* `git clone https://github.com/mkovac/PileUpWeight.git`

Make sure to have crab in your environment and launch the job to NTuplize the sample in e.g. config_2016.py:

* `python submit_ntuplizer_2016.py --train`

When the job is done, you should merge the crab output files to one root file.

* `python merge_ntuple_2016.py --train`

### Step 2 - Train the MVA with XGBoost

Launch the training with:

* `python training_2016.py`
