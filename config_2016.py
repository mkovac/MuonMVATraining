import os

if 'CMSSW_BASE' in os.environ:
    cmssw_base = os.environ['CMSSW_BASE']
else:
    cmssw_base = ''

cfg = {}

cfg['ntuplizer_cfg'] = cmssw_base + '/src/MuonMVANtuplizer/Ntuplizer/test/MuonMVANtuplizer_cfg.py'

cfg['storage_site'] = 'T2_FR_GRIF_LLR'
cfg["submit_version"] = "DY_2016_10_10_2019"
#cfg["submit_version"] = "SingleMuon_2016_10_9_2019"
#cfg["submit_version"] = "ttbar_2016_27_9_2019"


# Location of CRAB output files
cfg["crab_output_dir"] = '/store/user/mkovac/Muons/%s' % cfg["submit_version"]
cfg["crab_output_dir_full"] = "/dpm/in2p3.fr/home/cms/trivcat%s" % cfg["crab_output_dir"]

# Where to store the ntuples and dmatrices
cfg["ntuple_dir"] = "/home/llr/cms/kovac/CMS/RUN_2/Data/Muons/"
cfg['dmatrix_dir'] = "/home/llr/cms/kovac/CMS/RUN_2/Data/Muons/"
cfg['out_dir'] = "out"
cfg['cmssw_dir'] = "cmssw"

# The sample used for training and evaluating the xgboost classifier.
cfg["train_sample"] = '/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8/RunIISummer16MiniAODv2-PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6_ext2-v1/MINIAODSIM'
cfg["train_sample_request_name"] = 'DYJetsToLL_M-50_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8-train'

# The sample used for Data - MC comparison.
#cfg["train_sample"] = '/SingleMuon/Run2016B-17Jul2018_ver2-v1/MINIAOD'
#cfg["train_sample_request_name"] = 'SingleMuon_2016B-train'

# The sample used for Data - MC comparison.
#cfg["train_sample"] = '/TTTo2L2Nu_TuneCUETP8M2_ttHtranche3_13TeV-powheg-pythia8/RunIISummer16MiniAODv2-PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6-v1/MINIAODSIM'
#cfg["train_sample_request_name"] = 'TTTo2L2Nu_TuneCUETP8M2_ttHtranche3_13TeV-powheg-pythia8-train'


# The fraction of this sample used for training.
cfg["train_size"] = 0.75

# The sample used for unbiased testing (performance plots).
cfg["test_sample"] = '/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8/RunIISummer16MiniAODv2-PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6_ext2-v1/MINIAODSIM'
cfg["test_sample_request_name"] = 'DYJetsToLL_M-50_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8-test'

############
# Selection
############

cfg["selection_base"] = "n_obs_int > 1 and mu_dz < 1 and mu_dxy < 0.5 and mu_sip < 4 and (is_tracker_mu or is_global_mu)"
#cfg["selection_base"] = "genNpu > 1 and (is_tracker_mu or is_global_mu)"
cfg["selection_sig"]  = "matchedToGenMu == 1"
cfg["selection_bkg"]  = "matchedToGenMu == 0 or matchedToGenMu == 3"


###########
# Variables
###########

variables_base = ["mu_eta", "glb_valid_mu_hits", "glb_chi2", "tk_valid_hits", "tk_valid_pixel_hits"]
variables_iso  = ["mu_pf_photon_iso", "mu_pf_charged_had_iso", "mu_pf_neutral_had_iso", "mu_rho"]
#variables_iso  = ["mu_pf_photon_iso", "mu_pf_charged_had_iso", "mu_pf_neutral_had_iso"]
variables_sip  = ["mu_sip", "mu_dxy", "mu_dz"]


variables_base_iso     = variables_base + variables_iso
variables_base_iso_sip = variables_base + variables_iso + variables_sip


###############################
# Configuring the training bins
###############################

# Configure the different trainings.
# For each bin, you have:
#     - cut
#     - list of variables to use

# ID + ISO
cfg["trainings"] = {}
cfg["trainings"]["Summer_16_ID_ISO"] = {}

cfg["trainings"]["Summer_16_ID_ISO"]["pT_5"] = {
        "cut": "mu_pT < 10. & abs(mu_eta) <= 2.4",
        "variables": variables_base_iso,
        "label": r'5 < $p_T$ < 10 GeV, ($|\eta| < 2.4$)',
        }
        
cfg["trainings"]["Summer_16_ID_ISO"]["pT_10"] = {
        "cut": "mu_pT >= 10. & abs(mu_eta) <= 2.4",
        "variables": variables_base_iso,
        "label": r'5 < $p_T$ > 10 GeV, ($|\eta| < 2.4$)',
        }

    
# ID + ISO + SIP
#cfg["trainings"] = {}
#cfg["trainings"]["Summer_16_ID_ISO_SIP"] = {}
#
#cfg["trainings"]["Summer_16_ID_ISO_SIP"]["pT_5"] = {
#        "cut": "mu_pT < 10. & abs(mu_eta) <= 2.4",
#        "variables": variables_base_iso_sip,
#        "label": r'5 < $p_T$ < 10 GeV, ($|\eta| < 2.4$)',
#        }
#        
#cfg["trainings"]["Summer_16_ID_ISO_SIP"]["pT_10"] = {
#        "cut": "mu_pT >= 10. & abs(mu_eta) <= 2.4",
#        "variables": variables_base_iso_sip,
#        "label": r'5 < $p_T$ > 10 GeV, ($|\eta| < 2.4$)',
#        }




################################
# Configuring the working points
################################

#import numpy as np
#wp90_target = np.loadtxt("wp90.txt", skiprows=1)
#wp80_target = np.loadtxt("wp80.txt", skiprows=1)
#
#pt_bins     = wp90_target[:,:2]
#wp90_target = wp90_target[:,2]
#wp80_target = wp80_target[:,2]

#pt_bins     = []
#wp90_target = []
#wp80_target = []

cfg["working_points"] = {}
cfg["working_points"]["Summer_16_ID_ISO_SIP"] = {}


cfg["working_points"]["Summer_16_ID_ISO_SIP"]["MVA_Muon_Summer_16_ID_ISO_SIP_HZZ"] = {
    "type":           "constant_cut_sig_eff_targets",
    "categories":     ["pT_5", "pT_10"],
#    "targets":        [0.742660202208, 0.952040486026],
#    "targets":        [0.75, 0.95],
#    "targets":        [0.80, 0.96],
#    "targets":        [0.85, 0.97],
#    "targets":        [0.90, 0.96],
#    "targets":        [0.90, 0.98],
#    "targets":        [0.95, 0.97],
#    "targets":        [0.95, 0.98],
    "match_boundary": False
    }


#####################
# CMSSW configuration
#####################

cfg["cmssw_cff"] = {}
cfg["cmssw_cff"]["Summer_16_ID_ISO_SIP"] = {}

cfg["cmssw_cff"]["Summer_16_ID_ISO_SIP"] = {
        "producer_config_name": "MVA_Muon_Summer_16_ID_ISO_SIP_producer_config",
        "file_name": "MVA_Muon_Summer_16_ID_ISO_SIP_cff.py",
        "mvaTag": "MuonSummer16IdIsoSip",
        "mvaClassName": "ElectronMVAEstimatorRun2",
        }
