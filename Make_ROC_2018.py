import sys
import os
from config_2018 import cfg
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.gridspec as gridspec
from os.path import join
import uproot
import pandas

plot_dir = join("Plots", cfg['submit_version'])

def load_df(h5_file):
   
   df = pandas.read_hdf(h5_file)

   df["mu_iso"] = (df["mu_pf_charged_had_iso"] + np.clip(df["mu_pf_neutral_had_iso"] + df["mu_pf_photon_iso"] - 0.5*df["mu_pu_charged_had_iso"], 0, None))/df["mu_pT"]

#   df = df.query("y > -1 & mu_dxy < 0.5 & mu_dz < 1")
   df = df.query("y > -1")
   
   return df


################
# Basic settings
################

h5_dir = cfg['out_dir'] + '/' + cfg['submit_version']

if not os.path.exists(join(plot_dir, "ROC")):
   os.makedirs(join(plot_dir, "ROC"))

# Enable or disable performance plots
ROC = True

##################
# Other parameters
##################

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
colors = colors + ['k'] * 20

for i in range(6):
   colors[6+i] = colors[i]

roccolors = prop_cycle.by_key()['color']
roccolors[2] = roccolors[0]
roccolors[3] = roccolors[1]
roccolors[0] = 'k'
roccolors[1] = '#7f7f7f'

refcolors = ['#17becf'] * 3 + ['#bcbd22'] * 3

plot_args = [
        {"linewidth": 1, "color" : colors[0] , "markersize": 2, "marker": 'o', "linestyle": '-' , 'label': 'wp90'          },
        {"linewidth": 1, "color" : colors[1] , "markersize": 2, "marker": 'o', "linestyle": '-' , 'label': 'wp80'          },
        {"linewidth": 1, "color" : colors[2] , "markersize": 2, "marker": 'o', "linestyle": '-' , 'label': 'wpLoose'       },
        {"linewidth": 1, "color" : colors[3] , "markersize": 2, "marker": 'o', "linestyle": '-' , 'label': 'wp90 w/ iso'   },
        {"linewidth": 1, "color" : colors[4] , "markersize": 2, "marker": 'o', "linestyle": '-' , 'label': 'wp80 w/ iso'   },
        {"linewidth": 1, "color" : colors[5] , "markersize": 2, "marker": 'o', "linestyle": '-' , 'label': 'wpLoose w/ iso'},
        {"linewidth": 0.5, "color" : colors[6] , "markersize": 1, "marker": '.', "linestyle": '--', 'label': ''              },
        {"linewidth": 0.5, "color" : colors[7] , "markersize": 1, "marker": '.', "linestyle": '--', 'label': ''              },
        {"linewidth": 0.5, "color" : colors[8] , "markersize": 1, "marker": '.', "linestyle": '--', 'label': ''              },
        {"linewidth": 0.5, "color" : colors[9] , "markersize": 1, "marker": '.', "linestyle": '--', 'label': ''              },
        {"linewidth": 0.5, "color" : colors[10], "markersize": 1, "marker": '.', "linestyle": '--', 'label': ''              },
        {"linewidth": 0.5, "color" : colors[11], "markersize": 1, "marker": '.', "linestyle": '--', 'label': ''              },
        ]

plot_args_bkg = []

for i in range(len(plot_args)):
   plot_args_bkg.append(plot_args[i].copy())
   plot_args_bkg[i]["label"] = ''

plot_args = [plot_args, plot_args_bkg]

roc_curves = [
              ("2018", "bdt_score_default", "ID+ISO Muon XGBoost MVA"),
             ]

roc_plot_args = {
                 'markeredgewidth': 0,
                 'linewidth': 2,
                }


##################
# Helper functions
##################

def create_axes(yunits=4):
   fig = plt.figure(figsize=(6.4, 4.8))
   gs = gridspec.GridSpec(yunits, 1)
   ax1 = plt.subplot(gs[:2, :])
   ax2 = plt.subplot(gs[2:, :])
   axarr = [ax1, ax2]

   gs.update(wspace=0.025, hspace=0.075)

   plt.setp(ax1.get_xticklabels(), visible = False)

   ax1.grid()
   ax2.grid()

   return ax1, ax2, axarr


############
# ROC Curves
############

if ROC:

   print("Making ROC curves")

   for ptrange in ["5", "10"]:
      
      for location in ["pT"]:

         print("processing {0} {1}...".format(location, ptrange))

         h5_file = h5_dir + "/Spring_18_ID_ISO/" + location + "_" + ptrange + '/pt_eta_score.h5'
            
         df = load_df(h5_file)
         
         sig = df.query("y == 1")
         bkg = df.query("y == 0")
         
         n_sig = len(sig)
         n_bkg = len(bkg)
         
         sig_passing_ID_cut = sig.query("is_pf_mu == 1")
         sig_passing_ID_ISO_cut = sig_passing_ID_cut.query("mu_iso < 0.35")
#         sig_passing_ID_ISO_SIP_cut = sig_passing_ID_ISO_cut.query("mu_sip < 4")
#         sig_passing_ID_ISO_SIP_cut = sig.query("mu_sip < 4 & mu_dxy < 0.5 & mu_dz < 1")

#         sig_passing_ID_ISO_cut_sip_50 = sig_passing_ID_ISO_cut.query("mu_sip > 50")
#         sig_passing_sip_50 = sig.query("mu_sip > 50")



         bkg_passing_ID_cut = bkg.query("is_pf_mu == 1")
         bkg_passing_ID_ISO_cut = bkg_passing_ID_cut.query("mu_iso < 0.35")
#         bkg_passing_ID_ISO_SIP_cut = bkg_passing_ID_ISO_cut.query("mu_sip < 4")
#         bkg_passing_ID_ISO_SIP_cut = bkg.query("mu_sip < 4 & mu_dxy < 0.5 & mu_dz < 1")

#         bkg_passing_ID_ISO_cut_sip_50 = bkg_passing_ID_ISO_cut.query("mu_sip > 50")
#         bkg_passing_sip_50 = bkg.query("mu_sip > 50")



         
#         print "Signal"
#         print len(sig_passing_ID_ISO_cut_sip_50) * 1./len(sig_passing_ID_ISO_cut)*100
#         print len(sig_passing_sip_50) * 1./len(sig)*100
#         
#         print "Background"
#         print len(bkg_passing_ID_ISO_cut_sip_50) * 1./len(bkg_passing_ID_ISO_cut)*100
#         print len(bkg_passing_sip_50) * 1./len(bkg)*100
#         
#         
#         sig_barrel = sig.query("mu_eta < 1.2")
#         print sig_barrel["mu_sip"]
#         
#         sig_endcap = sig.query("mu_eta > 1.2")
#         print sig_endcap["mu_sip"]


         sig_id_eff = len(sig_passing_ID_cut) * 1./n_sig * 100
         bkg_id_eff = len(bkg_passing_ID_cut) * 1./n_bkg * 100

         sig_iso_eff = len(sig_passing_ID_ISO_cut) * 1./len(sig_passing_ID_cut) * 100
         bkg_iso_eff = len(bkg_passing_ID_ISO_cut) * 1./len(bkg_passing_ID_cut) * 100

#         sig_sip_eff = len(sig_passing_ID_ISO_SIP_cut) * 1./len(sig_passing_ID_ISO_cut) * 100
#         bkg_sip_eff = len(bkg_passing_ID_ISO_SIP_cut) * 1./len(bkg_passing_ID_ISO_cut) * 100
         
         sig_eff = len(sig_passing_ID_ISO_cut) * 1./n_sig * 100
         bkg_eff = len(bkg_passing_ID_ISO_cut) * 1./n_bkg * 100
         
         
         print('sig_id_eff = {0}'.format(sig_id_eff))
         print('bkg_id_eff = {0}'.format(bkg_id_eff))
         
         print('sig_iso_eff = {0}'.format(sig_iso_eff))
         print('bkg_iso_eff = {0}'.format(bkg_iso_eff))
         
#         print('sig_sip_eff = {0}'.format(sig_sip_eff))
#         print('bkg_sip_eff = {0}'.format(bkg_sip_eff))
         
         print('sig_eff = {0}'.format(sig_eff))
         print('bkg_eff = {0}'.format(bkg_eff))
         

         ax1, ax2, axes = create_axes(yunits=3)

         xmin = 60

         yref, xref, _ = metrics.roc_curve(df["y"] == 1, df["bdt_score_default"])
         xref = xref * 100
         yref = yref * 100

         k = 0
         for yr, cl, lbl in roc_curves:

            y, x, _ = metrics.roc_curve(df["y"] == 1, df[cl])
            x = x * 100
            y = y * 100

            sel = x > xmin

            ax1.semilogy(x[sel], y[sel], color=roccolors[k], label=lbl, **roc_plot_args)
#            ax1.semilogy(x[sel], y[sel], color=roccolors[k], **roc_plot_args)
            ax1.semilogy(sig_eff, bkg_eff, color="#AE3135", marker="o",  markersize=5)
            ax2.plot(x[sel], y[sel]/np.interp(x[sel], xref, yref), color=roccolors[k], **roc_plot_args)
            ax2.plot(sig_eff, (bkg_eff/np.interp(sig_eff, xref, yref))**(-1), color="#AE3135", marker="o",  markersize=5)

            k = k + 1

         # Styling the plot
         ax1.set_ylabel(r'Background efficiency [%]')

         ax2.set_xlabel(r'Signal efficiency [%]')
         ax2.set_ylabel(r'Ratio')

         ax1.set_ylim(0.101, 100)
         ax2.set_ylim(0.201, 1.09)

         ax1.legend(loc="upper left", ncol=1)

         ax1.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))

         plt.savefig(join(plot_dir, "ROC/2018_{0}_{1}.pdf".format(location, ptrange)), bbox_inches='tight')
#         plt.savefig(join(plot_dir, "ROC/2018_{0}_{1}.eps".format(location, ptrange)), bbox_inches='tight')
         plt.savefig(join(plot_dir, "ROC/2018_{0}_{1}.png".format(location, ptrange)), bbox_inches='tight')
#         os.system("convert -density 150 -quality 100 " + join(plot_dir, "roc/2017_{0}_{1}.eps".format(location, ptrange)) + " "
#                                                        + join(plot_dir, "roc/2017_{0}_{1}.png".format(location, ptrange)))

         plt.close()