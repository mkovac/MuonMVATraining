import sys
import os
from config_2016 import cfg
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.gridspec as gridspec
from os.path import join
import uproot


cuts = {
    "EB1": "abs(scl_eta) < 0.800",
    "EB2": "abs(scl_eta) >= 0.800 & abs(scl_eta) < 1.479",
    "EE" : "abs(scl_eta) >= 1.479",
    "5"  : "ele_pt < 10.",
    "10" : "ele_pt >= 10",
    }


def get_category(location, ptrange):

    training_bin = -1

    if   location == 'EB1' and ptrange == '5' : training_bin = 0
    elif location == 'EB2' and ptrange == '5' : training_bin = 1
    elif location == 'EE'  and ptrange == '5' : training_bin = 2
    elif location == 'EB1' and ptrange == '10': training_bin = 3
    elif location == 'EB2' and ptrange == '10': training_bin = 4
    elif location == 'EE'  and ptrange == '10': training_bin = 5

    return training_bin


# Inspired by https://root.cern.ch/doc/master/TGraphAsymmErrors_8cxx_source.html#l00573
def eff_with_err(p, t, cl=0.638, percent=False, alpha=1, beta=1):
    if not hasattr(p, "__len__"):
        p = [p]
    if not hasattr(t, "__len__"):
        t = [t]

    n = len(p)

    eff = np.zeros(n)
    lower = np.zeros(n)
    upper = np.zeros(n)

    for i in range(n):
        eff[i] = p[i]/t[i]
        aa = p[i] + alpha
        bb = t[i] - p[i] + beta
        # lower[i] = TEfficiency.BetaCentralInterval(cl, aa, bb, False)
        # upper[i] = TEfficiency.BetaCentralInterval(cl, aa, bb, True)
        lower[i] = 0
        upper[i] = 0

    if percent:
        return 100 * eff, 100 * lower, 100 * upper
    else:
        return eff, lower, upper

#================#
# Read root file #
#================#

def load_df(rootfile, branches, entrystop=None):

    ntuple_dir = join(cfg['ntuple_dir'], cfg['submit_version'])
    ntuple_file = join(ntuple_dir, 'test.root')
    tree = uproot.open(ntuple_file)["ntuplizer/tree"]

    df = tree.pandas.df(set(branches + ["mu_pT", "mu_eta", "is_tracker_mu", "is_global_mu", "matchedToGenMu", "is_pf_mu", "mu_dz", "mu_dxy", "genNpu"]), entrystop=entrystop)

    df = df.query(cfg["selection_base"])
    df.eval('y = ({0}) + 2 * ({1}) - 1'.format(cfg["selection_bkg"], cfg["selection_sig"]), inplace=True)
    
    df["mu_iso"] = (df["mu_pf_charged_had_iso"] + np.clip(df["mu_pf_neutral_had_iso"] + df["mu_pf_photon_iso"] - 0.5*df["mu_pu_charged_had_iso"], 0, None))/df["mu_pT"]
    
    df = df.query("y > -1")

    return df


#=============#
# Set folders #
#=============#

plot_dir   = join("Plots", cfg['submit_version'])
ntuple_dir = cfg['ntuple_dir'] + '/' + cfg['submit_version']
rootfile   = ntuple_dir + '/test.root'

if not os.path.exists(join(plot_dir, "roc")):
    os.makedirs(join(plot_dir, "roc"))
    
if not os.path.exists(join(plot_dir, "turnon")):
    os.makedirs(join(plot_dir, "turnon"))

if not os.path.exists(join(plot_dir, "etaeff")):
    os.makedirs(join(plot_dir, "etaeff"))

if not os.path.exists(join(plot_dir, "nvtx")):
    os.makedirs(join(plot_dir, "nvtx"))


#=====================================#
# Enable or disable performance plots #
#=====================================#

ROC    = True
turnon = False
etaeff = False
nvtx   = False

#nmax = 200000
#nmax = 2000000
nmax = None


#==================#
# Other parameters #
#==================#

ylim_sig = 0, 100
ylim_bkg = 0.3001, 50

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
colors = colors + ['k'] * 20

for i in range(6):
    colors[6+i] = colors[i]

roccolors = prop_cycle.by_key()['color']
roccolors[2] = "#ff7429"
roccolors[3] = "#79af67"
roccolors[0] = 'k'
roccolors[1] = '#0f4da5'

refcolors = ['#17becf'] * 3 + ['#bcbd22'] * 3

wps = ["Summer16IDISOHZZ", "Spring16HZZV1wpLoose"]

plot_args = [
        {"linewidth": 1, "color" : colors[0] , "markersize": 2, "marker": 'o', "linestyle": '-' , 'label': '2016 ID+ISO'  },
        {"linewidth": 1, "color" : colors[1] , "markersize": 2, "marker": 'o', "linestyle": '-' , 'label': '2016 ID only'},
        {"linewidth": 1, "color" : colors[2] , "markersize": 2, "marker": 'o', "linestyle": '-' , 'label': '2017 ID+ISO V2'},
        ]

plot_args_bkg = []

for i in range(len(plot_args)):
    plot_args_bkg.append(plot_args[i].copy())
    plot_args_bkg[i]["label"] = ''

plot_args = [plot_args, plot_args_bkg]

roc_curves = [
              ("2016", "score", "ID+ISO+SIP Muon XGBoost MVA")           
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

    plt.setp(ax1.get_xticklabels(), visible=False)

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

            print("[INFO] Processing {0} {1}...".format(location, ptrange))

            branches = ["score", "mu_sip", "mu_pf_charged_had_iso", "mu_pf_neutral_had_iso", "mu_pf_photon_iso", "mu_pu_charged_had_iso"]
            df = load_df(rootfile, branches, entrystop=nmax)
            df = df.query(cfg["trainings"]["Summer_16_ID_ISO_SIP"]["_".join([location, ptrange])]["cut"])


            sig = df.query("y == 1")
            bkg = df.query("y == 0")
            
            n_sig = len(sig)
            n_bkg = len(bkg)
	         
            sig_passing_ID_cut = sig.query("is_pf_mu == 1")
            sig_passing_ID_ISO_cut = sig_passing_ID_cut.query("mu_iso < 0.35")
            sig_passing_ID_ISO_SIP_cut = sig_passing_ID_ISO_cut.query("mu_sip < 4")	
            	         
            bkg_passing_ID_cut = bkg.query("is_pf_mu == 1")
            bkg_passing_ID_ISO_cut = bkg_passing_ID_cut.query("mu_iso < 0.35")
            bkg_passing_ID_ISO_SIP_cut = bkg_passing_ID_ISO_cut.query("mu_sip < 4")
            	         
            sig_id_eff = len(sig_passing_ID_cut) * 1./n_sig * 100
            bkg_id_eff = len(bkg_passing_ID_cut) * 1./n_bkg * 100
            	         
            sig_iso_eff = len(sig_passing_ID_ISO_cut) * 1./len(sig_passing_ID_cut) * 100
            bkg_iso_eff = len(bkg_passing_ID_ISO_cut) * 1./len(bkg_passing_ID_cut) * 100
            	         
            sig_sip_eff = len(sig_passing_ID_ISO_SIP_cut) * 1./len(sig_passing_ID_ISO_cut) * 100
            bkg_sip_eff = len(bkg_passing_ID_ISO_SIP_cut) * 1./len(bkg_passing_ID_ISO_cut) * 100
            	         
            sig_eff = len(sig_passing_ID_ISO_SIP_cut) * 1./n_sig * 100
            bkg_eff = len(bkg_passing_ID_ISO_SIP_cut) * 1./n_bkg * 100
            	        
            print('sig_id_eff = {0}'.format(sig_id_eff))
            print('bkg_id_eff = {0}'.format(bkg_id_eff))
            	         
            print('sig_iso_eff = {0}'.format(sig_iso_eff))
            print('bkg_iso_eff = {0}'.format(bkg_iso_eff))
            	         
            print('sig_sip_eff = {0}'.format(sig_sip_eff))
            print('bkg_sip_eff = {0}'.format(bkg_sip_eff))
            	         
            print('sig_eff = {0}'.format(sig_eff))
            print('bkg_eff = {0}'.format(bkg_eff))



            ax1, ax2, axes = create_axes(yunits=3)

            xmin = 60

            yref, xref, _ = metrics.roc_curve(df["y"] == 1, df["score"])
            xref = xref * 100
            yref = yref * 100

            k = 0
            for yr, cl, lbl in roc_curves:

               y, x, _ = metrics.roc_curve(df["y"] == 1, df[cl])
               x = x * 100
               y = y * 100

               sel = x > xmin

               ax1.semilogy(x[sel], y[sel], color=roccolors[k], label=lbl, **roc_plot_args)
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

            ax1.legend(loc="upper left", ncol=2)

            ax1.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))

            plt.savefig(join(plot_dir, "ROC/2016_{0}_{1}_test.pdf".format(location, ptrange)), bbox_inches='tight')
            plt.savefig(join(plot_dir, "ROC/2016_{0}_{1}_test.png".format(location, ptrange)), bbox_inches='tight')
#            plt.savefig(join(plot_dir, "roc/2017_{0}_{1}.png".format(location, ptrange)), bbox_inches='tight')
#            os.system("convert -density 150 -quality 100 " + join(plot_dir, "roc/2016_{0}_{1}.eps".format(location, ptrange)) + " "
#                                                           + join(plot_dir, "roc/2016_{0}_{1}.png".format(location, ptrange)))

            plt.close()

#########
# turn-on
#########

if turnon:

    print("Making turn-on curves")

    for location in ['EB1', 'EB2', 'EE']:

        print("processing {0}...".format(location))

        branches = wps + ["ele_pt"]
        df = load_df(rootfile, branches, entrystop=nmax)
        df = df.query(cuts[location])
        sig = df.query("y == 1")
        bkg = df.query("y == 0")

        tmp = np.concatenate([np.linspace(5, 10, 15), np.exp(np.linspace(np.log(11), np.log(40), 20)), np.array([43.3, 52, 70, 150, 250])])
        pt_bins = np.vstack([tmp[:-1], tmp[1:]]).T

        nbins = len(pt_bins)
        pt = np.array([p[0] + (p[1] - p[0])/2 for p in pt_bins])

        #for wplabel in ["wp90", "wp80", "HZZ"]:
        for wplabel in ["wpAll"]:

            print("Processing working point {0}...".format(wplabel))

            ax1, ax2, axes = create_axes()

            for k, wp in enumerate(wps):

                for l, class_ in enumerate([sig, bkg]):
                    x, xlow, xup = np.zeros(nbins), np.zeros(nbins), np.zeros(nbins)

                    for i in range(nbins):
                        a, b = pt_bins[i]

                        sel = np.logical_and(class_["ele_pt"] >= a, class_["ele_pt"] < b)

                        num = np.sum(np.logical_and(class_[wp] == 1, sel)) * 1.
                        den = np.sum(sel)

                        x[i], xlow[i], xup[i]  = eff_with_err(num, den, percent=True)

                    axes[l].semilogx(pt[pt > 10], x[pt > 10], **plot_args[l][k])
                    axes[l].semilogx(pt[pt < 10], x[pt < 10], **plot_args[1][k])

                    if k < 6:
                        axes[l].fill_between(pt[pt > 10], xlow[pt > 10], xup[pt > 10], alpha=0.5, facecolor=plot_args[l][k]['color'])
                        axes[l].fill_between(pt[pt < 10], xlow[pt < 10], xup[pt < 10], alpha=0.5, facecolor=plot_args[l][k]['color'])

            ax2.set_yscale('log', nonposy='clip')

            ax1.grid(axis='x', which='minor')
            ax2.grid(axis='x', which='minor')

            plt.tick_params(axis='x', which='minor')

            ax2.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
            ax2.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))

            ax1.set_ylabel(r'Signal eff. [%]')
            ax2.set_ylabel(r'Backgorund eff. [%]')
            ax2.set_xlabel(r'$p_T$ [GeV]')

            ax1.legend(loc="lower right", ncol=2)

            # Plot the transition between training bins
            ax1.set_xlim(5, 200)
            ax2.set_xlim(5, 200)

            ax1.set_ylim(ylim_sig)
            ax2.set_ylim(ylim_bkg)

            ax1.plot([10, 10], ax1.get_ylim(), 'k--')
            ax2.plot([10, 10], ax2.get_ylim(), 'k--')

            plt.savefig(join(plot_dir, "turnon/2017_{0}_{1}.pdf".format(location, wplabel)), bbox_inches='tight')
            plt.savefig(join(plot_dir, "turnon/2017_{0}_{1}.png".format(location, wplabel)), bbox_inches='tight')

            plt.close()
#####
# eta
#####

if etaeff:

    print("Making eta curves")

    for ptrange in ["5", "10"]:

        print("processing {0}...".format(ptrange))

        branches = wps + ["scl_eta"]
        df = load_df(rootfile, branches, entrystop=nmax)
        df = df.query(cuts[ptrange])
        sig = df.query("y == 1")
        bkg = df.query("y == 0")

        #tmp = np.linspace(-2.5, 2.5, 201)
        tmp = np.linspace(-2.5, 2.5, 101)
        bins = np.vstack([tmp[:-1], tmp[1:]]).T

        nbins = len(bins)
        eta = np.array([p[0] + (p[1] - p[0])/2 for p in bins])

        for wplabel in ["wpAll"]:

            print("Processing working point {0}...".format(wplabel))

            ax1, ax2, axes = create_axes()

            for k, wp in enumerate(wps):

                print("Processing {0}".format(wp))

                for l, class_ in enumerate([sig, bkg]):

                    x, xlow, xup = np.zeros(eta.shape), np.zeros(eta.shape), np.zeros(eta.shape)

                    for i in range(nbins):
                        a, b = bins[i]

                        sel = np.logical_and(class_["scl_eta"] >= a, class_["scl_eta"] < b)
                        num = np.sum(np.logical_and(class_[wp] == 1, sel)) * 1.
                        den = np.sum(sel)

                        x[i], xlow[i], xup[i]  = eff_with_err(num, den, percent=True)

                    axes[l].plot(eta, x, **plot_args[l][k])
                    if k < 6:
                        axes[l].fill_between(eta, xlow, xup, alpha=0.5, facecolor=plot_args[l][k]['color'])

            ax1.set_ylim(ylim_sig)
            ax2.set_ylim(ylim_bkg)

            for ax in [ax1, ax2]:
                ax.plot(2*[0.8], ax.get_ylim(), 'k--', linewidth=1)
                ax.plot(2*[1.5], ax.get_ylim(), 'k--', linewidth=1)
                ax.plot(2*[-0.8], ax.get_ylim(), 'k--', linewidth=1)
                ax.plot(2*[-1.5], ax.get_ylim(), 'k--', linewidth=1)

                ax.grid(axis='x', which='minor')

            ax2.set_yscale('log', nonposy='clip')

            ax1.set_ylabel(r'Signal eff. [%]')
            ax2.set_ylabel(r'Backgorund eff. [%]')
            ax2.set_xlabel(r'$\eta$')

            ax1.legend(loc="lower center", ncol=3)

            ax2.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))

            # if ptrange == '5':
                # ax1.set_title(r'Efficiencies vs $\eta$ for Fall17 V2 - 5 < $p_T$ < 10 GeV  ')
            # else:
                # ax1.set_title(r'Efficiencies vs $\eta$ for Fall17 V2 - $p_T$ > 10 GeV ')

            plt.savefig(join(plot_dir, "etaeff/{0}_{1}.pdf".format(ptrange, wplabel)), bbox_inches='tight')
            plt.savefig(join(plot_dir, "etaeff/{0}_{1}.png".format(ptrange, wplabel)), bbox_inches='tight')

            plt.close()

######
# nvtx
######

if nvtx:

    print("Making pileup curves")

    for ptrange in ["5", "10"]:

        for location in ['EB1', 'EB2', 'EE']:

            print("processing {0} {1}...".format(location, ptrange))

            branches = wps + ["genNpu"]
            df = load_df(rootfile, branches, entrystop=nmax)
            df = df.query(cfg["trainings"]["Fall17NoIsoV2"]["_".join([location, ptrange])]["cut"])
            sig = df.query("y == 1")
            bkg = df.query("y == 0")

            for wplabel in ["wpAll"]:

                print("Processing working point {0}...".format(wplabel))

                ax1, ax2, axes = create_axes()

                nvtx_step = 3

                X = np.array(range(10, 81))[::nvtx_step]

                for k, wp in enumerate(wps):

                    bins = X + nvtx_step // 2

                    for l, class_ in enumerate([sig, bkg]):
                        x, xlow, xup = np.zeros(X.shape), np.zeros(X.shape), np.zeros(X.shape)

                        for i, xi in enumerate(X):
                            sel = np.logical_and(class_["genNpu"] >= xi, class_["genNpu"] < xi + nvtx_step)

                            num = np.sum(np.logical_and(class_[wp], sel)) * 1.

                            x[i], xlow[i], xup[i]  = eff_with_err(num, np.sum(sel), percent=True)

                        axes[l].plot(bins, x, **plot_args[l][k])
                        if k < 6:
                            axes[l].fill_between(bins, xlow, xup, alpha=0.5, facecolor=plot_args[l][k]['color'])

                ax1.set_ylabel(r'Signal eff. [%]')
                ax2.set_ylabel(r'Backgorund eff. [%]')
                ax2.set_xlabel(r'True number of vertecies')

                ax1.legend(loc="lower left", ncol=3)

                ax2.set_yscale('log', nonposy='clip')

                ax1.set_ylim(ylim_sig)
                ax2.set_ylim(ylim_bkg)

                ax2.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))

                plt.savefig(join(plot_dir, "nvtx/2017_{0}_{1}_{2}.pdf".format(location, ptrange, wplabel)), bbox_inches='tight')
                plt.savefig(join(plot_dir, "nvtx/2017_{0}_{1}_{2}.png".format(location, ptrange, wplabel)), bbox_inches='tight')

                plt.close()
