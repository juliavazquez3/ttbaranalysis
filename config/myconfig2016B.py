from analysis_tools import ObjectCollection, Category, Process, Dataset, Feature, Systematic
from analysis_tools.utils import DotDict
from analysis_tools.utils import join_root_selection as jrs
from plotting_tools import Label
from collections import OrderedDict

from cmt.config.ul_2018 import Config_ul_2018 as cmt_config


class Config(cmt_config):
    def __init__(self, *args, **kwargs):
        # https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation102X
        self.btag=DotDict(tight=0.7264, medium=0.2770, loose=0.0494)

        super(Config, self).__init__(*args, **kwargs)

    def add_datasets(self):
        datasets = []

        datasets.append(
            Dataset("ttbar_sl",
                dataset="/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/"
                    "RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
                process=self.processes.get("ttbar_sl"),
                xs=365.34,
                tags=["ul"]),
        )
        datasets.append(
            Dataset("ttbar_dl",
                dataset="/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/"
                    "RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
                process=self.processes.get("ttbar_dl"),
                xs=88.29,
                tags=["ul"]),
        )
        datasets.append(
            Dataset("ttbar_dh",
                dataset="/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/"
                    "RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
                process=self.processes.get("ttbar_dh"),
                xs=377.96,
                tags=["ul"]),
        )
        datasets.append(
            Dataset("st_1",
                dataset="/ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8/"
                    "RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM",
                process=self.processes.get("st_1"),
                xs=35.85,
                tags=["ul"]),
        )
        datasets.append(
            Dataset("st_2",
                dataset="/ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8/"
                    "RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM",
                process=self.processes.get("st_2"),
                xs=35.85,
                tags=["ul"]),
        )
        datasets.append(
            Dataset("st_3",
                dataset="/ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5CR2_13TeV-powheg-madspin-pythia8/"
                    "RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
                process=self.processes.get("st_3"),
                xs=80.95,
                tags=["ul"]),
        )
        datasets.append(
            Dataset("st_4",
                dataset="/ST_t-channel_top_4f_InclusiveDecays_TuneCP5CR2_13TeV-powheg-madspin-pythia8/"
                    "RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
                process=self.processes.get("st_4"),
                xs=136.02,
                tags=["ul"]),
        )
        datasets.append(
            Dataset("wjets_1",
                dataset="/WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8/"
                    "RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
                process=self.processes.get("wjets_1"),
                xs=1292.0,
                tags=["ul"]),
        )
        datasets.append(
            Dataset("wjets_2",
                dataset="/WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8/"
                    "RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
                process=self.processes.get("wjets_2"),
                xs=1627.45,
                tags=["ul"]),
        )
        datasets.append(
            Dataset("wjets_3",
                dataset="/WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8/"
                    "RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
                process=self.processes.get("wjets_3"),
                xs=435.237,
                tags=["ul"]),
        )
        datasets.append(
            Dataset("wjets_4",
                dataset="/WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8/"
                    "RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
                process=self.processes.get("wjets_4"),
                xs=59.1811,
                tags=["ul"]),
        )
        datasets.append(
            Dataset("wjets_5",
                dataset="/WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8/"
                    "RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
                process=self.processes.get("wjets_5"),
                xs=14.5805,
                tags=["ul"]),
        )
        datasets.append(
            Dataset("wjets_6",
                dataset="/WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8/"
                    "RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
                process=self.processes.get("wjets_6"),
                xs=6.65621,
                tags=["ul"]),
        )
        datasets.append(
            Dataset("wjets_7",
                dataset="/WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8/"
                    "RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
                process=self.processes.get("wjets_7"),
                xs=1.60809,
                tags=["ul"]),
        )
        datasets.append(
            Dataset("wjets_8",
                dataset="/WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8/"
                    "RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM",
                process=self.processes.get("wjets_8"),
                xs=0.0389136,
                tags=["ul"]),
        )
        datasets.append(
            Dataset("ww",
                dataset="/WW_TuneCP5_13TeV-pythia8/"
                    "RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
                process=self.processes.get("ww"),
                xs=75.8,
                tags=["ul"]),
        )
        datasets.append(
            Dataset("wz",
                dataset="/WZ_TuneCP5_13TeV-pythia8/"
                    "RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
                process=self.processes.get("wz"),
                xs=27.6,
                tags=["ul"]),
        )
        datasets.append(
            Dataset("zz",
                dataset="/ZZ_TuneCP5_13TeV-pythia8/"
                    "RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
                process=self.processes.get("zz"),
                xs=12.14,
                tags=["ul"]),
        )
        datasets.append(
            Dataset("zjets_1",
                dataset="/DYJetsToLL_M-50_HT-70to100_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/"
                    "RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM",
                process=self.processes.get("zjets_1"),
                xs=208.977,
                tags=["ul"]),
        )
        datasets.append(
            Dataset("zjets_2",
                dataset="/DYJetsToLL_M-50_HT-100to200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/"
                    "RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM",
                process=self.processes.get("zjets_2"),
                xs=181.302,
                tags=["ul"]),
        )
        datasets.append(
            Dataset("zjets_3",
                dataset="/DYJetsToLL_M-50_HT-200to400_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/"
                    "RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM",
                process=self.processes.get("zjets_3"),
                xs=50.4177,
                tags=["ul"]),
        )
        datasets.append(
            Dataset("zjets_4",
                dataset="/DYJetsToLL_M-50_HT-400to600_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/"
                    "RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM",
                process=self.processes.get("zjets_4"),
                xs=6.98394,
                tags=["ul"]),
        )
        datasets.append(
            Dataset("zjets_5",
                dataset="/DYJetsToLL_M-50_HT-600to800_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/"
                    "RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM",
                process=self.processes.get("zjets_5"),
                xs=1.68141,
                tags=["ul"]),
        )
        datasets.append(
            Dataset("zjets_6",
                dataset="/DYJetsToLL_M-50_HT-800to1200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/"
                    "RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM",
                process=self.processes.get("zjets_6"),
                xs=0.775392,
                tags=["ul"]),
        )
        datasets.append(
            Dataset("zjets_7",
                dataset="/DYJetsToLL_M-50_HT-1200to2500_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/"
                    "RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM",
                process=self.processes.get("zjets_7"),
                xs=0.186222,
                tags=["ul"]),
        )
        datasets.append(
            Dataset("zjets_8",
                dataset="/DYJetsToLL_M-50_HT-2500toInf_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/"
                    "RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM",
                process=self.processes.get("zjets_8"),
                xs=0.00438495,
                tags=["ul"]),
        )
        datasets.append(
            Dataset("data_el_f",
                dataset="/SingleElectron/Run2016F-UL2016_MiniAODv2_NanoAODv9-v1/NANOAOD",
                selection="pairType == 1",
                process=self.processes.get("data_el"),
                runPeriod="F",
                # prefix="xrootd-cms.infn.it//",
                splitting=-1,
                tags=["ul","electron"]),
        )
        datasets.append(
            Dataset("data_el_g",
                dataset="/SingleElectron/Run2016G-UL2016_MiniAODv2_NanoAODv9-v1/NANOAOD",
                selection="pairType == 1",
                process=self.processes.get("data_el"),
                runPeriod="G",
                # prefix="xrootd-cms.infn.it//",
                splitting=-1,
                tags=["ul","electron"]),
        )
        datasets.append(
            Dataset("data_el_h",
                dataset="/SingleElectron/Run2016H-UL2016_MiniAODv2_NanoAODv9-v1/NANOAOD",
                selection="pairType == 1",
                process=self.processes.get("data_el"),
                runPeriod="H",
                # prefix="xrootd-cms.infn.it//",
                splitting=-1,
                tags=["ul","electron"]),
        )
        # SingleMuon 2018
        datasets.append(
            Dataset("data_mu_f",
                dataset="/SingleMuon/Run2016F-UL2016_MiniAODv2_NanoAODv9-v1/NANOAOD",
                selection="pairType == 0",
                process=self.processes.get("data_mu"),
                runPeriod="F",
                # prefix="xrootd-cms.infn.it//",
                splitting=-1,
                tags=["ul","muon"]),
        )
        datasets.append(
            Dataset("data_mu_g",
                dataset="/SingleMuon/Run2016G-UL2016_MiniAODv2_NanoAODv9-v1/NANOAOD",
                selection="pairType == 0",
                process=self.processes.get("data_mu"),
                runPeriod="G",
                # prefix="xrootd-cms.infn.it//",
                splitting=-1,
                tags=["ul","muon"]),
        )
        datasets.append(
            Dataset("data_mu_h",
                dataset="/SingleMuon/Run2016H-UL2016_MiniAODv2_NanoAODv9-v1/NANOAOD",
                selection="pairType == 0",
                process=self.processes.get("data_mu"),
                runPeriod="H",
                # prefix="xrootd-cms.infn.it//",
                splitting=-1,
                tags=["ul","muon"]),
        )
        return ObjectCollection(datasets)

    def add_processes(self):
        processes = [
            ### my additions
            Process("wjetsm", Label("W plus jets"), color=(255, 153, 0), llr_name="WJ"),
            Process("wjets_1", Label("W plus jets"), color=(205, 0, 9), parent_process="wjetsm"),
            Process("wjets_2", Label("W plus jets"), color=(205, 0, 9), parent_process="wjetsm"),
            Process("wjets_3", Label("W plus jets"), color=(205, 0, 9), parent_process="wjetsm"),
            Process("wjets_4", Label("W plus jets"), color=(205, 0, 9), parent_process="wjetsm"),
            Process("wjets_5", Label("W plus jets"), color=(205, 0, 9), parent_process="wjetsm"),
            Process("wjets_6", Label("W plus jets"), color=(205, 0, 9), parent_process="wjetsm"),
            Process("wjets_7", Label("W plus jets"), color=(205, 0, 9), parent_process="wjetsm"),
            Process("wjets_8", Label("W plus jets"), color=(205, 0, 9), parent_process="wjetsm"),

            Process("zjets", Label("Z plus jets"), color=(255, 153, 0), llr_name="ZJ"),
            Process("zjets_1", Label("Z plus jets"), color=(205, 0, 9), parent_process="zjets"),
            Process("zjets_2", Label("Z plus jets"), color=(205, 0, 9), parent_process="zjets"),
            Process("zjets_3", Label("Z plus jets"), color=(205, 0, 9), parent_process="zjets"),
            Process("zjets_4", Label("Z plus jets"), color=(205, 0, 9), parent_process="zjets"),
            Process("zjets_5", Label("Z plus jets"), color=(205, 0, 9), parent_process="zjets"),
            Process("zjets_6", Label("Z plus jets"), color=(205, 0, 9), parent_process="zjets"),
            Process("zjets_7", Label("Z plus jets"), color=(205, 0, 9), parent_process="zjets"),
            Process("zjets_8", Label("Z plus jets"), color=(205, 0, 9), parent_process="zjets"),

            Process("ww", Label("WW"), color=(255, 153, 0), llr_name="WW"),
            Process("wz", Label("WZ"), color=(255, 153, 0), llr_name="WZ"),
            Process("zz", Label("ZZ"), color=(255, 153, 0), llr_name="ZZ"),

            Process("ttbar", Label("t#bar{t}"), color=(255, 153, 0), llr_name="TT"),
            Process("ttbar_dl", Label("t#bar{t} DL"), color=(205, 0, 9), parent_process="ttbar"),
            Process("ttbar_sl", Label("t#bar{t} SL"), color=(255, 153, 0), parent_process="ttbar"),
            Process("ttbar_dh", Label("t#bar{t} FH"), color=(131, 38, 10), parent_process="ttbar"),

            Process("st_y", Label("st"), color=(255, 153, 0), llr_name="TT"),
            Process("st_1", Label("st"), color=(205, 0, 9), parent_process="st_y"),
            Process("st_2", Label("st"), color=(255, 153, 0), parent_process="st_y"),
            Process("st_3", Label("st"), color=(131, 38, 10), parent_process="st_y"),
            Process("st_4", Label("st"), color=(131, 38, 10), parent_process="st_y"),

            ### jaime's
            Process("ggf", Label("$HH_{ggF}$"), color=(0, 0, 0), isSignal=True, llr_name="ggH"),
            Process("ggf_sm", Label("$HH_{ggF}$"), color=(0, 0, 0), isSignal=True, parent_process="ggf"),

            Process("vbf", Label("$HH_{VBF}$"), color=(0, 0, 0), isSignal=True, llr_name="qqH"),
            Process("vbf_sm", Label("$HH_{VBF}$"), color=(0, 0, 0), isSignal=True, parent_process="vbf"),
            Process("vbf_0p5_1_1", Label("$HH_{VBF}^{(0.5,1,1)}$"), color=(0, 0, 0),
                isSignal=True, parent_process="vbf"),
            Process("vbf_1p5_1_1", Label("$HH_{VBF}^{(1.5,1,1)}$"), color=(0, 0, 0),
                isSignal=True, parent_process="vbf"),
            Process("vbf_1_0_1", Label("$HH_{VBF}^{(1,0,1)}$"), color=(0, 0, 0),
                isSignal=True, parent_process="vbf"),
            Process("vbf_1_1_0", Label("$HH_{VBF}^{(1,1,0)}$"), color=(0, 0, 0),
                isSignal=True, parent_process="vbf"),
            Process("vbf_1_1_2", Label("$HH_{VBF}^{(1,1,2)}$"),
                color=(0, 0, 0), isSignal=True, parent_process="vbf"),
            Process("vbf_1_2_1", Label("$HH_{VBF}^{(1,2,1)}$"),
                color=(0, 0, 0), isSignal=True, parent_process="vbf"),

            Process("dy", Label("DY"), color=(255, 102, 102), isDY=True, llr_name="DY"),
            Process("dy_high", Label("DY"), color=(255, 102, 102), isDY=True, parent_process="dy"),

            Process("tt", Label("t#bar{t}"), color=(255, 153, 0), llr_name="TT"),
            Process("tt_dl", Label("t#bar{t} DL"), color=(205, 0, 9), parent_process="tt"),
            Process("tt_sl", Label("t#bar{t} SL"), color=(255, 153, 0), parent_process="tt"),
            Process("tt_fh", Label("t#bar{t} FH"), color=(131, 38, 10), parent_process="tt"),

            Process("tth", Label("t#bar{t}H"), color=(255, 153, 0), llr_name="TTH"),
            Process("tth_bb", Label("t#bar{t}H"), color=(255, 153, 0), parent_process="tth"),
            Process("tth_tautau", Label("t#bar{t}H"), color=(255, 153, 0), parent_process="tth"),
            Process("tth_nonbb", Label("t#bar{t}H"), color=(255, 153, 0), parent_process="tth"),

            Process("others", Label("Others"), color=(134, 136, 138)),
            Process("wjets", Label("W + jets"), color=(134, 136, 138), parent_process="others",
                llr_name="WJets"),
            Process("tw", Label("t + W"), color=(134, 136, 138), parent_process="others",
                llr_name="TW"),
            Process("singlet", Label("Single t"), color=(134, 136, 138), parent_process="others",
                llr_name="singleT"),

            Process("data", Label("DATA"), color=(0, 0, 0), isData=True),
            Process("data_tau", Label("DATA\_TAU"), color=(0, 0, 0), parent_process="data", isData=True),
            Process("data_etau", Label("DATA\_E"), color=(0, 0, 0), parent_process="data", isData=True),
            Process("data_mutau", Label("DATA\_MU"), color=(0, 0, 0), parent_process="data", isData=True),
            Process("data_el", Label("DATA\_E"), color=(0, 0, 0), parent_process="data", isData=True),
            Process("data_mu", Label("DATA\_MU"), color=(0, 0, 0), parent_process="data", isData=True)
        ]

        process_group_names = {
            "default": [
                "ggf_sm",
                "data_tau",
                "dy_high",
                "tt_dl",
            ],
            "data_tau": [
                "data_tau",
            ],
            "data_etau": [
                "data_etau",
            ],
            "bkg": [
                "tt_dl",
            ],
            "signal": [
                "ggf_sm",
            ],
            "etau": [
                "tt_dl",
                "tt_sl",
                "dy",
                "wjets",
                "data_etau",
            ],
            "mutau": [
                "tt_dl",
                "tt_sl",
                "dy",
                "wjets",
                "data_mutau",
            ],
            "tautau": [
                "tt_dl",
                "tt_sl",
                "dy",
                "wjets",
                "data_tau",
            ],
            "vbf": [
                "vbf_sm",
                "vbf_0p5_1_1",
                "vbf_1p5_1_1",
                "vbf_1_0_1",
                "vbf_1_1_0",
                "vbf_1_1_2",
                "vbf_1_2_1"
            ]
        }

        process_training_names = {
            "default": [
                "ggf_sm",
                "dy"
            ]
        }

        return ObjectCollection(processes), process_group_names, process_training_names

    def add_features(self):
        features = [
            # Jet 
            Feature("jet_pt", "Jet_pt", binning=(10, 50, 150),
                x_title=Label("jet p_{t}"),
                units="GeV",
                central="jet_smearing"),
            Feature("jet_mass", "Jet_mass", binning=(10, 50, 150),
                x_title=Label("jet p_{t}"),
                units="GeV",
                central="jet_smearing"),


            # MET
            Feature("met_phi", "MET_phi", binning=(20, -3.2, 3.2),
                x_title=Label("MET #phi"),
                central="met_smearing"),
            Feature("met_pt", "MET_pt", binning=(10, 50, 150),
                x_title=Label("MET p_t"),
                units="GeV",
                central="met_smearing"),
            # Weights
            Feature("genWeight", "genWeight", binning=(20, 0, 2),
                x_title=Label("genWeight")),
            Feature("puWeight", "puWeight", binning=(20, 0, 2),
                x_title=Label("puWeight")),
            Feature("PUjetID_SF", "PUjetID_SF", binning=(20, 0, 2),
                x_title=Label("PUjetID_SF")),

        ]
        return ObjectCollection(features)

    def add_weights(self):
        weights = DotDict()
        weights.default = "1"

        # weights.total_events_weights = ["genWeight", "puWeight", "DYstitchWeight"]
        weights.total_events_weights = ["genWeight", "puWeight"]

        weights.mutau = ["genWeight", "puWeight", "trigSF","PUjetID_SF"]

        weights.etau = weights.mutau
        weights.tautau = weights.mutau

        # weights.channels_mult = {channel: jrs(weights.channels[channel], op="*")
            # for channel in weights.channels}
        return weights

    def add_systematics(self):
        systematics = [
            Systematic("met_smearing", ("MET", "MET_smeared")),
            Systematic("jet_smearing", "_nom"),
            Systematic("prefiring", "_Nom"),
            Systematic("prefiring_syst", "", up="_Up", down="_Dn"),
            Systematic("empty", "", up="", down="")
        ]
        return ObjectCollection(systematics)

    # other methods

    def get_channel_from_region(self, region):
        for sign in ["os", "ss"]:
            if sign in region.name:
                if region.name.startswith(sign):
                    return ""
                return region.name[:region.name.index("_%s" % sign)]
        return ""

    def add_channels(self):
        channels = [
            #Category("mutau", Label("#tau_{#mu}#tau_{h}"), selection="pairType == 0"),
            #Category("etau", Label("#tau_{e}#tau_{h}"), selection="pairType == 1"),
            #Category("tautau", Label("#tau_{h}#tau_{h}"), selection="pairType == 2"),
        ]
        return ObjectCollection(channels)

    def add_categories(self):
        categories = [
            Category("base", "base category", selection="1"),
            Category("base_sv", "base category", selection="1"),
            Category("base_wqq", "base category", selection="1"),
            Category("base_selection", "base category"),
            Category("dum", "dummy category", selection="event == 74472670"),
            Category("muchannel_sl", "#mu# channel", selection='lepton_type==0'),
            Category("echannel_sl", "#e# channel", selection='lepton_type==1'),
            Category("muchannel_sv", "#mu# channel", selection='lepton_type==0'),
            Category("echannel_sv", "#e# channel", selection='lepton_type==1'),
        ]
        return ObjectCollection(categories)

#    def get_qcd_regions(self, region, category, wp="", shape_region="os_inviso",
#            signal_region_wp="os_iso", sym=False):
#        # the region must be set and tagged os_iso
#        if not region:
#            raise Exception("region must not be empty")
#        # if not region.has_tag("qcd_os_iso"):
#        #     raise Exception("region must be tagged as 'qcd_os_iso' but isn't")
#        # the category must be compatible with the estimation technique
#        # if category.has_tag("qcd_incompatible"):
#        #     raise Exception("category '{}' incompatible with QCD estimation".format(category.name))
#
#        if wp != "":
#            wp = "__" + wp
#
#        # get other qcd regions
#        prefix = region.name[:-len(signal_region_wp)]
#        qcd_regions = {"ss_inviso": self.regions.get(prefix + "ss_inviso" + wp)}
#        # for the inverted regions, allow different working points
#        default_config = ["os_inviso", "ss_iso"]
#        for key in default_config:
#            region_name = (prefix + key + wp if key != "ss_iso"
#                else prefix + "ss_" + signal_region_wp[len("os_"):])
#            qcd_regions[key] = self.regions.get(region_name)
#
#        if sym:
#            qcd_regions["shape1"] = self.regions.get(prefix + shape_region + wp)
#            qcd_regions["shape2"] = self.regions.get(
#                prefix + "ss_" + signal_region_wp[len("os_"):])
#        else:
#            if shape_region == "os_inviso":
#                qcd_regions["shape"] = self.regions.get(prefix + shape_region + wp)
#            else:
#                qcd_regions["shape"] = self.regions.get(
#                    prefix + "ss_" + signal_region_wp[len("os_"):])
#        return DotDict(qcd_regions)


config = Config("myconfig2016B", year=2016, ecm=13, lumi_pb=59741, isPreVFP=False)
