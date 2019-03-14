import matplotlib.pyplot as plt
import numpy  as np
import pandas as pd
import sys
sys.path.append(\

%matplotlib inline
#plt.style.use('physics')

plt.rcParams['axes.grid'       ]  = False
plt.rcParams['xtick.labelsize' ]  = 14
plt.rcParams['ytick.labelsize' ]  = 14
plt.rcParams['axes.labelsize'  ]  = 14
plt.rcParams['legend.fancybox' ]  = False

pd.options.mode.chained_assignment = None
import os 
os.path.exists('2017_Analysis_Without_datadriven_PUJID_new_ptHjj_var.h5')
# data_raw = pd.read_hdf('../data/hgg-trees-moriond-2017.h5')
# data_raw = pd.read_hdf('From_Seth_without_datadriven_generated_by_me_no_photon_ID_cut.h5')
data_raw = pd.read_hdf('2017_Analysis_Without_datadriven_PUJID_new_ptHjj_var.h5')
data_raw['dijet_centrality_gg'] = np.exp(-4*(data_raw.dijet_Zep/data_raw.dijet_abs_dEta)**2)
def evtPassPUJID( row ):
    leadPtIndex = -1
    if row['dijet_LeadJPt'] > 30: leadPtIndex = 3
    elif row['dijet_LeadJPt'] > 20: leadPtIndex = 2
    elif row['dijet_LeadJPt'] > 10: leadPtIndex = 1
    elif row['dijet_LeadJPt'] > 0: leadPtIndex = 0
    
    #similarly for eta
    leadEtaIndex = -1
    if abs(row['dijet_leadEta']) > 3: leadEtaIndex = 3
    elif abs(row['dijet_leadEta']) > 2.75: leadEtaIndex = 2
    elif abs(row['dijet_leadEta']) > 2.5: leadEtaIndex = 1
    elif abs(row['dijet_leadEta']) > 0: leadEtaIndex = 0
        
    subleadPtIndex = -1
    if row['dijet_SubJPt'] > 30: subleadPtIndex = 3
    elif row['dijet_SubJPt'] > 20: subleadPtIndex = 2
    elif row['dijet_SubJPt'] > 10: subleadPtIndex = 1
    elif row['dijet_SubJPt'] > 0: subleadPtIndex = 0
    
    #similarly for eta
    subleadEtaIndex = -1
    if abs(row['dijet_subleadEta']) > 3: subleadEtaIndex = 3
    elif abs(row['dijet_subleadEta']) > 2.75: subleadEtaIndex = 2
    elif abs(row['dijet_subleadEta']) > 2.5: subleadEtaIndex = 1
    elif abs(row['dijet_subleadEta']) > 0: subleadEtaIndex = 0
            
    pujidCutsVals = {}
    pujidCutsVals[ (0,0) ] = 0.69 #cut value for zeroth pt bin, zeroth eta bin
    pujidCutsVals[ (1,0) ] = 0.69 #cut value for first pt bin, zeroth eta bin
    pujidCutsVals[ (2,0) ] = 0.69
    pujidCutsVals[ (3,0) ] = 0.86
    
    pujidCutsVals[ (0,1) ] = -0.35
    pujidCutsVals[ (1,1) ] = -0.35
    pujidCutsVals[ (2,1) ] = -0.35
    pujidCutsVals[ (3,1) ] = -0.10
    
    pujidCutsVals[ (0,2) ] = -0.26
    pujidCutsVals[ (1,2) ] = -0.26
    pujidCutsVals[ (2,2) ] = -0.26
    pujidCutsVals[ (3,2) ] = -0.05
    
    pujidCutsVals[ (0,3) ] = -0.21
    pujidCutsVals[ (1,3) ] = -0.21
    pujidCutsVals[ (2,3) ] = -0.21
    pujidCutsVals[ (3,3) ] = -0.01
        
    leadJetPasses = False
    if row['dijet_jet1_pujid_mva'] > pujidCutsVals[ (leadPtIndex, leadEtaIndex) ]: leadJetPasses=True
    
    #repeat for sublead jet
    subleadJetPasses = False
    if row['dijet_jet2_pujid_mva'] > pujidCutsVals[ (subleadPtIndex, subleadEtaIndex) ]: subleadJetPasses=True
    
    evtPasses = leadJetPasses and subleadJetPasses
    return evtPasses
data_raw[\
data_raw = data_raw[data_raw[\
data_all = data_raw[(data_raw.dipho_sublead_elveto == 1   ) & 
                    (data_raw.dipho_lead_elveto    == 1   ) ] 
data_all['m_sideband'] = np.abs(data_all.dipho_mass - 125 ) > 10
data_all['cr_region'] = np.chararray(data_all.shape[0], itemsize=4)

data_all.loc[(data_all.min_id>-0.2), 'cr_region'] = 'PP'
data_all.loc[(data_all.max_id<-0.4), 'cr_region'] = 'FF'
data_all.loc[(data_all.dipho_leadIDMVA    >-0.2) & (data_all.dipho_subleadIDMVA <-0.4), 'cr_region'] = 'PF'
data_all.loc[(data_all.dipho_subleadIDMVA >-0.2) & (data_all.dipho_leadIDMVA    <-0.4), 'cr_region'] = 'FP'
def vbf_presel(data):
    return (
        (data[\
        (data[\
        (data[\
        (data[\
        (data[\
        (data[\
        (data[\

data_all['isvbf'] = vbf_presel(data_all)
data_all['isvbf'] = vbf_presel(data_all)
data_all['eta_cat'] = np.chararray(data_all.shape[0], itemsize=4)

data_all.loc[((np.abs(data_all.dipho_leadEta   )>=0)&(np.abs(data_all.dipho_leadEta   )<1.5) & 
              (np.abs(data_all.dipho_subleadEta)>=0)&(np.abs(data_all.dipho_subleadEta)<1.5)), 'eta_cat']  = 'EBEB'

data_all.loc[((np.abs(data_all.dipho_leadEta   )>=0)&(np.abs(data_all.dipho_leadEta   )<1.5) & 
              (np.abs(data_all.dipho_subleadEta)>=1.5)&(np.abs(data_all.dipho_subleadEta)<=3)), 'eta_cat'] = 'EBEE'

data_all.loc[((np.abs(data_all.dipho_leadEta   )>=1.5)&(np.abs(data_all.dipho_leadEta   )<=3) & 
              (np.abs(data_all.dipho_subleadEta)>=0)&(np.abs(data_all.dipho_subleadEta)<1.5)), 'eta_cat']  = 'EEEB'

data_all.loc[((np.abs(data_all.dipho_leadEta   )>=1.5)&(np.abs(data_all.dipho_leadEta   )<=3) & 
              (np.abs(data_all.dipho_subleadEta)>=1.5)&(np.abs(data_all.dipho_subleadEta)<=3)), 'eta_cat'] = 'EEEE'
data_all['lead_eta_cat'] = np.chararray(data_all.shape[0], itemsize=4)
data_all['subl_eta_cat'] = np.chararray(data_all.shape[0], itemsize=4)

data_all.loc[((np.abs(data_all.dipho_leadEta)>=0  )&(np.abs(data_all.dipho_leadEta)<1.5)),'lead_eta_cat']='EB'
data_all.loc[((np.abs(data_all.dipho_leadEta)>=1.5)&(np.abs(data_all.dipho_leadEta)<= 3)),'lead_eta_cat']='EE'

data_all.loc[((np.abs(data_all.dipho_subleadEta)>=0  )&(np.abs(data_all.dipho_subleadEta)<1.5)),'subl_eta_cat']='EB'
data_all.loc[((np.abs(data_all.dipho_subleadEta)>=1.5)&(np.abs(data_all.dipho_subleadEta)<=3 )),'subl_eta_cat']='EE'
data_all['avg_et' ] = (data_all.dipho_leadEt +  data_all.dipho_subleadEt)/2.0
data_all['diff_et'] = (data_all.dipho_leadEt -  data_all.dipho_subleadEt)
def divide( a, b ):
    \
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide( a, b )
        c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
    return c
wbins = np.array([20,30,45,65,100,150,1000])
def fake_factor(data, variable, bins, region = 'FP'):
    xd = np.array([(bins[i+1]+ bins[i])/2.0 for i in range(len(bins)-1)])
    xe = np.array([(bins[i+1]- bins[i])/2.0 for i in range(len(bins)-1)])
    
    h_pass,_ = np.histogram(data[data.cr_region=='PP'][variable],
                            weights=data[data.cr_region=='PP'].weight, bins=bins )
    h_fail,_ = np.histogram(data[data.cr_region==region][variable],
                            weights=data[data.cr_region==region].weight, bins=bins )
    factor = divide(h_pass,h_fail)
    er1 = divide(1.0,np.sqrt(h_pass))
    er2 = divide(1.0,np.sqrt(h_fail))
    ferr = factor * np.sqrt(er1**2 + er2**2)
    up = dw = ferr/2.0
    return (xd,xe,factor, dw, up)
def qcd_purity(data, variable, bins, region = 'PP'):
    xd = np.array([(bins[i+1]+ bins[i])/2.0 for i in range(len(bins)-1)])
    xe = np.array([(bins[i+1]- bins[i])/2.0 for i in range(len(bins)-1)])
    
    h_QCD,_ = np.histogram(data[(data.cr_region==region)&(data.Z == 0)][variable],
                            weights=data[(data.cr_region==region)&(data.Z == 0)].weight, 
                            bins=bins )
    h_EWK,_ = np.histogram(data[(data.cr_region==region)&(data.Z==1)][variable],
                            weights=data[(data.cr_region==region)&(data.Z==1)].weight, 
                            bins=bins )
    h_ALL = h_EWK + h_QCD
    factor = divide(h_QCD,h_ALL)
    er1 = divide(1.0,np.sqrt(h_QCD))
    er2 = divide(1.0,np.sqrt(h_ALL))
    ferr = factor * (1 -  factor) / h_ALL
    
    up = np.minimum(ferr/2.0,np.ones (ferr.shape[0]))
    dw = np.maximum(ferr/2.0,np.zeros(ferr.shape[0]))
    return (xd,xe,factor,dw, up)
wbins = np.array([20,30,45,65,1000])


fig, ax = plt.subplots(1,4, sharex=True, sharey=True, figsize=(14,4))
fig.subplots_adjust(wspace=0)
for i, reg in enumerate(np.unique(data_all.lead_eta_cat)):
    x, xe,f_pf,dw_pf,up_pf = fake_factor(data_all[(data_all.Y==0)&(data_all.Z==0)&(data_all.lead_eta_cat==reg)],
                             'dipho_subleadPt',wbins,region='PF')
    x, xe,f_fp,dw_fp,up_fp = fake_factor(data_all[(data_all.Y==0)&(data_all.Z==0)&(data_all.subl_eta_cat==reg)],
                             'dipho_leadPt',wbins,region='FP')
    ax[i].errorbar(x,f_pf,xerr=xe,fmt='.', markersize=11,capthick=0, alpha=1.0, color='r', label='PF $\\\\to$ PP')
    ax[i+2].errorbar(x,f_fp,xerr=xe,fmt='.', markersize=11,capthick=0, alpha=1.0, color='b', label='FP $\\\\to$ PP')
    ax[i].bar(x,2*(up_pf),bottom = f_pf - dw_pf, width= 2*xe, color='r',alpha=0.3, zorder=9, 
            align='center',edgecolor='None',lw=0.0)
    ax[i+2].bar(x,2*(up_fp),bottom = f_fp - dw_fp, width= 2*xe, color='b',alpha=0.3, zorder=9, 
            align='center',edgecolor='None',lw=0.0)
    ax[i  ].axhline(0,color='black',ls='--',lw=0.5)
    ax[i+2].axhline(0,color='black',ls='--',lw=0.5)
    ax[i].set_xscale('log')
    ax[i  ].set_xlabel('$p_{T}(\\gamma_%i)$' % 2)
    ax[i+2].set_xlabel('$p_{T}(\\gamma_%i)$' % 1)
    ax[i].set_xlim(np.min(wbins), np.max(wbins))
    ax[i].set_ylim(-0.5, 1.5)
    ax[i].annotate(reg,xy=(0.1,0.9), fontsize=15,xycoords='axes fraction')
    ax[i+2].annotate(reg,xy=(0.1,0.9), fontsize=15,xycoords='axes fraction')
    ax[i].legend(fontsize=12)
    ax[i+2].legend(fontsize=12)
    
    ax[i].set_ylabel('fake factor')
wbins = np.array([20,30,45,65,1000])


fig, ax = plt.subplots(1,4, sharex=True, sharey=True, figsize=(14,4))
fig.subplots_adjust(wspace=0)
for i, reg in enumerate(np.unique(data_all.lead_eta_cat)):
    x, xe,pf,dw_pf,up_pf = qcd_purity(data_all[(data_all.Y==0)&(data_all.subl_eta_cat==reg)],'dipho_subleadPt',wbins,region='PF')
    x, xe,fp,dw_fp,up_fp = qcd_purity(data_all[(data_all.Y==0)&(data_all.lead_eta_cat==reg)],'dipho_leadPt'   ,wbins,region='FP')
    
    x, xe,ff,dw_ff,up_ff = qcd_purity(data_all[(data_all.Y==0)&(data_all.lead_eta_cat==reg)],'avg_pt',wbins,region='FF')
    x, xe,pp,dw_pp,up_pp = qcd_purity(data_all[(data_all.Y==0)&(data_all.lead_eta_cat==reg)],'avg_pt',wbins,region='PP')

    ax[i].errorbar(x,pf,xerr=xe,fmt='.', markersize=11,capthick=0, alpha=1.0, color='r', label='PF')
    ax[i].errorbar(x,fp,xerr=xe,fmt='.', markersize=11,capthick=0, alpha=1.0, color='b', label='FP')
    
    ax[i+2].errorbar(x,ff,xerr=xe,fmt='.', markersize=11,capthick=0, alpha=1.0, color='r', label='FF')
    ax[i+2].errorbar(x,pp,xerr=xe,fmt='.', markersize=11,capthick=0, alpha=1.0, color='b', label='PP')
    ax[i].bar(x,2*(up_pf),bottom = pf - dw_pf, width= 2*xe, color='r',alpha=0.3, zorder=9, 
            align='center',edgecolor='None',lw=0.0)
    ax[i].bar(x,2*(up_fp),bottom =(fp - dw_fp), width= 2*xe, color='b',alpha=0.3, zorder=9, 
            align='center',edgecolor='None',lw=0.0)
    
    ax[i+2].bar(x,2*(up_ff),bottom = ff - dw_ff, width= 2*xe, color='r',alpha=0.3, zorder=9, 
            align='center',edgecolor='None',lw=0.0)
    ax[i+2].bar(x,2*(up_pp),bottom =(pp - dw_pp), width= 2*xe, color='b',alpha=0.3, zorder=9, 
            align='center',edgecolor='None',lw=0.0)
    ax[i].axhline(1,color='red',ls='--',lw=0.5)
    ax[i+2].axhline(1,color='red',ls='--',lw=0.5)
    ax[i].set_xscale('log')
    ax[i].set_xlim(np.min(wbins), np.max(wbins))
    ax[i].set_ylim(0, 2)
    ax[i].annotate(reg,xy=(0.1,0.9), fontsize=15,xycoords='axes fraction')
    ax[i+2].annotate(reg,xy=(0.1,0.9), fontsize=15,xycoords='axes fraction')
    ax[i].legend(fontsize=12)
    ax[i+2].legend(fontsize=12)
    
ax[0].set_ylabel('QCD fraction')
np.unique(data_all[(data_all.Z==0)]['sample'])
fake_PF = {}
fake_PF['EB'] = fake_factor(data_all[(data_all['sample'] == 'gjet')&(data_all.lead_eta_cat=='EB')],
                            'dipho_subleadPt',wbins,region='PF')[2]
fake_PF['EE'] = fake_factor(data_all[(data_all['sample'] == 'gjet')&(data_all.lead_eta_cat=='EE')],
                            'dipho_subleadPt',wbins,region='PF')[2]
fake_PF
purity_PF = {}
purity_PF['EB'] = qcd_purity(data_all[(data_all.Y==0)&(data_all.lead_eta_cat=='EB')],
                            'dipho_subleadPt',wbins,region='PF')[2]
purity_PF['EE'] = qcd_purity(data_all[(data_all.Y==0)&(data_all.lead_eta_cat=='EE')],
                            'dipho_subleadPt',wbins,region='PF')[2]
purity_PF
fake_FP = {}
fake_FP['EB'] = fake_factor(data_all[(data_all['sample'] == 'gjet')&(data_all.lead_eta_cat=='EB')],
                            'dipho_leadPt',wbins,region='FP')[2]
fake_FP['EE'] = fake_factor(data_all[(data_all['sample'] == 'gjet')&(data_all.lead_eta_cat=='EE')],
                            'dipho_leadPt',wbins,region='FP')[2]

fake_FP
purity_FP = {}
purity_FP['EB'] = qcd_purity(data_all[(data_all.Y==0)&(data_all.lead_eta_cat=='EB')],
                            'dipho_leadPt',wbins,region='FP')[2]
purity_FP['EE'] = qcd_purity(data_all[(data_all.Y==0)&(data_all.lead_eta_cat=='EE')],
                            'dipho_leadPt',wbins,region='FP')[2]
purity_FP
fake_FF = {}
fake_FF['EB'] = fake_factor(data_all[(data_all['sample'] == 'qcd')&(data_all.eta_cat=='EB')],
                            'dipho_leadPt',wbins,region='FP')[2]
fake_FF['EE'] = fake_factor(data_all[(data_all['sample'] == 'qcd')&(data_all.eta_cat=='EE')],
                            'dipho_leadPt',wbins,region='FP')[2]
fake_FF
data_region = data_all[(data_all.pass_id == True) & (data_all.Y == 2)
                      & (data_all.isvbf)
                      ]
data_SMMC   = data_all[(data_all.pass_id == True) & (data_all.Y == 0)
                      & (data_all.isvbf)
                      ]
data_signal = data_all[(data_all.pass_id == True) & (data_all.Y == 1)
                      & (data_all.isvbf)
                      ]
dd_FP = data_all[(data_all.cr_region == 'FP') & (data_all.Y == 2)
                & (data_all.isvbf)
                ]
dd_PF = data_all[(data_all.cr_region == 'PF') & (data_all.Y == 2)
                & (data_all.isvbf)
                ]
dd_FF = data_all[(data_all.cr_region == 'FF') & (data_all.Y == 2)
                & (data_all.isvbf)
                ]
dd_FP.loc[:,'weight_bins'] = pd.cut( dd_FP.dipho_leadPt    ,wbins, labels= False)
dd_PF.loc[:,'weight_bins'] = pd.cut( dd_PF.dipho_subleadPt ,wbins, labels= False)
dd_FF.loc[:,'weight_bins'] = pd.cut( dd_FF.avg_pt          ,wbins, labels= False)
lumi = 41.86
dd_PF.weight = np.ones(dd_PF.shape[0])
for b in np.unique(dd_PF.weight_bins.values):
    for cat in np.unique(dd_PF.subl_eta_cat):
        w = dd_PF[(dd_PF.weight_bins == b) & (dd_PF.subl_eta_cat == cat)].weight.values 
        w = w * fake_PF[cat][b] * purity_PF[cat][b] / lumi
        dd_PF.loc[(dd_PF.weight_bins == b) & (dd_PF.subl_eta_cat == cat), 'weight'] = w
lumi = 41.86
dd_FP.weight = np.ones(dd_FP.shape[0])
for b in np.unique(dd_FP.weight_bins.values):
    for cat in np.unique(dd_FP.lead_eta_cat):
        w = dd_FP[(dd_FP.weight_bins == b) & (dd_FP.lead_eta_cat == cat)].weight.values 
        w = w * fake_FP[cat][b] * purity_FP[cat][b]/ lumi
        dd_FP.loc[(dd_FP.weight_bins == b) & (dd_FP.lead_eta_cat == cat), 'weight'] = w
lumi = 41.86
dd_FF.weight = np.ones(dd_FF.shape[0])
for b in np.unique(dd_FF.weight_bins.values):
    for cat in np.unique(dd_FF.eta_cat):
        w = dd_FF[(dd_FF.weight_bins == b) & (dd_FF.eta_cat == cat)].weight.values 
        w = -1 * w * fake_PF[cat[:-2]][b] * fake_FP[cat[2:]][b]/ lumi
        w = w * purity_PF[cat[:-2]][b] * purity_FP[cat[2:]][b] # this was commented before
#         w = -1 * w * fake_PF[cat[:-2]][b] / lumi
        dd_FF.loc[(dd_FF.weight_bins == b) & (dd_FF.eta_cat == cat), 'weight'] = w
import numpy as np
from scipy.optimize import minimize
def fit(var = 'dijet_Mjj',
        bins_ = np.linspace(0,1000,25), L=35.9 ) :
    hd, xd = np.histogram(data_region[(data_region.Y == 2)][var],
                          weights=data_region[(data_region.Y == 2)].weight,
                          bins=bins_)
    h_pf, xd = np.histogram(dd_PF[var], weights=dd_PF.weight * L, bins=bins_)
    h_fp, xd = np.histogram(dd_FP[var], weights=dd_FP.weight * L, bins=bins_)
    h_ff, xd = np.histogram(dd_FF[var], weights=dd_FF.weight * L, bins=bins_)
    
    pp, xd = np.histogram(data_SMMC[(data_SMMC['sample'] == 'dipho')][var],
                          weights=data_SMMC[(data_SMMC['sample'] == 'dipho')].weight * L,
                          bins=bins_)
    def residual(a,b):
        return divide(((pp + a*(h_pf + h_fp) + b*h_ff) - hd),np.sqrt(hd))**2
    def chi2(a,b):
        return  np.sum(divide(((pp + a*(h_pf + h_fp) + b*h_ff) - hd),
                              np.sqrt(np.abs(hd + (pp + a*(h_pf + h_fp) + b*h_ff))))**2 )
    chi2 = np.vectorize(chi2)
    def func_(x):
        return  np.sum(divide(hd - pp - x[0]*(h_pf + h_fp)- x[1]*h_ff,np.sqrt(hd))**2)/hd.shape[0]
    
    a = np.linspace(-2,2,101)
    b = np.linspace(-2,2,101)
    
    v_, w_ = np.meshgrid(a,b)
    
    x0 = np.array([1,1])
    res = minimize(func_, x0, method='nelder-mead', options={'xtol': 1e-8, 'disp': True})
    
    plt.figure(figsize=(4,4))
    plt.title(var)
    z_ = chi2(v_,w_)
    plt.contour(v_,w_, z_, np.linspace(np.min(z_), 2*np.min(z_), 3), cmap=plt.cm.RdYlGn_r)
    plt.xlabel(\
    plt.ylabel(\
    
    plt.axvline(1,ls='--',color='k')
    plt.axhline(1,ls='--',color='k')
    
    plt.axvline(res.x[0],ls='--',color='r')
    plt.axhline(res.x[1],ls='--',color='r')
    print res.x
    return  res.x
    
    
norm = fit(var = 'dipho_PToM'       , bins_ = np.linspace(0, 10,21), L=35.9)
dd_PF.weight = dd_PF.weight * norm[0]
dd_FP.weight = dd_FP.weight * norm[0]
dd_FF.weight = dd_FF.weight * norm[1]

data_driven = pd.concat([dd_PF, dd_FP, dd_FF])
# data_driven = pd.concat([dd_PF, dd_FF])
from matplotlib import gridspec

def varibale_data_driven(var = 'dijet_Mjj',var_label='$m_{jj}$ (GeV)', label='',
                         bins_ = np.linspace(0,1000,25),blind=False, title='VBF preselection',
                         mu=0.9, log = True, normed = False, L=35.9, xlog = False) :
    plt.figure(figsize=(6,9))
    gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1])
    #     ax1 = plt.subplot2grid((4,1), (0,0),rowspan=2)
    ax1 = plt.subplot(gs[0])
    
    hd, xd = np.histogram(data_region[(data_region.Y == 2)][var], normed=normed,
                          weights=data_region[(data_region.Y == 2)].weight,
                          bins=bins_)
    
    dd, xd = np.histogram(data_driven[(data_driven.Y == 2)][var], normed=normed,
                          weights=data_driven[(data_driven.Y == 2)].weight * L,
                          bins=bins_)
    gg, xd = np.histogram(data_SMMC[(data_SMMC['sample'] == 'dipho')][var], normed=normed,
                          weights=data_SMMC[(data_SMMC['sample'] == 'dipho')].weight * L,
                          bins=bins_)
    ee, xd = np.histogram(data_SMMC[(data_SMMC['sample'] == 'zee')][var], normed=normed,
                          weights=data_SMMC[(data_SMMC['sample'] == 'zee')].weight * L,
                          bins=bins_)
    
    mc, xd = np.histogram(data_SMMC[(data_SMMC['sample'] != 'ggh')][var], normed=normed,
                          weights=data_SMMC[(data_SMMC['sample'] != 'ggh')].weight * L,
                          bins=bins_)
    ggh, xd = np.histogram(data_SMMC[(data_SMMC['sample'] == 'ggh')][var], normed=normed,
                           weights=data_SMMC[(data_SMMC['sample'] == 'ggh')].weight * L * 10,
                           bins=bins_)
    vbf, xd = np.histogram(data_signal[(data_signal['sample'] == 'vbf')][var], normed=normed,
                           weights=data_signal[(data_signal['sample'] == 'vbf')].weight * L *100,
                           bins=bins_)

    ddw, xd = np.histogram(data_driven[(data_driven.Y == 2)][var], normed=normed,
                          weights=(data_driven[(data_driven.Y == 2)].weight)**2,
                          bins=bins_)
    mcw, xd = np.histogram(data_SMMC[(data_SMMC['sample'] != 'ggh')][var], normed=normed,
                          weights=(data_SMMC[(data_SMMC['sample'] != 'ggh')].weight * L)**2,
                          bins=bins_)
    ggw, xd = np.histogram(data_SMMC[(data_SMMC['sample'] == 'dipho')][var], normed=normed,
                          weights=(data_SMMC[(data_SMMC['sample'] == 'dipho')].weight * L)**2,
                          bins=bins_)
    eew, xd = np.histogram(data_SMMC[(data_SMMC['sample'] == 'zee')][var], normed=normed,
                          weights=(data_SMMC[(data_SMMC['sample'] == 'zee')].weight * L)**2,
                          bins=bins_)
    
    xd = np.array([(bins_[i+1]+ bins_[i])/2.0 for i in range(len(bins_)-1)])
    xe = np.array([(bins_[i+1]- bins_[i])/2.0 for i in range(len(bins_)-1)])
    
    ax1.annotate(title, xy=(1, 1.008), xycoords='axes fraction', fontsize=12,
                 horizontalalignment='right', verticalalignment='bottom')
    if blind:
        ax1.errorbar(xd[abs(xd-125)>10], hd[abs(xd-125)>10], xerr=xe[abs(xd-125)>10], 
                     yerr=np.sqrt(hd[abs(xd-125)>10]), fmt='.', c='black', 
                     markersize=11,capthick=0,
                     alpha=1.0, label='data')
    else:
        ax1.errorbar(xd, hd, xerr=xe, 
                     yerr=np.sqrt(hd), fmt='.', c='black', 
                     markersize=11,capthick=0,
                     alpha=1.0, label='data')
    plt.hist([data_SMMC[(data_SMMC['sample'] == 'dipho')][var], data_driven[var]], 
             normed=normed,
             weights=[
                 data_SMMC[(data_SMMC['sample'] == 'dipho')].weight * L,
                 data_driven.weight * L
             ],
             lw=1.2,bins=bins_, histtype='stepfilled', stacked=True,
             color=['#70C1B3','#C5D2DB'],label=['$\\gamma\\gamma$ (Sherpa)','$\\gamma$j + jj (data-driven)'])
    
    ax1.errorbar(xd, (dd + gg),xerr=xe, fmt='.', c='green', 
                 markersize=1,capthick=0,
                 alpha=1.0)
    ax1.errorbar(xd, mc,xerr=xe, fmt='.', c='blue', 
                 markersize=1,capthick=0,
                 alpha=1.0, label='MC')
    ax1.step(xd, mc, where='mid',color='blue')
    
    ax1.errorbar(xd, ggh,xerr=xe, fmt='.', c='#F9AD0B', 
                 markersize=1,capthick=0,
                 alpha=1.0, label='ggH $\\\\times 10$')
    ax1.step(xd, ggh, where='mid',color='#F9AD0B')
    
    ax1.errorbar(xd, vbf,xerr=xe, fmt='.', c='red', 
                 markersize=1,capthick=0,
                 alpha=1.0, label='VBF $\\\\times 10^2$')
    ax1.step(xd, vbf, where='mid',color='red')
    ax1.bar(xd,2*np.sqrt(ddw + ggw),
            bottom = (dd + gg)-np.sqrt(ddw + ggw),width= 2*xe, 
            color='red',alpha=0.3, zorder=12, 
            align='center',edgecolor='None',lw=0.0,
            label='stat')
    
    ax1.set_xlim([min(bins_), max(bins_)])
    
    if log : 
        ax1.set_ylim([0.1,1000*max([hd.max(),dd.max()])])
        ax1.set_yscale('log')
    else:
        ax1.set_ylim([0.1,1.8*max([hd.max(),dd.max()])])
    ax1.set_ylabel('events' ,ha='right')
   
    
    ax1.legend(fontsize=12, ncol=2)
    
    ax2 = plt.subplot(gs[1], sharex = ax1)
    #ax2 = plt.subplot2grid((4,1), (2,0),sharex=ax1)
    
    r_dd = divide(dd + gg + ee,hd)
    r_mc = divide(mc,hd)
    
    # Now the error propagation is for this ratio R=MC/Data is
    # (dR/R)^2 = (da/a)^2 + (db/b)^2
    # If db = 0 --> then (dR/R)^2 = (da/a)^2 --> dR = R (da/a)
    # 
   
    r_dd_err_mc =  r_dd * divide(np.sqrt(ddw + ggw), dd + gg)
    r_mc_err_mc =  r_mc * divide(np.sqrt(mcw), mc)
    
    r_dd_err =  r_dd*divide(np.sqrt(hd), dd + gg)
    r_mc_err =  r_mc*divide(np.sqrt(hd), mc)
    if blind:
        ax2.errorbar(xd[abs(xd-125)>10],r_dd[abs(xd-125)>10],xerr=xe[abs(xd-125)>10], 
                     yerr = r_mc_err[abs(xd-125)>10],
                     fmt='.', c='black', 
                     markersize=11,capthick=0,
                     alpha=1.0)
        ax2.axhline(y=1,color='Black',ls='--',lw=0.5)
        ax2.bar(xd,2*r_dd_err_mc,
            bottom = 1.0-r_dd_err_mc,width= 2*xe, 
            color='r',alpha=0.3, zorder=0, align='center',edgecolor='None',lw=0.0,
            label='stat')
        ax2.set_ylim([0,2])
    else:
        ax2.errorbar(xd,r_dd,xerr=xe, 
                     yerr = r_mc_err,
                     fmt='.', c='black', 
                     markersize=11,capthick=0,
                     alpha=1.0)
        ax2.axhline(y=1,color='Black',ls='--',lw=0.5)
        ax2.bar(xd,2*r_dd_err_mc,
            bottom = 1.0-r_dd_err_mc,width= 2*xe, 
            color='r',alpha=0.3, zorder=0, align='center',edgecolor='None',lw=0.0,
            label='stat')
        ax2.set_ylim([0,2])
    ax2.set_ylabel('$(MC + DD)/Data$')
    plt.setp(ax1.get_xticklabels(), visible=False)
    yticks = ax1.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)
    
    ax3 = plt.subplot(gs[2], sharex = ax1)
    if blind:
        ax3.errorbar(xd[abs(xd-125)>10], r_mc[abs(xd-125)>10] ,
                     xerr=xe[abs(xd-125)>10], 
                     yerr = r_mc_err[abs(xd-125)>10],
                     fmt='.', c='blue', 
                     markersize=11,capthick=0,
                     alpha=1.0)
        ax3.axhline(y=1,color='Black',ls='--',lw=0.5)
        ax3.bar(xd[abs(xd-125)>10],2*r_mc_err_mc[abs(xd-125)>10],
                bottom = 1.0 - r_mc_err_mc[abs(xd-125)>10],
                width= 2*xe[abs(xd-125)>10], 
                color='b',alpha=0.3, zorder=9, align='center',edgecolor='None',lw=0.0,
                label='stat')
        ax3.set_ylim([0,2])
        ax3.set_xlabel(var_label)
    else:
        ax3.errorbar(xd, r_mc ,xerr=xe, 
                     yerr = r_mc_err,
                     fmt='.', c='blue', 
                     markersize=11,capthick=0,
                     alpha=1.0)
        ax3.axhline(y=1,color='Black',ls='--',lw=0.5)
        ax3.bar(xd,2*r_mc_err_mc,
                bottom = 1.0 - r_mc_err_mc,width= 2*xe, 
                color='b',alpha=0.3, zorder=9, align='center',edgecolor='None',lw=0.0,
                label='stat')
        ax3.set_ylim([0,2])
        ax3.set_xlabel(var_label)
    ax3.set_xlim([bins_.min(), bins_.max()])
    ax3.set_ylabel('MC/Data')
    if xlog  : ax3.set_xscale('log')
    
    plt.setp(ax2.get_xticklabels(), visible=False)
    # remove last tick label for the second subplot
    yticks = ax3.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)
    plt.subplots_adjust(hspace=.0)
    
    #plt.savefig('data-driven-xcheck-%s-%s.pdf'% (label, var))
    plt.savefig('2017_fullVBFPreselection_withtightpujid_data-driven-xcheck-%s-%s.png'% (label, var))
# version = 'double-fake-0720-cut-0.9'
version = 'double-fake-0924'
varibale_data_driven(var = 'dipho_mass',var_label='$m_{\\gamma\\gamma}$ (GeV)', 
                     bins_ = np.linspace(100,180,41),log=False, blind=True,
                     label=version, L=35.9, mu=1.0)
varibale_data_driven(var = 'dipho_cosphi',
                     var_label='$\\cos\\Delta\\phi{\\gamma\\gamma}$', 
                     bins_ = np.linspace(0,1,21),log=False, mu=1.0,label=version)
varibale_data_driven(var = 'dijet_dipho_dphi_trunc',
                     var_label='$\\Delta\\phi{jj,\\gamma\\gamma}$', 
                     bins_ = np.linspace(0,3,16),log=False, mu=1.0,label=version)
varibale_data_driven(var = 'dijet_Mjj',
                     var_label='$m_{jj}$', 
                     bins_ = np.linspace(250,1000,31),log=False, mu=1.0,label=version)
varibale_data_driven(var = 'dijet_LeadJPt',
                     var_label='lead $p_{T}$', 
                     bins_ = np.linspace(40,440,11),log=False, mu=1.0,label=version)
varibale_data_driven(var = 'dijet_SubJPt',
                     var_label='sub-lead $p_{T}$', 
                     bins_ = np.linspace(30,430,11),log=False, mu=1.0,label=version)
varibale_data_driven(var = 'dipho_leadPt',
                     var_label='lead $p_{T}$', 
                     bins_ = np.linspace(0,400,11),log=False, mu=1.0,label=version)
varibale_data_driven(var = 'dipho_subleadPt',
                     var_label='sub-lead $p_{T}$', 
                     bins_ = np.linspace(0,200,11),log=False, mu=1.0,label=version)
varibale_data_driven(var = 'dipho_PToM',
                     var_label='$p_{T}^{\\gamma\\gamma}/m_{\\gamma\\gamma}$', 
                     bins_ = np.linspace(0,4,21),log=False, mu=1.0,label=version)
varibale_data_driven(var = 'n_jet_30',
                     var_label='$N_{jets} (p_{T}>30 GeV)$', 
                     bins_ = np.linspace(2,10,9),log=False, mu=1.0,label=version)
varibale_data_driven(var = 'leadPho_PToM',
                     var_label='$p_{T}^{\\gamma_1}/m_{\\gamma\\gamma}$', 
                     bins_ = np.linspace(0,4,11),log=False, mu=1.0,label=version)
varibale_data_driven(var = 'sublPho_PToM',
                     var_label='$p_{T}^{\\gamma_2}/m_{\\gamma\\gamma}$', 
                     bins_ = np.linspace(0,4,11),log=False, mu=1.0,label=version)
varibale_data_driven(var = 'dijet_abs_dEta',
                     var_label='${\\Delta}{\\eta_{jj}}$', 
                     bins_ = np.linspace(0,8,11),log=False, mu=1.0,label=version)
varibale_data_driven(var = 'dijet_minDRJetPho',
                     var_label='${\\Delta}{R_{min}}({\\gamma},j)$', 
                     bins_ = np.linspace(0.5,4,11),log=False, mu=1.0,label=version)
varibale_data_driven(var = 'dijet_centrality_gg',
                     var_label='$C_{\\gamma\\gamma}$', 
                     bins_ = np.linspace(0.0,0.8,11),log=False, mu=1.0,label=version)
## Save dataset
data_driven['sample'] = 'QCD'
data_driven.Y = 0
data_out = pd.concat([data_driven, data_region, data_SMMC, data_signal])
np.unique(data_out['sample'])
data_out.to_hdf('2017_Analysis_with_datadriven_PUJID_new_ptHjj_var.h5'   , 'results_table', mode='w', format='table')
