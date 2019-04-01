##### Importing libraries and preventing warnings from chained assignment
import matplotlib.pyplot as plt
import numpy  as np
import pandas as pd

#matplotlib inline
#plt.style.use('physics')

plt.rcParams['axes.grid'       ]  = False
plt.rcParams['xtick.labelsize' ]  = 14
plt.rcParams['ytick.labelsize' ]  = 14
plt.rcParams['axes.labelsize'  ]  = 14
plt.rcParams['legend.fancybox' ]  = False

pd.options.mode.chained_assignment = None

##### Reading the .h5 file without opening it
import os 
#os.path.exists('From_Seth_with_datadriven_generated_by_me.h5')
indata = pd.read_hdf('2017_Analysis_with_datadriven.h5')
# Training

##### Plotting histogram of categories
#define a frame after the photon ID cut
df_after_photon_id_cut = indata[
    (indata['pass_id'] == True  )]
categories = df_after_photon_id_cut["sample"].values

import collections
c = collections.Counter(categories)
#print(c)

print(c.keys())
print(c.values())

objects = c.keys()
y_pos = np.arange(len(objects))
number = c.values()
 
plt.bar(y_pos, number, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Number of samples')
plt.title('Category distribution')
#plt.savefig("categories.png") 

##### Can use the following to check the names of all the columns in the dataframe
#list(indata.columns.values) 
##### Checking what kind of data is in the sample column of the indata dataframe
###### What do these strings stand for?

my_list = indata["sample"].values
uniqueVals = indata["sample"].unique()
print(uniqueVals)



##### Adding a column to the dataframe
indata['dijet_centrality_gg'] = np.exp(-4*(indata.dijet_Zep/indata.dijet_abs_dEta)**2)
##### Defining a number of other data frames
##### df_bkgs is the indata dataframe but with all entries having 'QCD', 'dipho', 'ggh' in the sample column only
##### df_mc is the indata dataframe but with all entries having 'qcd', 'dipho', 'gjet' and 'ggh' in the sample column only
##### df_sign is the indata dataframe but with all entries having 'vbf' in the sample column only
##### df_data is the indata dataframe but with entries of 'data'  in the sample column only and the condition np.abs('dipho_mass')-125 > 10 
df_bkgs = indata[
    (indata['sample'] != 'data' ) & 
    (indata['sample'] != 'qcd'  ) & 
    (indata['sample'] != 'vbf'  ) &
    (indata['sample'] != 'gjet' ) & 
    (indata['sample'] != 'zee'  )
]
df_mc   = indata[
    (indata['sample'] != 'data') & 
    (indata['sample'] != 'vbf' ) &
    (indata['sample'] != 'zee' ) & 
    (indata['sample'] != 'QCD')
]
df_sign = indata[
    (indata['sample'] == 'vbf'  )
]
df_data = indata[
    (indata['sample'] == 'data')& 
    (np.abs(indata['dipho_mass'] - 125)>10)
]

df_ggh_and_dipho = indata[
    (indata['sample'] != 'data' ) & 
    (indata['sample'] != 'qcd'  ) & 
    (indata['sample'] != 'vbf'  ) &
    (indata['sample'] != 'gjet' ) & 
    (indata['sample'] != 'zee'  ) &
    (indata['sample'] != 'QCD'  )
]

### creating data frames to check the input variable distributions
df_SM_bkg = indata[
    (indata['sample'] == 'QCD'  ) |
    (indata['sample'] == 'dipho'  )]
    
df_ggh = indata[
    (indata['sample'] == 'ggh'  )]
df_vbf = indata[
    (indata['sample'] == 'vbf'  )]



def vbf_relax(data):
    return (
        (data["leadPho_PToM"       ]> (1/4.0))&
        (data["sublPho_PToM"       ]> (1/5.0))&
        (data["dijet_LeadJPt"      ]> 30     )& 
        (data["dijet_SubJPt"       ]> 20     )&
        (data["dijet_Mjj"          ]> 100    )&
        (data["dipho_mass"         ]> 100    )&
        (data["dipho_mass"         ]< 180    ))
def vbf_presel(data):
    return (
        (data["leadPho_PToM"       ]> (1/3.0))&
        (data["sublPho_PToM"       ]> (1/4.0))&
        (data["dijet_LeadJPt"      ]> 40     )& 
        (data["dijet_SubJPt"       ]> 30     )&
        (data["dijet_Mjj"          ]> 250    )&
        (data["dipho_mass"         ]> 100    )&
        (data["dipho_mass"         ]< 180    ))








##################################################

df_SM_bkg = df_SM_bkg[vbf_presel(df_SM_bkg)]
df_ggh = df_ggh[vbf_presel(df_ggh)]
df_vbf = df_vbf[vbf_presel(df_vbf)]
numpy_SM_bkg = df_SM_bkg['dijet_LeadJPt'].values
numpy_SM_bkg_weight = df_SM_bkg['weight'].values

numpy_ggh = df_ggh['dijet_LeadJPt'].values
numpy_ggh_weight = df_ggh['weight'].values

numpy_vbf = df_vbf['dijet_LeadJPt'].values
numpy_vbf_weight = df_vbf['weight'].values

#print(numpy_SM_bkg)
#print(numpy_SM_bkg_weight)
plt.figure(figsize=(6,6))
#plt.title('Transverse momentum of leading jet')
plt.hist(numpy_SM_bkg, bins=50,
             weights=numpy_SM_bkg_weight, 
             alpha=0.3, histtype='stepfilled', normed=1, color = 'blue', range = (40, 800), label = 'SM')
plt.hist(numpy_ggh, bins=50,
             weights=numpy_ggh_weight, 
             alpha=1, histtype='step',color = 'blue', normed=1, range = (40,800), label = 'ggh (125)')
plt.hist(numpy_vbf, bins=50,
             weights=numpy_vbf_weight, 
             alpha=0.3, histtype='stepfilled', normed=1, color = 'red', range = (40, 800), label = 'vbf (125)')
plt.legend()
plt.xticks([100,200,300,400,500,600,700,800])
plt.xlabel('$p_T(jet_1)$')
plt.ylabel('1/N dN/d($p_T(jet_1)$)')
plt.xlim([40,800])
plt.ylim([0,0.01])
plt.savefig('9-month-leading-jet-pT',bbox_inches = 'tight')
#plt.show()
######################################################


#####################################################
numpy_SM_bkg = df_SM_bkg['dijet_SubJPt'].values
numpy_SM_bkg_weight = df_SM_bkg['weight'].values

numpy_ggh = df_ggh['dijet_SubJPt'].values
numpy_ggh_weight = df_ggh['weight'].values

numpy_vbf = df_vbf['dijet_SubJPt'].values
numpy_vbf_weight = df_vbf['weight'].values

plt.figure(figsize=(6,6))
#plt.title('Transverse momentum of sub-leading jet')
plt.hist(numpy_SM_bkg, bins=50,
             weights=numpy_SM_bkg_weight, 
             alpha=0.3, histtype='stepfilled', normed=1, color = 'blue', range = (30, 500), label = 'SM')
plt.hist(numpy_ggh, bins=50,
             weights=numpy_ggh_weight, 
             alpha=1, histtype='step',color = 'blue', normed=1, range = (30,500), label = 'ggh (125)')
plt.hist(numpy_vbf, bins=50,
             weights=numpy_vbf_weight, 
             alpha=0.3, histtype='stepfilled', normed=1, color = 'red', range = (30, 500), label = 'vbf (125)')
plt.legend()
plt.xlabel('$p_T(jet_2)$')
plt.xticks([100,200,300,400,500])
plt.ylabel('1/N dN/d($p_T(jet_2)$)')
plt.xlim([30,500])
plt.ylim([0, 0.03])
plt.savefig('9-month-sublead-pT.png',bbox_inches = 'tight')

#plt.show()
##########################################################





#########################################################
numpy_SM_bkg = df_SM_bkg['dijet_abs_dEta'].values
numpy_SM_bkg_weight = df_SM_bkg['weight'].values

numpy_ggh = df_ggh['dijet_abs_dEta'].values
numpy_ggh_weight = df_ggh['weight'].values

numpy_vbf = df_vbf['dijet_abs_dEta'].values
numpy_vbf_weight = df_vbf['weight'].values

plt.figure(figsize=(6,6))
#plt.title('difference in pseudorapidity of the two jets')
plt.hist(numpy_SM_bkg, bins=50,
             weights=numpy_SM_bkg_weight, 
             alpha=0.3, histtype='stepfilled', normed=1, color = 'blue', range = (0, 8), label = 'SM')
plt.hist(numpy_ggh, bins=50,
             weights=numpy_ggh_weight, 
             alpha=1, histtype='step',color = 'blue', normed=1, range = (0,8), label = 'ggh (125)')
plt.hist(numpy_vbf, bins=50,
             weights=numpy_vbf_weight, 
             alpha=0.3, histtype='stepfilled', normed=1, color = 'red', range = (0, 8), label = 'vbf (125)')
plt.legend()
plt.xticks([1,2,3,4,5,6,7,8])
plt.xlabel('$\\Delta\\eta_{jj}$')
plt.ylabel('1/N dN/d($\\Delta\\eta_{jj}$)')

plt.xlim([0,8])
plt.ylim([0,0.35])
plt.savefig('9-month-diff-in-eta-jets.png',bbox_inches = 'tight')
#plt.show()



############################################################
numpy_SM_bkg = df_SM_bkg['dijet_Mjj'].values
numpy_SM_bkg_weight = df_SM_bkg['weight'].values

numpy_ggh = df_ggh['dijet_Mjj'].values
numpy_ggh_weight = df_ggh['weight'].values

numpy_vbf = df_vbf['dijet_Mjj'].values
numpy_vbf_weight = df_vbf['weight'].values

plt.figure(figsize=(6,6))
#plt.title('dijet invariant mass')
plt.hist(numpy_SM_bkg, bins=50,
             weights=numpy_SM_bkg_weight, 
             alpha=0.3, histtype='stepfilled', normed=1, color = 'blue', range = (0, 3500), label = 'SM')
plt.hist(numpy_ggh, bins=50,
             weights=numpy_ggh_weight, 
             alpha=1, histtype='step',color = 'blue', normed=1, range = (0,3500), label = 'ggh (125)')
plt.hist(numpy_vbf, bins=50,
             weights=numpy_vbf_weight, 
             alpha=0.3, histtype='stepfilled', normed=1, color = 'red', range = (0, 3500), label = 'vbf (125)')
plt.legend()
plt.xticks([500, 1000, 1500,2000,2500,3000,3500])
plt.xlabel('$m_{jj}$')
plt.ylabel('1/N dN/d($m_{jj}$)')
plt.xlim([250,3500])
plt.ylim([0,0.004])
plt.savefig('9-month-invariant-mass.png',bbox_inches = 'tight')

#plt.show()
############################################################





###############################################################
numpy_SM_bkg = df_SM_bkg['dijet_centrality_gg'].values
numpy_SM_bkg_weight = df_SM_bkg['weight'].values

numpy_ggh = df_ggh['dijet_centrality_gg'].values
numpy_ggh_weight = df_ggh['weight'].values

numpy_vbf = df_vbf['dijet_centrality_gg'].values
numpy_vbf_weight = df_vbf['weight'].values

plt.figure(figsize=(6,6))
#plt.title('dijet centrality variable')
plt.hist(numpy_SM_bkg, bins=50,
             weights=numpy_SM_bkg_weight, 
             alpha=0.3, histtype='stepfilled', normed=1, color = 'blue', range = (0, 1), label = 'SM')
plt.hist(numpy_ggh, bins=50,
             weights=numpy_ggh_weight, 
             alpha=1, histtype='step',color = 'blue', normed=1, range = (0,1), label = 'ggh (125)')
plt.hist(numpy_vbf, bins=50,
             weights=numpy_vbf_weight, 
             alpha=0.3, histtype='stepfilled', normed=1, color = 'red', range = (0, 1), label = 'vbf (125)')
plt.legend()

plt.xlabel('$C_{\\gamma\\gamma}$')
plt.ylabel('1/N dN/d($C_{\\gamma\\gamma}$)')

plt.xlim([0,1])
plt.ylim([0,12])
plt.savefig('9month-centrality.png',bbox_inches = 'tight' )
#plt.show()
###########################################################





##########################################################
numpy_SM_bkg = df_SM_bkg['dijet_dipho_dphi_trunc'].values
numpy_SM_bkg_weight = df_SM_bkg['weight'].values

numpy_ggh = df_ggh['dijet_dipho_dphi_trunc'].values
numpy_ggh_weight = df_ggh['weight'].values

numpy_vbf = df_vbf['dijet_dipho_dphi_trunc'].values
numpy_vbf_weight = df_vbf['weight'].values

plt.figure(figsize=(6,6))
#plt.title('difference between dijet and diphoton azimuth angle')
plt.hist(numpy_SM_bkg, bins=50,
             weights=numpy_SM_bkg_weight, 
             alpha=0.3, histtype='stepfilled', normed=1, color = 'blue', range = (0, 3), label = 'SM')
plt.hist(numpy_ggh, bins=50,
             weights=numpy_ggh_weight, 
             alpha=1, histtype='step',color = 'blue', normed=1, range = (0,3), label = 'ggh (125)')
plt.hist(numpy_vbf, bins=50,
             weights=numpy_vbf_weight, 
             alpha=0.3, histtype='stepfilled', normed=1, color = 'red', range = (0, 3), label = 'vbf (125)')
plt.legend()

plt.xlabel('$\\Delta\\phi(jj,\\gamma\\gamma)}$')
plt.ylabel('1/N dN/d($\\Delta\\phi(jj,\\gamma\\gamma}$)')


plt.xlim([0,3])
plt.ylim([0,12])
plt.savefig('9-month-dijet-dphoton-azimuth-diff.png', bbox_inches = 'tight')
plt.legend(loc ='upper left')
#plt.show()



################################################################
numpy_SM_bkg = df_SM_bkg['leadPho_PToM'].values
numpy_SM_bkg_weight = df_SM_bkg['weight'].values

numpy_ggh = df_ggh['leadPho_PToM'].values
numpy_ggh_weight = df_ggh['weight'].values

numpy_vbf = df_vbf['leadPho_PToM'].values
numpy_vbf_weight = df_vbf['weight'].values

plt.figure(figsize=(6,6))
#plt.title('transverse momentum of leading photon divided by diphoton invariant mass')
plt.hist(numpy_SM_bkg, bins=50,
             weights=numpy_SM_bkg_weight, 
             alpha=0.3, histtype='stepfilled', normed=1, color = 'blue', range = (0, 3.5), label = 'SM')
plt.hist(numpy_ggh, bins=50,
             weights=numpy_ggh_weight, 
             alpha=1, histtype='step',color = 'blue', normed=1, range = (0,3.5), label = 'ggh(125)')
plt.hist(numpy_vbf, bins=50,
             weights=numpy_vbf_weight, 
             alpha=0.3, histtype='stepfilled', normed=1, color = 'red', range = (0, 3.5), label = 'vbf(125)')
plt.legend()
plt.xticks([0.5,1.0,1.5,2.0,2.5,3.0,3.5])
plt.xlabel('$p_T(\\gamma_1)/m_{\\gamma\\gamma}$')
plt.ylabel('1/N dN/d($p_T(\\gamma_1)/m_{\\gamma\\gamma}$)')
plt.xlim([0.25,3.5])
plt.ylim([0,2.5])
plt.savefig('9-month-lead-pToM.png',bbox_inches = 'tight')

#plt.show()


################################################################

numpy_SM_bkg = df_SM_bkg['sublPho_PToM'].values
numpy_SM_bkg_weight = df_SM_bkg['weight'].values

numpy_ggh = df_ggh['sublPho_PToM'].values
numpy_ggh_weight = df_ggh['weight'].values

numpy_vbf = df_vbf['sublPho_PToM'].values
numpy_vbf_weight = df_vbf['weight'].values

plt.figure(figsize=(6,6))
#plt.title('transverse momentum of sub-leading photon divided by diphoton invariant mass')
plt.hist(numpy_SM_bkg, bins=50,
             weights=numpy_SM_bkg_weight, 
             alpha=0.3, histtype='stepfilled', normed=1, color = 'blue', range = (0.4, 1.6), label = 'SM')
plt.hist(numpy_ggh, bins=50,
             weights=numpy_ggh_weight, 
             alpha=1, histtype='step',color = 'blue', normed=1, range = (0.4,1.6), label = 'ggh (125)')
plt.hist(numpy_vbf, bins=50,
             weights=numpy_vbf_weight, 
             alpha=0.3, histtype='stepfilled', normed=1, color = 'red', range = (0.4, 1.6), label = 'vbf (125)')
plt.legend()
plt.xticks([0.4,0.6,0.8,1.0,1.2,1.4])
plt.xlabel('$p_T(\\gamma_2)/m_{\\gamma\\gamma}$')
plt.ylabel('1/N dN/d($p_T(\\gamma_2)/m_{\\gamma\\gamma}$)')
plt.xlim([0.4,1.6])
plt.ylim([0,10])
plt.savefig('9-month-sublead-pToM.png',bbox_inches = 'tight')

#plt.show()
############################################################




numpy_SM_bkg = df_SM_bkg['dijet_minDRJetPho'].values
numpy_SM_bkg_weight = df_SM_bkg['weight'].values

numpy_ggh = df_ggh['dijet_minDRJetPho'].values
numpy_ggh_weight = df_ggh['weight'].values

numpy_vbf = df_vbf['dijet_minDRJetPho'].values
numpy_vbf_weight = df_vbf['weight'].values

plt.figure(figsize=(6,6))
#plt.title('minimum distance between leading/subleading jet and leading/subleading photon')
plt.hist(numpy_SM_bkg, bins=50,
             weights=numpy_SM_bkg_weight, 
             alpha=0.3, histtype='stepfilled', normed=1, color = 'blue', range = (0.5, 4.0), label = 'SM')
plt.hist(numpy_ggh, bins=50,
             weights=numpy_ggh_weight, 
             alpha=1, histtype='step',color = 'blue', normed=1, range = (0.5,4.0), label = 'ggh (125)')
plt.hist(numpy_vbf, bins=50,
             weights=numpy_vbf_weight, 
             alpha=0.3, histtype='stepfilled', normed=1, color = 'red', range = (0.5, 4.0), label = 'vbf (125)')
plt.legend()
plt.xticks([0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0])
plt.xlabel('$\\Delta R_{min}(\\gamma,j)$')
plt.ylabel('1/N dN/d($\\Delta R_{min}(\\gamma,j)$)')
plt.xlim([0.5,4.0])
plt.ylim([0,0.7])
plt.savefig('9-month-Rmin.png',bbox_inches = 'tight')

#plt.show()








################################################################################
numpy_SM_bkg = df_SM_bkg['dijet_dphi'].values
numpy_SM_bkg_weight = df_SM_bkg['weight'].values

numpy_ggh = df_ggh['dijet_dphi'].values
numpy_ggh_weight = df_ggh['weight'].values

numpy_vbf = df_vbf['dijet_dphi'].values
numpy_vbf_weight = df_vbf['weight'].values

plt.figure(figsize=(6,6))
#plt.title('difference in azimuth angle between two leading jets')
plt.hist(numpy_SM_bkg, bins=50,
             weights=numpy_SM_bkg_weight, 
             alpha=0.3, histtype='stepfilled', normed=1, color = 'blue', range = (0, 3.5), label = 'SM')
plt.hist(numpy_ggh, bins=50,
             weights=numpy_ggh_weight, 
             alpha=1, histtype='step',color = 'blue', normed=1, range = (0,3.5), label = 'ggh (125)')
plt.hist(numpy_vbf, bins=50,
             weights=numpy_vbf_weight, 
             alpha=0.3, histtype='stepfilled', normed=1, color = 'red', range = (0,3.5), label = 'vbf (125)')
plt.legend()
plt.xticks([0.5,1.0,1.5,2.0,2.5,3.0])

plt.xlabel('$\\Delta \\phi_{jj}$')
plt.ylabel('1/N dN/d($\\Delta \\phi_{jj}$)')
plt.xlim([0,3.1])
plt.ylim([0,1.0])

plt.legend(loc = 'upper left')
plt.savefig('9-month-diff in azimuth-bet-two-jets.png',bbox_inches = 'tight' )


#plt.show()







######################################
numpy_SM_bkg = df_SM_bkg['dipho_mva'].values
numpy_SM_bkg_weight = df_SM_bkg['weight'].values

numpy_ggh = df_ggh['dipho_mva'].values
numpy_ggh_weight = df_ggh['weight'].values

numpy_vbf = df_vbf['dipho_mva'].values
numpy_vbf_weight = df_vbf['weight'].values

plt.figure(figsize=(6,6))
#plt.title('Diphoton MVA score')
plt.hist(numpy_SM_bkg, bins=50,
             weights=numpy_SM_bkg_weight, 
             alpha=0.3, histtype='stepfilled', normed=1, color = 'blue', range = (-1,1), label = 'SM')
plt.hist(numpy_ggh, bins=50,
             weights=numpy_ggh_weight, 
             alpha=1, histtype='step',color = 'blue', normed=1, range = (-1,1), label = 'ggh(125)')
plt.hist(numpy_vbf, bins=50,
             weights=numpy_vbf_weight, 
             alpha=0.3, histtype='stepfilled', normed=1, color = 'red', range = (-1,1), label = 'vbf(125)')
plt.legend(loc = 'lower right')

plt.savefig('output_diphoton_MVA_score.png')
plt.xlabel('diphoton MVA')




#plt.show()




##### A subset of the number of the columns in  the indata dataframe will be used for training
##### Try changing the features that are used i.e see if using more features gives better results
##### The functions for preselecting the vbf events are defined (full and relaxed selections). These are pretty self-explanatory
import numpy  as np
import pandas as pd
from sklearn.preprocessing import label_binarize
# == sklearn ==
from sklearn.model_selection  import train_test_split
from sklearn.model_selection  import cross_val_score

_features_ = [u'dijet_LeadJPt'  ,u'dijet_SubJPt', 
              u'dijet_abs_dEta' ,u'dijet_Mjj', 
              u'dijet_centrality_gg',u'dijet_dipho_dphi_trunc',
              u'dijet_dphi'     ,u'dijet_minDRJetPho',
              u'leadPho_PToM'   ,u'sublPho_PToM']


def vbf_presel(data):
    return (
        (data["leadPho_PToM"       ]> (1/3.0))&
        (data["sublPho_PToM"       ]> (1/4.0))&
        (data["dijet_LeadJPt"      ]> 40     )& 
        (data["dijet_SubJPt"       ]> 30     )&
        (data["dijet_Mjj"          ]> 250    )&
        (data["dipho_mass"         ]> 100    )&
        (data["dipho_mass"         ]< 180    ))

def vbf_relax(data):
    return (
        (data["leadPho_PToM"       ]> (1/4.0))&
        (data["sublPho_PToM"       ]> (1/5.0))&
        (data["dijet_LeadJPt"      ]> 30     )& 
        (data["dijet_SubJPt"       ]> 20     )&
        (data["dijet_Mjj"          ]> 100    )&
        (data["dipho_mass"         ]> 100    )&
        (data["dipho_mass"         ]< 180    ))


#####  Here the vbf preselection function (full) is applied to all the defined dataframes
#The numpy array D will contain 'vbf' samples from df_sign and 'QCD', 'dipho' and 'ggH' samples from df_bkgs
#(only feature columns included)
#The numpy array Y will be the label array of binary labels, 1 for 'vbf' events and 0 for bkg events i.e for 'QCD', 'dipho', 'ggH'
#The numpy array W will contain values in the weight column of df_sign and df_bkg
#The numpy array I will contain values in the sample column of df_sign (this should be 'vbf') and values in the 'sample' column of df_bkg (i.e a mixture of 'QCD', 'dipho' and 'ggH'.


#The same is then done for df_sign and df_data to get the X_data, Y_data, W_data, I_data, O_data

#choose how to train

Train_datadriven_test_datadriven = True
Train_MC_test_MC = False
Train_MSB_test_datadriven = False
Train_MSB_test_MC = False
Train_gghanddipho_test_datadriven = False
Train_gghanddipho_test_mc = False



if (Train_datadriven_test_datadriven or Train_MSB_test_datadriven or Train_gghanddipho_test_datadriven):
   df_bkgs = df_bkgs[vbf_presel(df_bkgs)]
   df_mc   = df_mc  [vbf_presel(df_mc  )]
   df_sign = df_sign[vbf_presel(df_sign)]
   df_data = df_data[vbf_presel(df_data)]


if (Train_datadriven_test_datadriven or Train_MSB_test_datadriven or Train_gghanddipho_test_datadriven):  
   D  =  np.concatenate((df_sign[_features_],df_bkgs[_features_]))
   Y  =  np.concatenate((np.ones(df_sign.shape[0]),np.zeros(df_bkgs.shape[0])))
   W  =  np.concatenate((df_sign['weight'],df_bkgs['weight']))
   I  =  np.concatenate((df_sign['sample'],df_bkgs['sample']))
   O  =  np.concatenate((df_sign['dipho_mass'],df_bkgs['dipho_mass']))

if (Train_datadriven_test_datadriven or Train_MC_test_MC or Train_MSB_test_datadriven or Train_MSB_test_MC):

   X_data  =  np.concatenate((df_sign[_features_],df_data[_features_]))
   Y_data  =  np.concatenate((np.ones(df_sign.shape[0]),np.zeros(df_data.shape[0])))
   W_data  =  np.concatenate((df_sign['weight'],df_data['weight']))
   I_data  =  np.concatenate((df_sign['sample'],df_data['sample']))
   O_data  =  np.concatenate((df_sign['dipho_mass'],df_data['dipho_mass']))




if (Train_MC_test_MC or Train_MSB_test_MC or Train_gghanddipho_test_mc ):
    df_bkgs = df_bkgs[vbf_relax(df_bkgs)]
    df_mc   = df_mc  [vbf_relax(df_mc  )]
    df_sign = df_sign[vbf_relax(df_sign)]
    df_data = df_data[vbf_relax(df_data)]




if (Train_MC_test_MC or Train_MSB_test_MC or Train_gghanddipho_test_mc):

    D  =  np.concatenate((df_sign[_features_],df_mc[_features_]))
    Y  =  np.concatenate((np.ones(df_sign.shape[0]),np.zeros(df_mc.shape[0])))
    W  =  np.concatenate((df_sign['weight'],df_mc['weight']))
    I  =  np.concatenate((df_sign['sample'],df_mc['sample']))
    O  =  np.concatenate((df_sign['dipho_mass'],df_mc['dipho_mass']))



if(Train_gghanddipho_test_datadriven):
    

    X_data  =  np.concatenate((df_sign[_features_],df_ggh_and_dipho[_features_]))
    Y_data  =  np.concatenate((np.ones(df_sign.shape[0]),np.zeros(df_ggh_and_dipho.shape[0])))
    W_data  =  np.concatenate((df_sign['weight'],df_ggh_and_dipho['weight']))
    I_data  =  np.concatenate((df_sign['sample'],df_ggh_and_dipho['sample']))
    O_data  =  np.concatenate((df_sign['dipho_mass'],df_ggh_and_dipho['dipho_mass']))


if(Train_gghanddipho_test_mc):
   
   X_data  =  np.concatenate((df_sign[_features_],df_ggh_and_dipho[_features_]))
   Y_data  =  np.concatenate((np.ones(df_sign.shape[0]),np.zeros(df_ggh_and_dipho.shape[0])))
   W_data  =  np.concatenate((df_sign['weight'],df_ggh_and_dipho['weight']))
   I_data  =  np.concatenate((df_sign['sample'],df_ggh_and_dipho['sample']))
   O_data  =  np.concatenate((df_sign['dipho_mass'],df_ggh_and_dipho['dipho_mass']))





from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score

np.random.seed(42)

if (Train_datadriven_test_datadriven or Train_MC_test_MC):
    (
        X_train, X_valid, 
        Y_train, Y_valid,
        W_train, W_valid,
        I_train, I_valid,
        O_train, O_valid
    
        ) = train_test_split(
        D, Y, W, I, O,
        test_size=0.8,  # 0.2 was thedefault
        random_state=17
        )

if (Train_MSB_test_datadriven or Train_MSB_test_MC or Train_gghanddipho_test_datadriven or Train_gghanddipho_test_mc):
    (
        X_train, X_valid,
        Y_train, Y_valid,
        W_train, W_valid,
        I_train, I_valid,
        O_train, O_valid

        ) = train_test_split(
        X_data, Y_data, W_data, I_data, O_data,
        test_size=0.8,  # 0.2 was thedefault
        random_state=17
        )





W_train = W_train * (W.sum()/W_train.sum())
W_valid = W_valid * (W.sum()/W_valid.sum())


import collections 
print "number of classes (samples) inside the dataset ... train (validation)"
for p in collections.Counter(I):
    print "%10s nevent = %10.2f [%10.2f]" % ( p, W_train[I_train==p].shape[0], W_valid[I_valid==p].shape[0])



def normalize_weights(weights, classes):
    weights_ = np.copy(weights)
    for cl in np.unique(classes):
        weights_[classes==cl] = weights_[classes==cl]/np.sum(weights_[classes==cl])
    return weights_

weights_train = {}

weights_train['target'] = normalize_weights(W_train,Y_train)
from sklearn.ensemble   import GradientBoostingClassifier as GBC
from pprint import pprint




classifier = {
    "moriond" : GBC(
        learning_rate=0.1,
        max_depth=5,
        max_features='auto', 
        min_samples_leaf=0.001, 
        min_samples_split=2,
        n_estimators=100,
        presort='auto', 
        subsample=0.5, 
        verbose=1,
        warm_start=False)
}
pprint(classifier)

from sklearn.base import clone

clfs = {}
for i in weights_train:
    for c in classifier:
        if 'xsec' in i and 'new' in c: continue
        print i +'-'+ c
        clfs[i+'-'+c]   = clone(classifier[c])


clf_info = GBC(
        learning_rate=0.1,
        max_depth=5,
        max_features='auto', 
        min_samples_leaf=0.001, 
        min_samples_split=2,
        n_estimators=100,
        presort='auto', 
        subsample=0.5, 
        verbose=1,
        warm_start=False)
#print(clf_info.n_estimators)
##### changing from labelled set to array
print(weights_train)
print(weights_train[i.split('-')[0]])
print(clfs.items())

if (Train_datadriven_test_datadriven):
   rocs_valid_Y_ggh = Y_valid[(I_valid=='ggh')| (I_valid=='vbf')]
   rocs_valid_W_ggh = W_valid[(I_valid=='ggh')| (I_valid=='vbf')]
   X_valid_ggh = X_valid[(I_valid=='ggh')| (I_valid=='vbf')]

   rocs_valid_Y_dipho = Y_valid[(I_valid=='dipho')| (I_valid=='vbf')]
   rocs_valid_W_dipho = W_valid[(I_valid=='dipho')| (I_valid=='vbf')]
   X_valid_dipho = X_valid[(I_valid=='dipho')| (I_valid=='vbf')]

   rocs_valid_Y_QCD = Y_valid[(I_valid=='QCD')| (I_valid=='vbf')]
   rocs_valid_W_QCD = W_valid[(I_valid=='QCD')| (I_valid=='vbf')]
   X_valid_QCD = X_valid[(I_valid=='QCD')| (I_valid=='vbf')]


if (Train_MC_test_MC):
   rocs_valid_Y_ggh = Y_valid[(I_valid=='ggh')| (I_valid=='vbf')]
   rocs_valid_W_ggh = W_valid[(I_valid=='ggh')| (I_valid=='vbf')]
   X_valid_ggh = X_valid[(I_valid=='ggh')| (I_valid=='vbf')]

   rocs_valid_Y_dipho = Y_valid[(I_valid=='dipho')| (I_valid=='vbf')]
   rocs_valid_W_dipho = W_valid[(I_valid=='dipho')| (I_valid=='vbf')]
   X_valid_dipho = X_valid[(I_valid=='dipho')| (I_valid=='vbf')]

   rocs_valid_Y_qcd = Y_valid[(I_valid=='qcd')| (I_valid=='vbf')]
   rocs_valid_W_qcd = W_valid[(I_valid=='qcd')| (I_valid=='vbf')]
   X_valid_qcd = X_valid[(I_valid=='qcd')| (I_valid=='vbf')]

   rocs_valid_Y_gjet = Y_valid[(I_valid=='gjet')| (I_valid=='vbf')]
   rocs_valid_W_gjet = W_valid[(I_valid=='gjet')| (I_valid=='vbf')]
   X_valid_gjet = X_valid[(I_valid=='gjet')| (I_valid=='vbf')]


if (Train_MSB_test_datadriven):
   rocs_valid_Y_ggh = Y[(I=='ggh')| (I=='vbf')]
   rocs_valid_W_ggh = W[(I=='ggh')| (I=='vbf')]
   X_valid_ggh = D[(I=='ggh')| (I=='vbf')]

   rocs_valid_Y_dipho = Y[(I=='dipho')| (I=='vbf')]
   rocs_valid_W_dipho = W[(I=='dipho')| (I=='vbf')]
   X_valid_dipho = D[(I=='dipho')| (I=='vbf')]

   rocs_valid_Y_QCD = Y[(I=='QCD')| (I=='vbf')]
   rocs_valid_W_QCD = W[(I=='QCD')| (I=='vbf')]
   X_valid_QCD = D[(I=='QCD')| (I=='vbf')]


if(Train_MSB_test_MC):
    rocs_valid_Y_ggh = Y[(I=='ggh')| (I=='vbf')]
    rocs_valid_W_ggh = W[(I=='ggh')| (I=='vbf')]
    X_valid_ggh = D[(I=='ggh')| (I=='vbf')]

    rocs_valid_Y_dipho = Y[(I=='dipho')| (I=='vbf')]
    rocs_valid_W_dipho = W[(I=='dipho')| (I=='vbf')]
    X_valid_dipho = D[(I=='dipho')| (I=='vbf')]

    rocs_valid_Y_qcd = Y[(I=='qcd')| (I=='vbf')]
    rocs_valid_W_qcd = W[(I=='qcd')| (I=='vbf')]
    X_valid_qcd = D[(I=='qcd')| (I=='vbf')]

    rocs_valid_Y_gjet = Y[(I=='gjet')| (I=='vbf')]
    rocs_valid_W_gjet = W[(I=='gjet')| (I=='vbf')]
    X_valid_gjet = D[(I=='gjet')| (I=='vbf')]


if(Train_gghanddipho_test_datadriven):
  rocs_valid_Y_ggh = Y[(I=='ggh')| (I=='vbf')]
  rocs_valid_W_ggh = W[(I=='ggh')| (I=='vbf')]
  X_valid_ggh = D[(I=='ggh')| (I=='vbf')]

  rocs_valid_Y_dipho = Y[(I=='dipho')| (I=='vbf')]
  rocs_valid_W_dipho = W[(I=='dipho')| (I=='vbf')]
  X_valid_dipho = D[(I=='dipho')| (I=='vbf')]

  rocs_valid_Y_QCD = Y[(I=='QCD')| (I=='vbf')]
  rocs_valid_W_QCD = W[(I=='QCD')| (I=='vbf')]
  X_valid_QCD = D[(I=='QCD')| (I=='vbf')]


if(Train_gghanddipho_test_mc):
  rocs_valid_Y_ggh = Y[(I=='ggh')| (I=='vbf')]
  rocs_valid_W_ggh = W[(I=='ggh')| (I=='vbf')]
  X_valid_ggh = D[(I=='ggh')| (I=='vbf')]

  rocs_valid_Y_dipho = Y[(I=='dipho')| (I=='vbf')]
  rocs_valid_W_dipho = W[(I=='dipho')| (I=='vbf')]
  X_valid_dipho = D[(I=='dipho')| (I=='vbf')]

  rocs_valid_Y_qcd = Y[(I=='qcd')| (I=='vbf')]
  rocs_valid_W_qcd = W[(I=='qcd')| (I=='vbf')]
  X_valid_qcd = D[(I=='qcd')| (I=='vbf')]

  rocs_valid_Y_gjet = Y[(I=='gjet')| (I=='vbf')]
  rocs_valid_W_gjet = W[(I=='gjet')| (I=='vbf')]
  X_valid_gjet = D[(I=='gjet')| (I=='vbf')]


from sklearn.metrics import roc_curve, auc, roc_auc_score
rocs = {}
prob = {}
cfun = {}

rocs_valid = {}
prob_valid = {}
cfun_valid = {}


if (Train_datadriven_test_datadriven or Train_MSB_test_datadriven or Train_gghanddipho_test_datadriven):
   rocs_valid_ggh = {}
   rocs_valid_dipho = {} 
   rocs_valid_QCD = {}
   prob_valid_ggh = {}
   prob_valid_dipho = {}
   prob_valid_QCD = {}


if (Train_MC_test_MC or Train_MSB_test_MC or Train_gghanddipho_test_mc):
   rocs_valid_ggh = {}
   rocs_valid_dipho = {}
   rocs_valid_qcd = {}
   rocs_valid_gjet = {}
   prob_valid_ggh = {}
   prob_valid_dipho = {}
   prob_valid_qcd = {}
   prob_valid_gjet = {}



for i, c in clfs.items():
    print ' -- training : ', i
    c.fit ( X_train, Y_train, sample_weight= weights_train[i.split('-')[0]])
    
    prob[i] = c.predict_proba(X_train)[:,1]
    rocs[i] = roc_curve( Y_train,prob[i],sample_weight=W_train)

    prob_valid[i] = c.predict_proba(X_valid)[:,1]
    rocs_valid[i] = roc_curve( Y_valid,prob_valid[i],sample_weight=W_valid)


    if (Train_datadriven_test_datadriven or Train_MSB_test_datadriven or Train_gghanddipho_test_datadriven):
        prob_valid_ggh[i] = c.predict_proba(X_valid_ggh)[:,1]
        rocs_valid_ggh[i] = roc_curve(rocs_valid_Y_ggh,prob_valid_ggh[i],sample_weight=rocs_valid_W_ggh)
    
        prob_valid_dipho[i] = c.predict_proba(X_valid_dipho)[:,1]
        rocs_valid_dipho[i] = roc_curve(rocs_valid_Y_dipho,prob_valid_dipho[i],sample_weight=rocs_valid_W_dipho)
    
        prob_valid_QCD[i] = c.predict_proba(X_valid_QCD)[:,1]
        rocs_valid_QCD[i] = roc_curve(rocs_valid_Y_QCD,prob_valid_QCD[i],sample_weight=rocs_valid_W_QCD)
    
    if (Train_MC_test_MC or Train_MSB_test_MC or Train_gghanddipho_test_mc):
       prob_valid_ggh[i] = c.predict_proba(X_valid_ggh)[:,1]
       rocs_valid_ggh[i] = roc_curve(rocs_valid_Y_ggh,prob_valid_ggh[i],sample_weight=rocs_valid_W_ggh)
    
       prob_valid_dipho[i] = c.predict_proba(X_valid_dipho)[:,1]
       rocs_valid_dipho[i] = roc_curve(rocs_valid_Y_dipho,prob_valid_dipho[i],sample_weight=rocs_valid_W_dipho)
    
       prob_valid_qcd[i] = c.predict_proba(X_valid_qcd)[:,1]
       rocs_valid_qcd[i] = roc_curve(rocs_valid_Y_qcd,prob_valid_qcd[i],sample_weight=rocs_valid_W_qcd)
    
       prob_valid_gjet[i] = c.predict_proba(X_valid_gjet)[:,1]
       rocs_valid_gjet[i] = roc_curve(rocs_valid_Y_gjet,prob_valid_gjet[i],sample_weight=rocs_valid_W_gjet)






from sklearn.externals import joblib
joblib.dump(clfs['target-moriond'], 'Classifier.pkl')
def evaluate_sklearn(cls, vals, coef=1):
    scale = 1.0 / cls.n_estimators
    ret = np.zeros(vals.shape[0])

    learning_rate = cls.learning_rate
    for itree, t in enumerate(cls.estimators_[:, 0]):
        r = t.predict(vals)
        ret += r * scale
    return 2.0/(1.0 + np.exp(-coef/learning_rate * ret)) - 1
for i in clfs.keys():
    tmva_train = evaluate_sklearn(clfs[i],X_train)
    tmva_valid = evaluate_sklearn(clfs[i],X_valid)
    plt.figure(figsize=(5,5))
    plt.title(i)
    plt.hist(tmva_train[(Y_train<0.5)], bins=50,
             weights=W_train[(Y_train<0.5)], 
             range=[-1,1],
             alpha=0.5, histtype='stepfilled', normed=1, label = 'bkg')
    plt.hist(tmva_train[(Y_train>0.5)], bins=50,
             weights=W_train[(Y_train>0.5)], 
             range=[-1,1],
             alpha=0.5, histtype='stepfilled', normed=1, label = 'VBF')

    plt.hist(tmva_train[I_train == 'ggh'], bins=50,
             weights=W_train[I_train == 'ggh'], 
             range=[-1,1],
             alpha=0.5, histtype='step', color = 'blue',lw=1.2, normed=1, label = 'ggh')
    plt.legend()
    plt.savefig('output_dijet_MVA_score.png')
    #plt.show()



from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics           import roc_curve, auc, roc_auc_score
def plot_rocs(rocs = {}, dump=False, range=[[0,1],[0,1]], label='', title='', rocs_train=None):
    plt.figure(figsize=(5,4.5))
    for k, spine in plt.gca().spines.items():
        spine.set_zorder(10)
    plt.gca().xaxis.grid(which='major', color='0.7' , linestyle='--',dashes=(5,1),zorder=0)
    plt.gca().yaxis.grid(which='major', color='0.7' , linestyle='--',dashes=(5,1),zorder=0)
    print '--- draw some roc curve with scores : '
    for name,roc in rocs.items():
        fpr, tpr, thr = roc
        roc_auc_ = auc(fpr, tpr, reorder=True)
        print "%20s %1.4f" % ( name, roc_auc_ )
        plt.plot(fpr, tpr, label=name+'(area = %0.4f)'%(roc_auc_), zorder=5, lw=1.2)
    if rocs_train is not None : 
        for name,roc in rocs_train.items():
            fpr, tpr, thr = roc
            roc_auc_ = auc(fpr, tpr, reorder=True)
            print "%20s %1.4f" % ( name, roc_auc_ )
            plt.plot(fpr, tpr, label=name+' train (area = %0.4f)'%(roc_auc_), zorder=5, lw=1.2, ls='--')
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck', zorder=5)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title (title)
    plt.legend(loc='best')
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.grid()
    plt.xlim(range[0])
    plt.ylim(range[1])
    plt.tight_layout()
    plt.savefig('roc_'+label+'.pdf')
    #plt.show()

####X-CHECK PLOTS

if (Train_datadriven_test_datadriven):
    plot_rocs(rocs_valid, label='trained on datadriven-tested on data-driven', title = 'ROC for rejection of all backgrounds')
    plot_rocs(rocs_valid_ggh, label='trained on datadriven-tested on data-driven-xcheck2', title = 'ROC for ggh rejection')
    plot_rocs(rocs_valid_dipho, label='trained on datadriven-tested on data-driven-xcheck3', title = 'ROC for dipho rejection')
    plot_rocs(rocs_valid_QCD, label='trained on datadriven-tested on data-driven-xcheck4', title = 'ROC for QCD rejection')


if (Train_MC_test_MC):
   plot_rocs(rocs_valid, label='trained on MC, tested on MC-xcheck1', title = 'ROC for all backgrounds')
   plot_rocs(rocs_valid_ggh, label='trained on MC, tested on MC-xcheck2', title = 'ROC for ggh rejection')
   plot_rocs(rocs_valid_dipho, label='trained on MC, tested on MC-xcheck3', title = 'ROC for dipho rejection')
   plot_rocs(rocs_valid_qcd, label='trained on MC, tested on MC-xcheck4', title = 'ROC for qcd rejection')
   plot_rocs(rocs_valid_gjet, label='trained on MC, tested on MC-xcheck5', title = 'ROC for gjet rejection')


if (Train_MSB_test_datadriven):
   plot_rocs(rocs_valid, label='ROC_XCHECK1_full_VBF_preselection_trained_on_MSB_tested_on_datadriven', title = 'ROC for all backgrounds')
   plot_rocs(rocs_valid_ggh, label='ROC_XCHECK2_full_VBF_preselection_trained_on_MSB_tested_on_datadriven', title = 'ROC for ggh rejection')
   plot_rocs(rocs_valid_dipho, label='ROC_XCHECK3_full_VBF_preselection_trained_on_MSB_tested_on_datadriven', title = 'ROC for dipho rejection')
   plot_rocs(rocs_valid_QCD, label='ROC_XCHECK4_full_VBF_preselection_trained_on_MSB_tested_on_datadriven', title = 'ROC for QCD rejection')


if(Train_MSB_test_MC):
   plot_rocs(rocs_valid, label='ROC_XCHECK1_relaxed_VBF_preselection_trained_on_MSB_tested_on_MC', title = 'ROC for all backgrounds')
   plot_rocs(rocs_valid_ggh, label='ROC_XCHECK2_relaxed_VBF_preselection_trained_on_MSB_tested_on_MC', title = 'ROC for ggh rejection')
   plot_rocs(rocs_valid_dipho, label='ROC_XCHECK3_relaxed_VBF_preselection_trained_on_MSB_tested_on_MC', title = 'ROC for dipho rejection')
   plot_rocs(rocs_valid_qcd, label='ROC_XCHECK4_relaxed_VBF_preselection_trained_on_MSB_tested_on_MC', title = 'ROC for qcd rejection')
   plot_rocs(rocs_valid_gjet, label='ROC_XCHECK5_relaxed_VBF_preselection_trained_on_MSB_tested_on_MC', title = 'ROC for gjet rejection')

if(Train_gghanddipho_test_mc):
    plot_rocs(rocs_valid, label='ROC_XCHECK1_relax_VBF_preselection_trained_on_ggh_and_dipho_tested_on_mc', title = 'ROC for all backgrounds')
    plot_rocs(rocs_valid_ggh, label='ROC_XCHECK2_relax_VBF_preselection_trained_on_ggh_and_dipho_tested_on_mc', title = 'ROC for ggh rejection')
    plot_rocs(rocs_valid_dipho, label='ROC_XCHECK3_relax_VBF_preselection_trained_on_ggh_and_dipho_tested_on_mc', title = 'ROC for dipho rejection')
    plot_rocs(rocs_valid_qcd, label='ROC_XCHECK4_relax_VBF_preselection_trained_on_ggh_and_dipho_tested_on_mc', title = 'ROC for jet-jet rejection')  
    plot_rocs(rocs_valid_gjet, label='ROC_XCHECK5_relax_VBF_preselection_trained_on_ggh_and_dipho_tested_on_mc', title = 'ROC for gjet rejection')



plot_rocs(rocs_valid, label='cross-check', title = 'cross-check')
def compare_train_test(clf,x_train,y_train,w_train,x_test,y_test,w_test, bins=100, label=''):
    fig = plt.figure(figsize=(5,5))
    plt.title(label)
    decisions = []
    weight    = []
    print clf
    for x,y,w in ((x_train, y_train, w_train), (x_test, y_test, w_test)):
        print x.shape
        #d1 = clf.predict_proba(x[y>0.5])[:,1]
        #d2 = clf.predict_proba(x[y<0.5])[:,1].ravel()
        d1 = evaluate_sklearn(clf,x[y>0.5])
        d2 = evaluate_sklearn(clf,x[y<0.5])
        w1 = w[y>0.5]
        w2 = w[y<0.5]
        decisions += [d1, d2]
        weight    += [w1, w2]
        
    low  = min(np.min(d) for d in decisions)
    high = max(np.max(d) for d in decisions)
    low_high = (low,high)
    
    plt.hist(decisions[0],
             color='r', alpha=0.5, range=low_high, bins=bins,
             histtype='stepfilled', normed=True,
             weights = weight[0], 
             label='S (train)')
    plt.hist(decisions[1],
             color='b', alpha=0.5, range=low_high, bins=bins,
             histtype='stepfilled', normed=True,
             weights = weight[1], 
             label='B (train)')

    hist, bins = np.histogram(decisions[2],
                              bins=bins, range=low_high, normed=True, weights = weight[2] )
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale
    
    width = (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.errorbar(center, hist, yerr=err, fmt='.', c='r', label='S (test)', markersize=8,capthick=0)
    
    hist, bins = np.histogram(decisions[3],
                              bins=bins, range=low_high, normed=True, weights = weight[3])
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale

    plt.errorbar(center, hist, yerr=err, fmt='.', c='b', label='B (test)', markersize=8,capthick=0)
    plt.xlabel("BDT output")
    plt.ylabel("Arbitrary units")
    plt.legend(loc='best')
    plt.ylim([0.01, 2*max(hist)])
    plt.savefig('overtrain_%s.pdf' % label)
    #plt.show()


for i, c in clfs.items():
    print " --- ", i
    compare_train_test(c,
                       X_train,Y_train,W_train, 
                       X_valid,Y_valid,W_valid, label=i)


### Adding a few lines to extract the xml file
import converter as con
xml_file_name = 'DataDriven_xml.xml'
con.convert_bdt__Grad(bdt_clf = clfs['target-moriond'], input_var_list = _features_, tmva_outfile_xml = xml_file_name, X_train= X_train)
