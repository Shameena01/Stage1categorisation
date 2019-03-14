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
Let load the data from the root file using the small libray that you can find in here [1]. 
This read a jason file that contains the features and the samples that will be used in the training. 
# Use wget to get the example file
! wget https://yhaddad.web.cern.ch/yhaddad/VBF/misc/hgg-double-fake-trees-training-2017.h5
##### Reading the .h5 file without opening it
import os 
os.path.exists('From_Seth_without_datadriven_generated_by_me.h5')
indata = pd.read_hdf('From_Seth_without_datadriven_generated_by_me.h5')
# Training
#### Displaying all columns
pd.set_option(\
pd.get_option(\
##### .head(n=5) reads the first n rows, default is 5
indata.head()
##### Plotting histogram of categories
categories = indata[\
#print(categories)
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
 
plt.show()
##### Can use the following to check the names of all the columns in the dataframe
list(indata.columns.values) 
##### Checking what kind of data is in the sample column of the indata dataframe
###### What do these strings stand for?
my_list = indata[\
print(my_list)
uniqueVals = indata[\
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
#####  Checking what is in the dataframes defined above
my_list2 = df_bkgs[\
print(my_list2)
uniqueVals = df_bkgs[\
print(uniqueVals)
my_list3 = df_mc[\
print(my_list3)
uniqueVals = df_mc[\
print(uniqueVals)
my_list4 = df_sign[\
print(my_list4)
uniqueVals = df_sign[\
print(uniqueVals)
my_list5 = df_data[\
print(my_list5)
uniqueVals = df_data[\
print(uniqueVals)
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
        (data[\
        (data[\
        (data[\
        (data[\
        (data[\
        (data[\
        (data[\

def vbf_relax(data):
    return (
        (data[\
        (data[\
        (data[\
        (data[\
        (data[\
        (data[\
        (data[\
df_sign.columns
#####  Here the vbf preselection function (full) is applied to all the defined dataframes
The numpy array D will contain 'vbf' samples from df_sign and 'QCD', 'dipho' and 'ggH' samples from df_bkgs
(only feature columns included)
The numpy array Y will be the label array of binary labels, 1 for 'vbf' events and 0 for bkg events i.e for 'QCD', 'dipho', 'ggH'
The numpy array W will contain values in the weight column of df_sign and df_bkg
The numpy array I will contain values in the sample column of df_sign (this should be 'vbf') and values in the 'sample' column of df_bkg (i.e a mixture of 'QCD', 'dipho' and 'ggH'.


The same is then done for df_sign and df_data to get the X_data, Y_data, W_data, I_data, O_data
df_bkgs = df_bkgs[vbf_relax(df_bkgs)]
df_mc   = df_mc  [vbf_relax(df_mc  )]
df_sign = df_sign[vbf_relax(df_sign)]
df_data = df_data[vbf_relax(df_data)]

D  =  np.concatenate((df_sign[_features_],df_mc[_features_]))
Y  =  np.concatenate((np.ones(df_sign.shape[0]),np.zeros(df_mc.shape[0])))
W  =  np.concatenate((df_sign['weight'],df_mc['weight']))
I  =  np.concatenate((df_sign['sample'],df_mc['sample']))
O  =  np.concatenate((df_sign['dipho_mass'],df_mc['dipho_mass']))

X_data  =  np.concatenate((df_sign[_features_],df_data[_features_]))
Y_data  =  np.concatenate((np.ones(df_sign.shape[0]),np.zeros(df_data.shape[0])))
W_data  =  np.concatenate((df_sign['weight'],df_data['weight']))
I_data  =  np.concatenate((df_sign['sample'],df_data['sample']))
O_data  =  np.concatenate((df_sign['dipho_mass'],df_data['dipho_mass']))
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score

np.random.seed(42)

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
print(np.unique(Y_train))
print(np.unique(I_data))
import collections 
print \
for p in collections.Counter(I_data):
    print \
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
    \
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
###### checking what the clfs means now
print (clfs)
##### changing from labelled set to array
print(weights_train)
print(weights_train[i.split('-')[0]])
print(clfs.items())
### Preparing different Validation sets for checking rejection by process
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
#print(X_valid_ggh)
#print(rocs_valid_Y_ggh)
#print(rocs_valid_W_ggh)
#### Training while storing different validation sets
from sklearn.metrics import roc_curve, auc, roc_auc_score
rocs = {}
prob = {}
cfun = {}

rocs_valid = {}

rocs_valid_ggh = {}
rocs_valid_dipho = {}
rocs_valid_qcd = {}
rocs_valid_gjet = {}

prob_valid = {}
prob_valid_ggh = {}
prob_valid_dipho = {}
prob_valid_qcd = {}
prob_valid_gjet = {}

cfun_valid = {}

for i, c in clfs.items():
    print ' -- training : ', i
    c.fit ( X_train, Y_train, sample_weight= weights_train[i.split('-')[0]])
    
    prob[i] = c.predict_proba(X_train)[:,1]
    rocs[i] = roc_curve(Y_train,prob[i],sample_weight=W_train)

    prob_valid[i] = c.predict_proba(X_valid)[:,1]
    rocs_valid[i] = roc_curve(Y_valid,prob_valid[i],sample_weight=W_valid)
    
    prob_valid_ggh[i] = c.predict_proba(X_valid_ggh)[:,1]
    rocs_valid_ggh[i] = roc_curve(rocs_valid_Y_ggh,prob_valid_ggh[i],sample_weight=rocs_valid_W_ggh)
    
    prob_valid_dipho[i] = c.predict_proba(X_valid_dipho)[:,1]
    rocs_valid_dipho[i] = roc_curve(rocs_valid_Y_dipho,prob_valid_dipho[i],sample_weight=rocs_valid_W_dipho)
    
    prob_valid_qcd[i] = c.predict_proba(X_valid_qcd)[:,1]
    rocs_valid_qcd[i] = roc_curve(rocs_valid_Y_qcd,prob_valid_qcd[i],sample_weight=rocs_valid_W_qcd)
    
    prob_valid_gjet[i] = c.predict_proba(X_valid_gjet)[:,1]
    rocs_valid_gjet[i] = roc_curve(rocs_valid_Y_gjet,prob_valid_gjet[i],sample_weight=rocs_valid_W_gjet)
         
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
    plt.title('2016_relax_preselection_mva_score_plot_trained_on_MSB')
    plt.hist(tmva_train[(Y_train<0.5)], bins=50,
             weights=W_train[(Y_train<0.5)], 
             range=[-1,1],
             alpha=0.5, histtype='stepfilled', normed=1, label = 'MSB background')
    plt.hist(tmva_train[(Y_train>0.5)], bins=50,
             weights=W_train[(Y_train>0.5)], 
             range=[-1,1],
             alpha=0.5, histtype='stepfilled', normed=1, label = 'vbf')

    plt.hist(tmva_train[I_train == 'ggh'], bins=50,
             weights=W_train[I_train == 'ggh'], 
             range=[-1,1],
             alpha=0.5, histtype='step', color = 'blue',lw=1.2, normed=1)
    plt.legend(loc = 'upper center')
    plt.savefig('2016_relax_preselection_mva_score_plot_trained_on_MSB.png', bbox_inches = 'tight')
    plt.show()
print(tmva_train[(Y_train<0.5)])
print(np.unique(Y_train))
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
        print \
        plt.plot(fpr, tpr, label=name+'(area = %0.4f)'%(roc_auc_), zorder=5, lw=1.2)
    if rocs_train is not None : 
        for name,roc in rocs_train.items():
            fpr, tpr, thr = roc
            roc_auc_ = auc(fpr, tpr, reorder=True)
            print \
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
    plt.savefig('roc_'+label+'.png')
    plt.show()
plot_rocs(rocs_valid, label='2016_ROC_XCHECK1_relaxed_VBF_preselection_trained_on_MSB_tested_on_MC', title = 'ROC for all backgrounds')
plot_rocs(rocs_valid_ggh, label='2016_ROC_XCHECK2_relaxed_VBF_preselection_trained_on_MSB_tested_on_MC', title = 'ROC for ggh rejection')
plot_rocs(rocs_valid_dipho, label='2016_ROC_XCHECK3_relaxed_VBF_preselection_trained_on_MSB_tested_on_MC', title = 'ROC for dipho rejection')
plot_rocs(rocs_valid_qcd, label='2016_ROC_XCHECK4_relaxed_VBF_preselection_trained_on_MSB_tested_on_MC', title = 'ROC for qcd rejection')
plot_rocs(rocs_valid_gjet, label='2016_ROC_XCHECK5_relaxed_VBF_preselection_trained_on_MSB_tested_on_MC', title = 'ROC for gjet rejection')
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

    plt.xlabel(\
    plt.ylabel(\
    plt.legend(loc='best')
    plt.ylim([0.01, 2*max(hist)])
    plt.savefig('2016_relax_preselection_overtrain_xcheck_plot_trained_on_MSB.png')
    plt.show()
for i, c in clfs.items():
    print \
    compare_train_test(c,
                       X_train,Y_train,W_train, 
                       X_valid,Y_valid,W_valid, label='2016_relax_preselection_overtrain_xcheck_plot_trained_on_MSB')
