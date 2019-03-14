import matplotlib.pyplot as plt
import numpy  as np
import pandas as pd
from scipy import optimize
%matplotlib inline
#plt.style.use('physics')

plt.rcParams['axes.grid'       ]  = False
plt.rcParams['xtick.labelsize' ]  = 14
plt.rcParams['ytick.labelsize' ]  = 14
plt.rcParams['axes.labelsize'  ]  = 14
plt.rcParams['legend.fancybox' ]  = False

pd.options.mode.chained_assignment = None

import binopt

from scipy import special as sp

def divide( a, b ):
    \
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide( a, b )
#         c[ ~ np.isfinite( c )] = 0
    return c
#df = pd.read_hdf('hgg-double-fake-trees-training-2017_From_Yacine_datadriven.h5')
df = pd.read_hdf('2017_Analysis_with_datadriven.h5')
def vbf_presel(data):
    return (
        (data[\
        (data[\
        (data[\
        (data[\
        (data[\
        (data[\
        (data[\

df = df[vbf_presel(df)]
df.columns.values
from sklearn.externals import joblib
clf = joblib.load('Classifier_From_Yacine_Generated_by_me.pkl') 

def evaluate_sklearn(cls, vals, coef=1):
    scale = 1.0 / cls.n_estimators
    ret = np.zeros(vals.shape[0])

    learning_rate = cls.learning_rate
    for itree, t in enumerate(cls.estimators_[:, 0]):
        r = t.predict(vals)
        ret += r * scale
    return 2.0/(1.0 + np.exp(-coef/learning_rate * ret)) - 1

df['dijet_centrality_gg'] = np.exp(-4*(df.dijet_Zep/df.dijet_abs_dEta)**2)
_dijetvar_ = [u'dijet_LeadJPt'  ,u'dijet_SubJPt', 
              u'dijet_abs_dEta' ,u'dijet_Mjj', 
              u'dijet_centrality_gg',u'dijet_dipho_dphi_trunc',
              u'dijet_dphi'     ,u'dijet_minDRJetPho',
              u'leadPho_PToM'   ,u'sublPho_PToM']

df['dijet_bdt'] = evaluate_sklearn(clf,df[_dijetvar_])
df_bkgs = df[
    (df['sample'] != 'data' ) & 
    (df['sample'] != 'qcd'  ) & 
    (df['sample'] != 'vbf'  ) &
    (df['sample'] != 'gjet' ) & 
    (df['sample'] != 'zee'  )
]
df_sign = df[
    (df['sample'] == 'vbf'  )
]
plt.figure(figsize=(4,4))

plt.hist(df_bkgs.dijet_bdt,bins=100, range=[-1,1], 
         alpha=0.4, weights=df_bkgs.weight, 
         histtype='stepfilled',lw=1, normed=1)
plt.hist(df_sign.dijet_bdt,bins=100, range=[-1,1], 
         alpha=0.4, weights=df_sign.weight, 
         histtype='stepfilled',lw=1, normed=1)
# plt.yscale('log')
plt.show()
rng = np.random.RandomState(15)  # deterministic random data

s = df_sign.dijet_bdt
b = df_bkgs.dijet_bdt

ms = df_sign.dipho_mass
mb = df_bkgs.dipho_mass

ws = df_sign.weight
wb = df_bkgs.weight

X = np.concatenate([s,b])
Y = np.concatenate([np.ones(s.shape[0]), np.zeros(b.shape[0])])
W = np.concatenate([ws,wb])
M = np.concatenate([ms,mb])


plt.figure(figsize=(5,5))
we_s, x = np.histogram(s, bins=50, range=[-1,1], weights=ws**2)
we_b, _ = np.histogram(b, bins=50, range=[-1,1], weights=wb**2)

he_s, _ = np.histogram(s, bins=50, range=[-1,1], weights=ws)
he_b, _ = np.histogram(b, bins=50, range=[-1,1], weights=wb)

x = np.array([(x[i] + x[i+1])/2.0  for i in range(0,len(x)-1)])
plt.errorbar(x,he_s, yerr=np.sqrt(we_s), fmt='.', c='r', markersize=8, capthick=0)
plt.errorbar(x,he_b, yerr=np.sqrt(we_b), fmt='.', c='b', markersize=8, capthick=0)

plt.yscale('log')
plt.xlim([-1,1])
plt.legend()
plt.show()
help(binopt.optimize_bin)
binner = binopt.optimize_bin(nbins=3, range=[-1,1], 
                             drop_last_bin=True, fix_upper=True, 
                             fix_lower=False, use_kde_density=False)
help(binner.fit)
binner.fit(X, Y, sample_weights=W, method=\
plt.figure(figsize=(5,5))

we_s, x = np.histogram(s, bins=50, range=[-1,1], weights=ws**2)
we_b, _ = np.histogram(b, bins=50, range=[-1,1], weights=wb**2)

he_s, _ = np.histogram(s, bins=50, range=[-1,1], weights=ws)
he_b, _ = np.histogram(b, bins=50, range=[-1,1], weights=wb)

x = np.array([(x[i] + x[i+1])/2.0  for i in range(0,len(x)-1)])
plt.errorbar(x,he_s, yerr=np.sqrt(we_s), fmt='.', c='r', markersize=8, capthick=0, label = 'signal')
plt.errorbar(x,he_b, yerr=np.sqrt(we_b), fmt='.', c='b', markersize=8, capthick=0,label = 'background')

#plt.hist(s, bins=50, range=[0,1], weights=ws,
          #color='red' ,histtype='step',lw=1.2, normed=0, label='signal')
#plt.hist(b, bins=50, range=[0,1], weights=wb,
          #color='blue',histtype='step',lw=1.2, normed=0, label='signal')

for x in binner.result.x:
    plt.axvline(x, ls='--', c = 'k')  
plt.title('3 bin boundary')
plt.yscale('log')
plt.xlim([-1,1])
plt.legend()
#plt.savefig('2_bins_before_significance_from_Yacine_datadriven_extended.png')
plt.show()
print binner.binned_score(binner.result.x)
print binner.binned_stats(binner.result.x)[0]
print binner.binned_stats(binner.result.x)[1]
print binner.binned_stats(binner.result.x)[2]
print binner.binned_stats(binner.result.x)[3]
 #print (df_bkgs['weight'])
(df_sign['weight'].sum())
df_bkgs_region = df_bkgs[(np.abs(df_bkgs['dipho_mass']-125)<2)]
df_sign_region = df_sign[(np.abs(df_sign['dipho_mass']-125)<2)]
#df_bkgs_region = df_bkgs
#df_sign_region = df_sign
df_bkgs_bin0 = df_bkgs_region[
    (df_bkgs_region['dijet_bdt'] > 0.9916 )]
df_sign_bin0 = df_sign_region[
    (df_sign_region['dijet_bdt'] > 0.9916 )]
df_bkgs_bin1 = df_bkgs_region[(df_bkgs_region['dijet_bdt'] > 0.9633) & (df_bkgs_region['dijet_bdt'] < 0.9916 ) ]
df_sign_bin1 = df_sign_region[(df_sign_region['dijet_bdt'] > 0.9633) & (df_sign_region['dijet_bdt'] < 0.9916 ) ]
df_bkgs_bin2 = df_bkgs_region[(df_bkgs_region['dijet_bdt'] < 0.9633) & (df_bkgs_region['dijet_bdt']> 0.8478)]
df_sign_bin2 = df_sign_region[(df_sign_region['dijet_bdt'] < 0.9633) & (df_sign_region['dijet_bdt']> 0.8478)]
#summing weights in each bin
#sum of weights in bin0

df_bkgs_bin0['weight'].sum()
df_sign_bin0['weight'].sum()
df_bkgs_bin1['weight'].sum()
df_sign_bin1['weight'].sum()
df_bkgs_bin2['weight'].sum()
df_sign_bin2['weight'].sum()
def sig(s, b, lum):
    berr_m = 5
    term_a_m = (s*lum + b + berr_m)
    #print term_a_m
    term_b_m = (1 + np.true_divide(s*lum, (b + berr_m)))
    #print term_b_m
    c_bin = np.sqrt(2*(term_a_m * np.log(term_b_m) - s*lum))
    return c_bin
bin0sig=sig(0.00028844195685451268,0.15559852912664313, 35.8)
print(bin0sig)
bin1sig = sig(0.019614677620734877,0.0041797538656282995, 35.8)
bin2sig = sig(0.31569419551633748,0.5500002132830436, 35.8)
total_sig = np.sqrt((bin0sig*bin0sig)+(bin1sig*bin1sig)+(bin2sig*bin2sig))
print (total_sig)
## Optimisation of the boundaries using $\\sigma_{\\rm eff}$ of signla peak
from scipy.stats import norm
from scipy.optimize import minimize
import scipy.stats as st

def binned_score_mgg(bounds, X, y, W, mass, nsig=1):
    \
    Input should contain a resonance of some sort.
    \
    _bounds_ = np.sort(np.append(bounds,binner.range))
    #_bounds_ = np.sort(np.insert(bounds, [0, bounds.shape[0]], [binner.range]))
    _cats_ = np.digitize(X, _bounds_)
    _seff_ = np.zeros(_bounds_.shape[0])
    _nums_ = np.zeros(_bounds_.shape[0])
    _numb_ = np.zeros(_bounds_.shape[0])
    _errb_ = np.zeros(_bounds_.shape[0])
    frac = np.abs(norm.cdf(0, -nsig, 1) - norm.cdf(0, nsig, 1))
    for cid in range(1,_bounds_.shape[0]):
        max_, min_ = binopt.tools.weighted_quantile(
            mass[(_cats_ == cid)& (y==1)],
            [norm.cdf(0, -nsig, 1), norm.cdf(0, nsig, 1)],
            sample_weight=W[(_cats_ == cid)& (y==1)])
        
        _seff_[cid] = np.abs(max_-min_)/2.0
        _nums_[cid] = W[(_cats_ == cid) & (y==1)].sum()
        _numb_[cid] = W[(_cats_ == cid) & (y==0)&
                        (mass<max_)&(mass>min_) ].sum()*nsig*_seff_[cid]
        _errb_[cid] = np.sqrt((W[(_cats_ == cid) & (y==0)&
                        (mass<max_)&(mass>min_) ]**2).sum())
#         print \
#     return _errb_
    return binner._fom_(_nums_, _numb_,_errb_, method=\

def binned_score_fit(bounds, X, y, W, mass, nsig=1):
    \
    Input should contain a resonance of some sort.
    \
    _bounds_ = np.sort(np.append(bounds, binner.range))
    #_bounds_ = np.sort(np.insert(bounds, [0, bounds.shape[0]], [binner.range]))
    _cats_ = np.digitize(X, _bounds_)
    _seff_ = np.zeros(_bounds_.shape[0])
    _nums_ = np.zeros(_bounds_.shape[0])
    _numb_ = np.zeros(_bounds_.shape[0])
    _errb_ = np.zeros(_bounds_.shape[0])
    frac = np.abs(norm.cdf(0, -nsig, 1) - norm.cdf(0, nsig, 1))
    
    for cid in range(1,_bounds_.shape[0]):
        def _obj(x):
            out = -np.sum(
                W[(_cats_ == cid) & (y==0)]*st.expon(
                    loc=100, scale=np.exp(x)
                ).logpdf(mass[(_cats_ == cid) & (y==0)])
            )
            if np.isnan(out):
                return 0
            else:
                return out
        _fit = minimize(_obj, x0=[0.03], method='Powell')
        min_, max_ = binopt.tools.weighted_quantile(
            mass[(_cats_ == cid)& (y==1)],
            [norm.cdf(0, -nsig, 1), norm.cdf(0, nsig, 1)],
            sample_weight=W[(_cats_ == cid)& (y==1)])
        
        _seff_[cid] = np.abs(max_-min_)/2.0
        _nums_[cid] = W[(_cats_ == cid) & (y==1)].sum()*frac
        _numb_[cid] = np.abs(
            st.expon(loc=100,scale=np.exp(_fit.x)).cdf(min_)-
            st.expon(loc=100,scale=np.exp(_fit.x)).cdf(max_)
        )
        _errb_[cid] = np.sqrt((W[(_cats_ == cid) & (y==0)]**2).sum()*_numb_[cid])
        _numb_[cid] *= W[(_cats_ == cid) & (y==0)].sum()
    return binner._fom_(_nums_, _numb_,_errb_, method=\

def cost_fun_mgg(x):
        \
        z = None
        z = binned_score_mgg(x, X, Y, W, M)
        return -np.sqrt((z[1:]**2).sum())
    
def cost_fun_fit(x):
        \
        z = None
        z = binned_score_fit(x, X, Y, W, M)
        return -np.sqrt((z[1:]**2).sum())
print(X)
print(Y)
print (W)
print(M)
print(binner.result.x)
print[binner.range]
print(binner.result.x.shape[0])
np.insert([0.8,0.9,0.95],[0,3],[[-1,1]])
np.sort(np.append ([0.7, 0.8,0.9],[-1,1]))
print \
print cost_fun_mgg(binner.result.x)
print \
print cost_fun_fit(binner.result.x)
def cost_mgg_(x):
    return cost_fun_mgg(np.array([x]))

def cost_fit_(x):
    return cost_fun_fit(np.array([x]))

def cost_std_(x):
    return binner.cost_fun(np.array([x]))
cost_mgg_ = np.vectorize(cost_mgg_)
cost_fit_ = np.vectorize(cost_fit_)
cost_std_ = np.vectorize(cost_std_)
plt.figure(figsize=(5,5))

t = np.linspace(0,X.max(),100)
plt.plot(t, cost_mgg_(t), 'b-')
plt.plot(t, cost_fit_(t), 'r-')
# plt.plot(t, cost_std_(t), 'g-')


# plt.yscale('log')
plt.xlim([0,1])
# plt.ylim([-1.1,-0.7])
plt.legend()
plt.show()
plt.figure(figsize=(5,5))

t = np.linspace(0,X.max(),100)
plt.plot(t, -cost_mgg_(t)/cost_mgg_(t).min(), 'b-', label = \
plt.plot(t, -cost_fit_(t)/cost_fit_(t).min(), 'r-', label = \
plt.plot(t, -cost_std_(t)/cost_std_(t).min(), 'g-', label = \
# plt.yscale('log')
plt.xlim([0,1])
# plt.ylim([-1.1,-0.7])
plt.xlabel(\
plt.legend(loc = \
plt.show()
