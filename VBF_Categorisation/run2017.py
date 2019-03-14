from mlearn import classify
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

clf = classify.train(config="train_vbf_data.json")
clf.read_cfg()
clf.book_data()
data_all = pd.concat((clf.df_sig,clf.df_bkg,clf.df_data))

data_all['avg_pt' ] = (data_all.dipho_leadPt  + data_all.dipho_subleadPt )/2.0
data_all['max_id' ] = np.maximum(data_all.dipho_leadIDMVA,  data_all.dipho_subleadIDMVA )
data_all['min_id' ] = np.minimum(data_all.dipho_leadIDMVA,  data_all.dipho_subleadIDMVA )
data_all['min_eta'] = np.minimum(np.abs(data_all.dipho_leadEta ),  np.abs(data_all.dipho_subleadEta))
data_all['max_eta'] = np.maximum(np.abs(data_all.dipho_leadEta ),  np.abs(data_all.dipho_subleadEta))

data_all['pass_id'] = (data_all.min_id>-0.2)
data_all['fail_id'] = (data_all.min_id<-0.4)

data_all['Y'] = np.zeros(data_all.shape[0])
data_all.loc[( data_all['sample'] == 'vbf'), 'Y'] = 1*np.ones(data_all[(data_all['sample'] == 'vbf')].shape[0])
data_all.loc[( data_all['sample'] == 'data'), 'Y'] = 2*np.ones(data_all[(data_all['sample'] == 'data')].shape[0])

sample_sm =((data_all['sample'] == 'ggh') |
            (data_all['sample'] == 'dipho') |
            (data_all['sample'] == 'zee'))

data_all['Z'] = np.zeros(data_all.shape[0])
data_all.loc[sample_sm, 'Z'] = np.ones(data_all[sample_sm].shape[0])

data_all.to_hdf(
    '2017_Analysis_Without_datadriven.h5',
    'results_table',
    mode='w',
    format='table'
)
