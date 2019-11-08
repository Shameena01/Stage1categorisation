#usual imports
import ROOT as r
import numpy as np
import pandas as pd
import xgboost as xg
import uproot as upr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import sklearn
from sklearn.metrics import roc_auc_score, roc_curve
from os import path, system

from addRowFunctions import truthProcess, ProcessWeight, truthDipho
from otherHelpers import prettyHist, getAMS, computeBkg, getRealSigma
from root_numpy import fill_hist
import usefulStyle as useSty

from matplotlib import rc
from bayes_opt import BayesianOptimization
from catOptim import CatOptim

print 'imports done'

pd.options.mode.chained_assignment = None

np.random.seed(42)



#configure options
from optparse import OptionParser
parser = OptionParser()
parser.add_option('-t','--trainDir', help='Directory for input files')
parser.add_option('-x','--modelDir', help = 'Directory for models')
parser.add_option('-d','--dataFrame', default=None, help='Path to dataframe if it already exists')
parser.add_option('--intLumi',type='float', default=35.9, help='Integrated luminosity')
parser.add_option('--trainParams',default=None, help='Comma-separated list of colon-separated pairs corresponding to parameters for the training')
#parser.add_option('-m','--modelName', default=None, help='Name of model for testing')

parser.add_option('-n','--nIterations', default=10000, help='Number of iterations to run for random significance optimisation')
parser.add_option('-s','--signalFrame', default=None, help='Name of signal dataframe if it already exists')
parser.add_option('-m','--diphomodelName', default=None, help='Name of diphomodel for testing')
#parser.add_option('-v','--dijetmodelName', default = None, help = 'Name of dijet model for testing')

(opts,args)=parser.parse_args()


print 'option added'


#setup global variables
trainDir = opts.trainDir
if trainDir.endswith('/'): trainDir = trainDir[:-1]# slice the string to remove the last character i.e the "/"
frameDir = trainDir.replace('trees','frames')
modelDir = trainDir.replace('trees','models')

if opts.trainParams: opts.trainParams = opts.trainParams.split(',')#separate train options based on comma (used to define parameter pairs)

#get trees from files, put them in data frames
procFileMap = {'ggh':'ggH.root', 'vbf':'VBF.root', 'Dipho':'Dipho.root','GJet':'GJet.root','QCD':'QCD.root'}# a dictionary with file names
theProcs = procFileMap.keys()# list of keys i.e 'ggh','vbf','Data'


print 'processes defined'


#define the different sets of variables used                                                                                                                                                                      
diphoVars  = ['dipho_leadIDMVA','dipho_subleadIDMVA','dipho_lead_ptoM','dipho_sublead_ptoM',
              'dipho_leadEta','dipho_subleadEta',
              'dipho_cosphi','vtxprob','sigmarv','sigmawv']

dijetVars = ['dipho_lead_ptoM','dipho_sublead_ptoM','dijet_LeadJPt','dijet_SubJPt','dijet_abs_dEta','dijet_Mjj','dijet_centrality_gg','dijet_dphi','dijet_minDRJetPho','dijet_dipho_dphi_trunc']



allVars = ['dipho_leadIDMVA','dipho_subleadIDMVA','dipho_lead_ptoM','dipho_sublead_ptoM',
           'dipho_leadEta','dipho_subleadEta','dijet_leadEta','dijet_subleadEta','dijet_nj',
              'dipho_cosphi','vtxprob','sigmarv','sigmawv','HTXSstage1cat','dipho_mass','weight','dijet_LeadJPt','dijet_SubJPt','dijet_abs_dEta','dijet_Mjj','dijet_dphi','dijet_minDRJetPho','dijet_dipho_dphi_trunc','dijet_Zep','dipho_dijet_ptHjj']

dataVars = ['dipho_leadIDMVA','dipho_subleadIDMVA','dipho_lead_ptoM','dipho_sublead_ptoM',
              'dipho_leadEta','dipho_subleadEta',
              'dipho_cosphi','vtxprob','sigmarv','sigmawv','dipho_mass','weight','dijet_LeadJPt','dijet_SubJPt','dijet_abs_dEta','dijet_Mjj','dijet_dphi','dijet_minDRJetPho','dijet_dipho_dphi_trunc','dijet_Zep','dipho_dijet_ptHjj']



print 'variables chosen'


#either get existing data frame or create it
trainTotal = None
if not opts.dataFrame:#if the dataframe option was not used while running, create dataframe from files in folder
  trainFrames = {}
  #get the trees, turn them into arrays
  for proc,fn in procFileMap.iteritems():#proc, fn are the pairs 'proc':'fn' in the file map 
      trainFile   = upr.open('%s/%s'%(trainDir,fn))
      print proc 
      print fn
  #is a reader and a writer of the ROOT file format using only Python and Numpy.
  #Unlike PyROOT and root_numpy, uproot does not depend on C++ ROOT. Instead, it uses Numpy to cast blocks of data from the ROOT file as Numpy arrays.
      if (proc=='Dipho'):
         trainTree = trainFile['vbfTagDumper/trees/dipho_13TeV_GeneralDipho']
      elif (proc=='GJet'):
         trainTree = trainFile['vbfTagDumper/trees/gjet_anyfake_13TeV_GeneralDipho']
      elif (proc=='QCD'):
         trainTree = trainFile['vbfTagDumper/trees/qcd_anyfake_13TeV_GeneralDipho']
      elif (proc=='ggh'):
         trainTree = trainFile['vbfTagDumper/trees/ggh_125_13TeV_GeneralDipho']
      elif (proc=='vbf'):
         trainTree = trainFile['vbfTagDumper/trees/vbf_125_13TeV_GeneralDipho']
      else:
         trainTree = trainFile['vbfTagDumper/trees/%s_13TeV_VBFDiJet'%proc]
      print 'ok1'
      trainFrames[proc] = trainTree.pandas.df(allVars)
      print'ok'
      trainFrames[proc]['proc'] = proc #adding a column for the process
  print 'got trees'



#create one total frame
  trainList = []
  for proc in theProcs:
      trainList.append(trainFrames[proc])
  trainTotal = pd.concat(trainList)
  del trainFrames
  print 'created total frame'


#then filter out the events into only those with the phase space we are interested in
  trainTotal = trainTotal[((trainTotal['proc']=='ggh')&(trainTotal['dipho_mass']>100.)&(trainTotal['dipho_mass']<180.))|((trainTotal['proc']=='vbf')&(trainTotal['dipho_mass']>100.)&(trainTotal['dipho_mass']<180.))|((trainTotal['proc']=='Dipho')&(trainTotal['dipho_mass']>100.)&(trainTotal['dipho_mass']<180.))|((trainTotal['proc']=='GJet')&(trainTotal['dipho_mass']>100.)&(trainTotal['dipho_mass']<180.))|((trainTotal['proc']=='QCD')&(trainTotal['dipho_mass']>100.)&(trainTotal['dipho_mass']<180.))]# diphoton mass range
  #trainTotal = trainTotal[trainTotal.dipho_mass<180.]# diphoton mass range
  print 'done mass cuts'
#some extra cuts that are applied for vhHad BDT in the AN
  trainTotal = trainTotal[trainTotal.dipho_leadIDMVA>-0.2]
  trainTotal = trainTotal[trainTotal.dipho_subleadIDMVA>-0.2]
  trainTotal = trainTotal[trainTotal.dipho_lead_ptoM>0.333]
  trainTotal = trainTotal[trainTotal.dipho_sublead_ptoM>0.25]
  print 'done basic preselection cuts'
#cut on the jet pT to require at least 2 jets
  trainTotal = trainTotal[trainTotal.dijet_LeadJPt>40.]
  trainTotal = trainTotal[trainTotal.dijet_SubJPt>30.]
  print 'done jet pT cuts'
#consider the VH hadronic mjj region (ideally to cut on gen mjj for this)
  trainTotal = trainTotal[trainTotal.dijet_Mjj>250.]
  print 'done mjj cuts'


#adding variables that need to be calculated

  trainTotal['dijet_dipho_dEta']=((trainTotal.dijet_leadEta+trainTotal.dijet_subleadEta)/2)-((trainTotal.dipho_leadEta+trainTotal.dipho_subleadEta)/2)
  trainTotal['dijet_centrality_gg']=np.exp(-4*(trainTotal.dijet_Zep/trainTotal.dijet_abs_dEta)**2)
  print 'calculated variables added'

  trainTotal['truthDipho'] = trainTotal.apply(truthDipho,axis=1)
  trainTotal['truthProcess'] = trainTotal.apply(truthProcess,axis=1)#the truthProcess function returns 0 for ggh. 1 for vbf and 2 for background processes




  def adjust_qcd_weight(row):
      if row['proc']=='QCD':
         return row['weight']/25
      else:
         return row['weight']

  trainTotal['weightR'] = trainTotal.apply(adjust_qcd_weight, axis=1)

#add the target variable and the equalised weight
  trainTotal['truthProcess'] = trainTotal.apply(truthProcess,axis=1)#the truthProcess function returns 0 for ggh. 1 for vbf and 2 for background processes
  gghSumW = np.sum(trainTotal[trainTotal.truthProcess==0]['weightR'].values)#summing weights of ggh events
  vbfSumW = np.sum(trainTotal[trainTotal.truthProcess==1]['weightR'].values)#summing weights of vbf events
  dataSumW = np.sum(trainTotal[trainTotal.truthProcess==2]['weightR'].values)#summing weights of data events  
  totalSumW = gghSumW+vbfSumW+dataSumW
#getting number of events
  ggH_df = trainTotal[trainTotal.truthProcess==0]
  vbf_df = trainTotal[trainTotal.truthProcess==1]
  data_df = trainTotal[trainTotal.truthProcess==2]

  print 'ggh events'
  print ggH_df.shape[0]
  print 'vbf events'
  print vbf_df.shape[0]
  print 'data events'
  print data_df.shape[0]

  print 'weights before lum adjustment'
  print 'gghSumW, vbfSumW, dataSumW, ratio_ggh_data, ratio_vbf_data = %.3f, %.3f, %.3f, %.3f,%.3f'%(gghSumW, vbfSumW,dataSumW, gghSumW/dataSumW, vbfSumW/dataSumW)
  print 'ratios'
  print 'ggh ratio, vbf ratio, bkg ratio = %.3f, %.3f, %.3f'%(gghSumW/totalSumW, vbfSumW/totalSumW, dataSumW/totalSumW)


  trainTotal['ProcessWeight'] = trainTotal.apply(ProcessWeight, axis=1, args=[dataSumW/gghSumW,dataSumW/vbfSumW])#multiply each of the VH weight values by sum of nonsig weight/sum of sig weight 

#applying lum factors for ggh and vbf for training without equalised weights
  def weight_adjust (row):
      if row['truthProcess'] == 0:
         return 41.5 * row['weightR']
      if row['truthProcess'] == 1:
         return 41.5 * row['weightR']
      if row['truthProcess'] == 2:
         return 41.5 * row ['weightR']

  trainTotal['weightLUM'] = trainTotal.apply(weight_adjust, axis=1)


  gghSumW = np.sum(trainTotal[trainTotal.truthProcess==0]['weightLUM'].values)#summing weights of ggh events
  vbfSumW = np.sum(trainTotal[trainTotal.truthProcess==1]['weightLUM'].values)#summing weights of vbf events
  dataSumW = np.sum(trainTotal[trainTotal.truthProcess==2]['weightLUM'].values)#summing weights of data events
  totalSumW = gghSumW+vbfSumW+dataSumW

  print 'weights after lum adjustment'
  print 'gghSumW, vbfSumW, dataSumW, ratio_ggh_data, ratio_vbf_data = %.3f, %.3f, %.3f, %.3f,%.3f'%(gghSumW, vbfSumW,dataSumW,gghSumW/dataSumW, vbfSumW/dataSumW)
  print 'ggh ratio, vbf ratio, bkg ratio = %.3f, %.3f, %.3f'%(gghSumW/totalSumW, vbfSumW/totalSumW, dataSumW/totalSumW)


#trainTotal = trainTotal[trainTotal.truthVhHad>-0.5]

  print 'done weight equalisation'

#save as a pickle file
#if not path.isdir(frameDir): 
  system('mkdir -p %s'%frameDir)
  trainTotal.to_pickle('%s/ThreeClassTotal.pkl'%frameDir)
  print 'frame saved as %s/ThreeClassTotal.pkl'%frameDir

  #print 'printing ThreeClassTotal.pkl'
  #print trainTotal.head()



#read in dataframe if above steps done before
#else:

 # trainTotal = pd.read_pickle('%s/%s'%(frameDir,opts.dataFrame))
print 'Successfully loaded the  train frame'


#______________________-getting the data.root for the bkg yields later__________________________

print 'printing ThreeClassTotal.pkl'
print trainTotal.head()


#____________________________________________
#__________________________________________________

print 'making data frame'

dataFileMap = {'Data':'Data.root'}# a dictionary with file names  
dataTotal = None
#if not opts.dataFrame:
dataFrames = {}
  #get the trees, turn them into arrays
for proc,fn in dataFileMap.iteritems():
    trainFile   = upr.open('%s/%s'%(trainDir,fn))
    if (proc=='Dipho'):
         trainTree = trainFile['vbfTagDumper/trees/dipho_13TeV_GeneralDipho']
    elif (proc=='GJet'):
         trainTree = trainFile['vbfTagDumper/trees/gjet_anyfake_13TeV_GeneralDipho']
    elif (proc=='QCD'):
         trainTree = trainFile['vbfTagDumper/trees/qcd_anyfake_13TeV_GeneralDipho']
    elif (proc=='ggh'):
         trainTree = trainFile['vbfTagDumper/trees/ggh_125_13TeV_GeneralDipho']
    elif (proc=='vbf'):
         trainTree = trainFile['vbfTagDumper/trees/vbf_125_13TeV_GeneralDipho']
    else:
         trainTree = trainFile['vbfTagDumper/trees/%s_13TeV_GeneralDipho'%proc]
    #trainFile   = r.TFile('%s/%s'%(trainDir,fn))
    #if proc[-1].count('h') or 'vbf' in proc: trainTree = trainFile.Get('vbfTagDumper/trees/%s_125_13TeV_VBFDiJet'%proc)
    #else: trainTree = trainFile.Get('vbfTagDumper/trees/%s_13TeV_VBFDiJet'%proc)
#    trainTree.SetBranchStatus('nvtx',0)
 #   trainTree.SetBranchStatus('dijet_*',0)
  #  trainTree.SetBranchStatus('dijet_Mjj',1)
   # trainTree.SetBranchStatus('dijet_LeadJPt',1)
   # trainTree.SetBranchStatus('dijet_SubJPt',1)
   # trainTree.SetBranchStatus('dZ',0)
   # trainTree.SetBranchStatus('centralObjectWeight',0)
   # trainTree.SetBranchStatus('rho',0)
   # trainTree.SetBranchStatus('nvtx',0)
   # trainTree.SetBranchStatus('event',0)
   # trainTree.SetBranchStatus('lumi',0)
   # trainTree.SetBranchStatus('processIndex',0)
   # trainTree.SetBranchStatus('run',0)
   # trainTree.SetBranchStatus('npu',0)
   # trainTree.SetBranchStatus('puweight',0)
    #newFile = r.TFile('/vols/cms/es811/Stage1categorisation/trainTrees/new.root','RECREATE')
    #newTree = trainTree.CloneTree()
    #dataFrames[proc] = pd.DataFrame( tree2array(newTree) )
    #del newTree
    #del newFile
    dataFrames[proc] = trainTree.pandas.df(dataVars)
    dataFrames[proc]['proc'] = proc
print 'got trees'




dataTotal = dataFrames['Data']
  
#then filter out the events into only those with the phase space we are interested in
dataTotal = dataTotal[dataTotal.dipho_mass>100.]
dataTotal = dataTotal[dataTotal.dipho_mass<180.]
print 'done mass cuts'
  
#apply the full VBF preselection
dataTotal = dataTotal[dataTotal.dipho_leadIDMVA>-0.2]
dataTotal = dataTotal[dataTotal.dipho_subleadIDMVA>-0.2]
dataTotal = dataTotal[dataTotal.dipho_lead_ptoM>0.333]
dataTotal = dataTotal[dataTotal.dipho_sublead_ptoM>0.25]
dataTotal = dataTotal[dataTotal.dijet_Mjj>250.]
dataTotal = dataTotal[dataTotal.dijet_LeadJPt>40.]
dataTotal = dataTotal[dataTotal.dijet_SubJPt>30.]
print 'done VBF preselection cuts'
 # dataTotal['dijet_centrality_gg']=np.exp(-4*(dataTotal.dijet_Zep/dataTotal.dijet_abs_dEta)**2)
  #save as a pickle file
  #if not path.isdir(frameDir): 
    #system('mkdir -p %s'%frameDir)
  #dataTotal.to_pickle('%s/dataTotal.pkl'%frameDir)
  #print 'frame saved as %s/dataTotal.pkl'%frameDir
#else:
  #dataTotal = pd.read_pickle('%s/%s'%(frameDir,opts.dataFrame))

dataTotal['dijet_centrality_gg']=np.exp(-4*(dataTotal.dijet_Zep/dataTotal.dijet_abs_dEta)**2)

print 'printing data'
print dataTotal.head()

#set up train set and randomise the inputs
trainFrac = 0.90


theShape = trainTotal.shape[0]#number of rows in total dataframe
theShuffle = np.random.permutation(theShape)
trainLimit = int(theShape*trainFrac)


#define the values needed for training as numpy arrays
#vhHadVars = ['dipho_lead_ptoM','dipho_sublead_ptoM','dipho_mva', 'dijet_leadEta','dijet_subleadEta','dijet_LeadJPt','dijet_SubJPt','dijet_abs_dEta', 'dijet_Mjj', 'dijet_nj', 'cosThetaStar','cos_dijet_dipho_dphi', 'dijet_dipho_dEta']#do not provide dipho_mass=>do not bias the BDT by the Higgs mass used in signal MC

#BDTVars = ['dipho_lead_ptoM','dipho_sublead_ptoM','dipho_mva','dijet_leadEta','dijet_subleadEta','dijet_LeadJPt','dijet_SubJPt','dijet_abs_dEta','dijet_Mjj','dijet_nj', 'cosThetaStar', 'cos_dijet_dipho_dphi','dijet_dipho_dEta','dijet_centrality_gg','dijet_jet1_QGL','dijet_jet2_QGL','dijet_dphi','dijet_minDRJetPho']

#dijet MVA var
BDTVars = ['dipho_lead_ptoM','dipho_sublead_ptoM','dijet_LeadJPt','dijet_SubJPt','dijet_abs_dEta','dijet_Mjj','dijet_centrality_gg','dijet_dphi','dijet_minDRJetPho','dijet_dipho_dphi_trunc']

#dijet+diphoton MVA var
#BDTVars = ['dipho_lead_ptoM','dipho_sublead_ptoM','dijet_LeadJPt','dijet_SubJPt','dijet_abs_dEta','dijet_Mjj','dijet_centrality_gg','dijet_dphi','dijet_minDRJetPho','dijet_dipho_dphi_trunc','dipho_leadEta','dipho_subleadEta','dipho_cosphi','dipho_leadIDMVA','dipho_subleadIDMVA','sigmarv','sigmawv','vtxprob']



#lessvar
#BDTVars = ['dipho_lead_ptoM','dipho_sublead_ptoM','dipho_mva','dijet_leadEta','dijet_subleadEta','dijet_LeadJPt','dijet_SubJPt','dijet_abs_dEta','dijet_Mjj','dijet_nj', 'cosThetaStar', 'cos_dijet_dipho_dphi','dijet_dipho_dEta','dijet_centrality_gg','dijet_jet1_QGL','dijet_jet2_QGL','dijet_dphi','dijet_minDRJetPho','dipho_leadIDMVA','dipho_subleadIDMVA', 'dipho_cosphi','dipho_leadEta','dipho_subleadEta','dipho_leadPhi','dipho_subleadPhi','dijet_dipho_dphi_trunc','dijet_dipho_pt','dijet_mva','dipho_dijet_MVA']

#BDTVars = ['dipho_lead_ptoM','dipho_sublead_ptoM','dipho_mva','dijet_leadEta','dijet_subleadEta','dijet_LeadJPt','dijet_SubJPt','dijet_abs_dEta','dijet_Mjj','dijet_nj', 'cosThetaStar', 'cos_dijet_dipho_dphi','dijet_dipho_dEta','dijet_centrality_gg','dijet_jet1_QGL','dijet_jet2_QGL','dijet_dphi','dijet_minDRJetPho','dipho_leadIDMVA','dipho_subleadIDMVA', 'dipho_cosphi','vtxprob','sigmarv','sigmawv','dipho_leadEta','dipho_subleadEta','dipho_leadPhi','dipho_subleadPhi','dipho_leadR9','dipho_subleadR9','dijet_dipho_dphi_trunc','dijet_dipho_pt','dijet_mva','dipho_dijet_MVA','dijet_jet1_RMS','dijet_jet2_RMS','dipho_lead_hoe','dipho_sublead_hoe','dipho_lead_elveto','dipho_sublead_elveto','jet1_HFHadronEnergyFraction','jet1_HFEMEnergyFraction', 'jet2_HFHadronEnergyFraction','jet2_HFEMEnergyFraction']





BDTX  = trainTotal[BDTVars].values# the train input variables defined in the above list
BDTY  = trainTotal['truthProcess'].values#the training target two classes 1 for vh had 0 for other processes 
BDTTW = trainTotal['ProcessWeight'].values
BDTFW = trainTotal['weightLUM'].values
BDTM  = trainTotal['dipho_mass'].values
BDTNJ = trainTotal['dijet_nj'].values


#do the shuffle
BDTX  = BDTX[theShuffle]
BDTY  = BDTY[theShuffle]
BDTTW = BDTTW[theShuffle]
BDTFW = BDTFW[theShuffle]
BDTM  = BDTM[theShuffle]
BDTNJ = BDTNJ[theShuffle]

#split into train and test
BDTTrainX,  BDTTestX  = np.split( BDTX,  [trainLimit] )
BDTTrainY,  BDTTestY  = np.split( BDTY,  [trainLimit] )
BDTTrainTW, BDTTestTW = np.split( BDTTW, [trainLimit] )
BDTTrainFW, BDTTestFW = np.split( BDTFW, [trainLimit] )
BDTTrainM,  BDTTestM  = np.split( BDTM,  [trainLimit] )
BDTTrainNJ, BDTTestNJ = np.split( BDTNJ, [trainLimit] )

#set up the training and testing matrices
trainMatrix = xg.DMatrix(BDTTrainX, label=BDTTrainY, weight=BDTTrainTW, feature_names=BDTVars)
testMatrix  = xg.DMatrix(BDTTestX, label=BDTTestY, weight=BDTTestTW, feature_names=BDTVars)

trainMatrix_testing_odd_behaviour = xg.DMatrix(BDTTrainX, label=BDTTrainY, weight=BDTTrainFW, feature_names=BDTVars)
testMatrix_testing_odd_behaviour = xg.DMatrix(BDTTestX, label=BDTTestY, weight=BDTTestFW, feature_names=BDTVars)



#train on equalised weights
#trainMatrix = xg.DMatrix(BDTTrainX, label=BDTTrainY, weight=BDTTrainTW, feature_names=BDTVars)
#testMatrix  = xg.DMatrix(BDTTestX, label=BDTTestY, weight=BDTTestTW, feature_names=BDTVars)

trainParams = {}
trainParams['objective'] = 'multi:softprob'
trainParams['num_class']=3

trainParams['nthread'] = 1#--number of parallel threads used to run xgboost



#playing with parameters
trainParams['eta']=0.2
#trainParams['max_depth']=7
#trainParams['subsample']=0.9
#trainParams['colsample_bytree']=0.7
#trainParams['min_child_weight']=0
#trainParams['gamma']=0
trainParams['eval_metric']='merror'

trainParams['seed'] = 123456
#trainParams['reg_alpha']=0.1
#trainParams['reg_lambda']=

#add any specified training parameters
paramExt = ''
if opts.trainParams:
  paramExt = '__'
  for pair in opts.trainParams:
    key  = pair.split(':')[0]
    data = pair.split(':')[1]
    trainParams[key] = data
    paramExt += '%s_%s__'%(key,data)
  paramExt = paramExt[:-2]

progress = dict()
watchlist  = [(trainMatrix,'train'), (testMatrix, 'eval')]


print 'trying cross-validation'
#x_parameters = {"max_depth":[5,6,7,8,10], "objective":'multi:softprob', "num_class":3, "nthread":1, "eta":[0.05,0.1,0.3], "subsample":[0.5, 0.8, 1.],"eval_metric":'merror', "seed":123456}
#xg.cv(x_parameters, trainMatrix)





#train the BDT (specify number of epochs here)
print 'about to train BDT'
ThreeClassModel = xg.train(trainParams, trainMatrix,15,watchlist)
print 'done'
print progress

#ThreeClassModel = xg.Booster()
#ThreeClassModel.load_model('%s/%s'%(modelDir,opts.modelName))
 

#save it
modelDir = trainDir.replace('trees','models')
if not path.isdir(modelDir):
  system('mkdir -p %s'%modelDir)
ThreeClassModel.save_model('%s/ThreeClassModel%s.model'%(modelDir,paramExt))
print 'saved as %s/ThreeClassModel%s.model'%(modelDir,paramExt)




#evaluate performance 
print 'predicting test and train sets from trained model'
BDTPredYtrain = ThreeClassModel.predict(trainMatrix)
BDTPredYtest  = ThreeClassModel.predict(testMatrix)

BDTPredYtrain_T = ThreeClassModel.predict(trainMatrix_testing_odd_behaviour)
BDTPredYtest_T  = ThreeClassModel.predict(testMatrix_testing_odd_behaviour)

print 'prediction probabilities column 0 -ggh'
print BDTPredYtrain[:,0]

print 'prediction probabilities column 0 -ggh - FW'
print BDTPredYtrain[:,0]

#print 'prediction probabilities column 1 -vbf'
#print BDTPredYtrain[:,1]
#print 'prediction probabilities column 2 -bkg'
#print BDTPredYtrain[:,2]

#print 'trying maximum value'
#print np.argmax(BDTPredYtrain,axis=1)


#print 'labels'


BDTPredClassTrain = np.argmax(BDTPredYtrain,axis=1)
BDTPredClassTest = np.argmax(BDTPredYtest,axis=1)
###########################





#define the variables used as input to the classifier
trainM  = trainTotal['dipho_mass'].values
trainFW = trainTotal['weight'].values
trainH  = trainTotal['dipho_dijet_ptHjj'].values
trainJ  = trainTotal['dijet_Mjj'].values
trainL  = trainTotal['dijet_LeadJPt'].values

dataM  = dataTotal['dipho_mass'].values
dataFW = np.ones(dataM.shape[0])
dataH  = dataTotal['dipho_dijet_ptHjj'].values
dataJ  = dataTotal['dijet_Mjj'].values
dataL  = dataTotal['dijet_LeadJPt'].values


#obtain diphoton MVA predictions
diphoX = trainTotal[diphoVars].values
data_diphoX  = dataTotal[diphoVars].values
diphoP = trainTotal['truthDipho'].values
diphoMatrix = xg.DMatrix(diphoX, label=diphoP, weight=trainFW, feature_names=diphoVars)
datadiphoMatrix  = xg.DMatrix(data_diphoX,  label=dataFW, weight=dataFW,  feature_names=diphoVars)
diphoModel = xg.Booster()
diphoModel.load_model('%s/%s'%(modelDir,opts.diphomodelName))
diphoMVA = diphoModel.predict(diphoMatrix)
data_diphoMVA  = diphoModel.predict(datadiphoMatrix)

#obtain dijet MVA predictions
dijetX = trainTotal[dijetVars].values
data_dijetX = dataTotal[dijetVars].values
dijetP = trainTotal['truthProcess'].values
dijetMatrix = xg.DMatrix(dijetX,label = dijetP, weight = trainFW, feature_names = dijetVars)
datadijetMatrix = xg.DMatrix(data_dijetX , label = dataFW, weight = dataFW, feature_names = dijetVars)##################
#dijetModel = xg.Booster()
#dijetModel.load_model('%s/%s'%(modelDir,opts.dijetmodelName))
dijetMVA = ThreeClassModel.predict(dijetMatrix)

data_dijetMVA = ThreeClassModel.predict(datadijetMatrix)


dijetMVA_ggHprob = dijetMVA[:,0]
dijetMVA_vbfprob = dijetMVA[:,1]
data_dijetMVA_ggHprob =1-data_dijetMVA[:,0]
data_dijetMVA_vbfprob = data_dijetMVA[:,1]

x_ggh = (dijetMVA[:,0])[dijetP==0]
y_ggh =(dijetMVA[:,1])[dijetP==0]
w_ggh = trainFW[dijetP==0]

#print x_ggh
plt.hist2d(x_ggh, y_ggh,bins = 50,weights = w_ggh, range = [[0,1],[0,1]], label = 'ggh events')
plt.title('ggH events')
plt.xlabel('ggh probability')
plt.ylabel('vbf probability')
plt.savefig('2D_ggh_behav.png',bbox_inches = 'tight')
plt.savefig('2D_ggh_behav.pdf',bbox_inches = 'tight')


#plotting the diphoton BDT distributions for ggh events
x_ggh_dipho = (diphoMVA)[dijetP==0]
plt.figure()
plt.hist(x_ggh_dipho, bins = 50, weights = w_ggh, range = [0,1], label = 'ggh events')
plt.xlabel('diphoton BDT probability')
plt.title('ggh events')
plt.savefig('Dipho_dist_ggh.png', bbox_inches = 'tight')
plt.savefig('Dipho_dist_ggh.pdf', bbox_inches = 'tight')



#now estimate significance using the amount of background in a plus/mins 1 sigma window
#set up parameters for the optimiser
ptHjjCut = 25. 
ranges = [ [0,1.], [0,1.],[0,1]]
#ranges = [[0,1.],[0,1.]]

names  = ['DijetBDTvbf','DijetBDTggh','DiphotonBDT']
#names = ['DijetBDTvbf', 'DiphotonBDT']
printStr = ''


#optimising ot three categories (stage 0)
sigWeights = trainFW * (dijetP==1) * (trainJ>400)*(trainL<200)
bkgWeights = dataFW *(dataJ>400)*(dataL<200)
optimiser = CatOptim(sigWeights, trainM, [dijetMVA_vbfprob,dijetMVA_ggHprob, diphoMVA], bkgWeights, dataM, [data_dijetMVA_vbfprob, data_dijetMVA_ggHprob, data_diphoMVA], 3, ranges, names)
#optimiser = CatOptim(sigWeights, trainM, [dijetMVA_vbfprob, diphoMVA], bkgWeights, dataM, [data_dijetMVA_vbfprob, data_diphoMVA], 3, ranges, names)
#optimiser.setConstantBkg(True)
optimiser.setOpposite("DijetBDTggh")
print 'about to optimise cuts'

optimiser.optimise(opts.intLumi, opts.nIterations)
printStr += 'Results for 3 vbf categories'
printStr += optimiser.getPrintableResult()

print printStr
