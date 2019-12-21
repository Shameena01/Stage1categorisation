def addPt(row):
    return row['CMS_hgg_mass']*row['diphoptom']

def truthDipho(row):
    if not row['HTXSstage1cat']==0: return 1
    else: return 0

#def truthVhHad(row):
#    if row['tempStage1bin']==203: return 1
#    elif row['tempStage1bin']>107 and row['tempStage1bin']<111:return 0
#   else:return -1

def truthVhHad(row):
    if row['proc']=='vh':return 1
    elif row['proc']=='vbf':return 1
    else: return 0

def truthProcess(row):
    if row['sample']=='ggh': return 0
    elif row['sample']=='VBF': return 1
    elif row['sample']=='QCD': return 2
    elif row['sample']=='dipho': return 2
    else:return 2
def truthProcessTwoClass(row):
    #if row['proc']=='ggh': return 0
    if row['proc']=='vbf': return 1
    else:return 0


def vhHadWeight(row, ratio):
    weight =1000. * abs(row['weight'])
    if row['truthVhHad']==1: 
      return ratio * weight
    else: return weight

def ProcessWeight(row,ratio1, ratio2):
    weight =abs(row['weightR'])
    if row['truthProcess']==0:
       return ratio1 * weight
    elif row['truthProcess']==1:
       return ratio2 * weight
    else: return weight 

def ProcessWeightTwoClass(row,ratio1):
    weight =abs(row['weightR'])
    if row['truthProcess']==1:
       return ratio1 * weight
    #elif row['truthProcess']==1:
       #return ratio2 * weight
    else: return weight





def truthClass(row):
    if not row['stage1cat']==0: return int(row['stage1cat']-3)
    else: return 0

def truthJets(row):
    if row['stage1cat']==3: return 0
    elif row['stage1cat']>=4 and row['stage1cat']<=7: return 1
    elif row['stage1cat']>=8 and row['stage1cat']<=11: return 2
    else: return -1

def reco(row):
    if row['n_rec_jets']==0: return 0
    elif row['n_rec_jets']==1:
        if row['diphopt'] < 60: return 1
        elif row['diphopt'] < 120: return 2
        elif row['diphopt'] < 200: return 3
        else: return 4
    else:
        if row['diphopt'] < 60: return 5
        elif row['diphopt'] < 120: return 6
        elif row['diphopt'] < 200: return 7
        else: return 8

def diphoWeight(row, sigWeight=1.):
    weight = row['weight']
    if row['proc'].count('qcd'): 
        weight *= 0.04 #downweight bc too few events
    elif row['HTXSstage1cat'] > 0.01:
        weight *= sigWeight #arbitrary change in signal weight, to be optimised
    #now account for the resolution
    if row['sigmarv']>0. and row['sigmawv']>0.:
        weight *= ( (row['vtxprob']/row['sigmarv']) + ((1.-row['vtxprob'])/row['sigmawv']) )
    weight = abs(weight)
    return weight

def combinedWeight(row):
    weight = row['weight']
    if row['proc'].count('qcd'): 
        weight *= 0.04 #downweight bc too few events
    weight = abs(weight)
    return weight

def normWeight(row, bkgWeight=100., zerojWeight=1.):
    weightFactors = [0.0002994, 0.0000757, 0.0000530, 0.0000099, 0.0000029, 0.0000154, 0.0000235, 0.0000165, 0.0000104] #FIXME update these
    weight = row['weight']
    if row['proc'].count('qcd'): 
        weight *= 0.04 / weightFactors[ int(row['truthClass']) ] #reduce because too large by default
    else: 
        weight *= 1. / weightFactors[ int(row['truthClass']) ] #otherwise just reweight by xs
    weight = abs(weight)
    #arbitrary weight changes to be optimised
    if row['proc'] != 'ggh':
        weight *= bkgWeight
    elif row['reco'] == 0: 
        weight *= zerojWeight
    return weight

def jetWeight(row):
    weightFactors = [0.606560, 0.270464, 0.122976]
    weight = row['weight']
    weight *= 1. / weightFactors[ int(row['truthJets']) ] #otherwise just reweight by xs
    weight = abs(weight)
    return weight

#def altDiphoWeight(row, sigWeight=1./0.001169):
def altDiphoWeight(row, sigWeight=1./0.001297):
    weight = row['weight']
    if row['proc'].count('qcd'):
        weight *= 0.04 #downweight bc too few events
    elif row['HTXSstage1cat'] > 0.01:
        weight *= sigWeight #arbitrary change in signal weight, to be optimised
    #now account for the resolution
    if row['sigmarv']>0. and row['sigmawv']>0.:
        weight *= ( (row['vtxprob']/row['sigmarv']) + ((1.-row['vtxprob'])/row['sigmawv']) )
    weight = abs(weight)
    return weight

#define resolution weighting
def resolution_weighting(row):
    weight = row['weight']
    if row['sigmarv']>0. and row['sigmawv']>0.:
        weight *= ( (row['vtxprob']/row['sigmarv']) + ((1.-row['vtxprob'])/row['sigmawv']) )
    weight = abs(weight)
    return weight
