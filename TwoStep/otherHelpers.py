import numpy as np

def prettyHist( hist ):
    hist.SetStats(0)
    hist.GetXaxis().SetTitle('Process')
    hist.GetXaxis().SetTickLength(0.)
    hist.GetYaxis().SetTitle('Category')
    hist.GetYaxis().SetTitleOffset(1.5)
    hist.GetYaxis().SetTickLength(0.)
    hist.SetMinimum(-0.00001)
    hist.SetMaximum(100.)
    
def getRealSigma( hist ):
    sigma = 2.
    if hist.GetEntries() > 0 and hist.Integral()>0.000001:
        hist.Fit('gaus')
        fit = hist.GetFunction('gaus')
        sigma = fit.GetParameter(2)
    return sigma

def getAMS(s, b, breg=3.):
    b = b + breg
    val = 0.
    if b > 0.:
        val = (s + b)*np.log(1. + (s/b))
        val = 2*(val - s)
        val = np.sqrt(val)
    return val

def computeBkg( hist, effSigma ):
    bkgVal = 9999.
    if hist.GetEntries() > 0 and hist.Integral()>0.000001:
        hist.Fit('expo')
        fit = hist.GetFunction('expo')
        bkgVal = fit.Integral(125. - effSigma, 125. + effSigma)
    return bkgVal

def jetPtToClass( jets, pt ):
  return ( (jets>0).astype(int) + (4*((jets>1).astype(int))) + ((jets>0)*(pt>60.)).astype(int) + ((jets>0)*(pt>120.)).astype(int) + ((jets>0)*(pt>200.)).astype(int) )
