def run_KK(Fq,Zr,Zi,KK_threshold,f_min,f_max):  
    import numpy as np, pandas as pd
    from . import KKZhit as KKZhit 
    import plotly.graph_objects as go
    from scipy.optimize import minimize 
    f_set=0
    datap=pd.DataFrame({'F':Fq,'Z': Zr+Zi*1j}).dropna()
    datap['ZrKK']=KKZhit.Z_hit(np.array(datap['Z']),np.array(datap['F']),f_set) 
    datap_raw=datap
    xo=np.abs(datap['Z'][(datap['F']>f_min)&(datap['F']<f_max)].to_numpy())
    yo=datap['ZrKK'][(datap['F']>f_min)&(datap['F']<f_max)].to_numpy()
    zo=0
    def kk_dist(zo):
        res = abs(np.trapz(xo)-np.trapz(yo-zo))
        return res
    res=minimize(kk_dist, zo,method='BFGS',options={'gtol': 1e-07})    
    datap['ZrKK']=np.array(datap['ZrKK'])-res.x[0]
    datap = datap[np.abs(np.real(datap['Z'])/np.real(datap['ZrKK'])-1)<=KK_threshold]
    return datap, datap_raw