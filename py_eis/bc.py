def back_calc(gg,tt):
    import numpy as np
    graph = gg
    tau_res = tt
    ind = ~np.isnan(graph)
    tau_res = tau_res[ind]
    graph = graph[ind]
    ww = np.logspace(-1,6,2400)
    cz = []
    tfs = np.divide(graph,tau_res)
    rs = np.trapz(x=tau_res,y=tfs)
    for iw in ww:
        cz.append(np.trapz(x=tau_res,y=np.divide(tfs,1+1j*iw*tau_res)))
    cz = np.array(cz) 
    return cz