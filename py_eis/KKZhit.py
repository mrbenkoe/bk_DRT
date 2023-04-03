def Z_hit(Z,f,f_set):
    import math
    import scipy.fftpack
    import scipy.io as sio
    import numpy.matlib
    import sys
    import scipy.signal as signal
    import scipy.integrate as integrate
    from scipy.special import zeta
    import numpy as np
    Z=np.array(Z)
    faza=np.angle(Z)
    amplit=np.abs(Z)
    f = np.array(f)
    w=2.*np.pi*f
    dw=np.log(w)
    integ1=-2/np.pi*integrate.trapz(faza[len(dw)-1:],dw[len(dw)-1:])
    for i in range(2,len(dw)+1):
        integ_rac= -2/np.pi*integrate.trapz(faza[len(dw)-i:],dw[len(dw)-i:])
        integ1=np.append(integ1,integ_rac)
    integ1=integ1[::-1]
    gradien = -2/np.pi*zeta(1+1)*(2**(-1))*signal.savgol_filter(faza,13,9,1,delta=np.diff(dw)[0])
    gradien3 = -2/np.pi*zeta(1+3)*(2**(-1))*signal.savgol_filter(faza,13,9,3,delta=np.diff(dw)[0])
    ln_amp= integ1 + gradien  +  np.log(amplit[-1])
    #offset_at_10hz=np.argwhere(f>=6)[0]
    offset_at_10hz=np.argwhere(f>=f_set)[0]
    ln_amp= ln_amp + (np.log(amplit)[offset_at_10hz] - ln_amp[offset_at_10hz])
    return np.exp(ln_amp)
