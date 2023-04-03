import numpy as np
import pandas as pd
from scipy.optimize import minimize

def small_calc(bc, shift, add_shift, Z_data):     
        bc_shifted = np.real(bc)+shift+np.imag(bc)*1j + add_shift          
        points = []
        for z in Z_data:
            dist = abs(bc_shifted - z)
            loc = np.where(dist == min(dist))[0][0]
            points.append(bc_shifted[loc])        
        diff = (abs(Z_data)-abs(np.array(points)))*1000           
        return bc_shifted, points, diff

def validation(data_list,oi,bc,start=0.1,toler=1e-07,evals=100):
    shift = data_list[oi][1][np.where(data_list[oi][2]<=0)[0][-1]]
    Fq_red = data_list[oi][0][np.where(data_list[oi][2]<=0)]
    Zr_red = data_list[oi][1][np.where(data_list[oi][2]<=0)]
    Zi_red = data_list[oi][2][np.where(data_list[oi][2]<=0)] 
    Z_data = Zr_red+Zi_red*1j 
    
    add_shift=shift*start
    
    def kk_dist(add_shift):
        bc_shifted, points, diff = small_calc(bc, shift, add_shift, Z_data)
        qual = np.mean(abs(diff))
        return qual
                     
    res = minimize(kk_dist, add_shift, method='L-BFGS-B', bounds=[(-shift*0.25,shift*0.25)],options={'gtol': toler, 
                                                                                                     'maxls': evals,
                                                                                                     'maxiter': 100000})    
    
    bc_shifted, points, diff = small_calc(bc, shift, res.x[0], Z_data)
    error = 0
    error_lines = pd.DataFrame([points,Z_data])
    for col in error_lines.columns:
        error += abs(error_lines[col][0]-error_lines[col][1])
    return shift, Fq_red, Z_data, add_shift, bc_shifted, points, diff, res, error, error_lines