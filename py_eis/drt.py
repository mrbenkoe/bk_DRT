"""
DRT calculation based on https://sites.google.com/site/drttools/
Only Gaussian window
"""
import numpy as np
import scipy.optimize as opt
import scipy.integrate as integrate
from scipy.linalg import toeplitz
from cvxopt import solvers
import cvxopt

def quad_format_combined(A_re,A_im,b_re,b_im,M_re,M_im,lam) :
    H=2*(0.5*(A_re.T@A_re+A_im.T@A_im)+lam*M_re);
    c=-2*0.5*(b_im.T@A_im+b_re.T@A_re);
    return H,c

def  quad_format(A,b,M,lam):
    H=2*(A.T@A+lam*M);
    c=-2*b.T@A
    return H,c

def  inner_prod_rbf(freq_n, freq_m, epsilon, rbf_type):
    a = epsilon*np.log(freq_n/freq_m);
    if rbf_type == 'gaussian':
            out_IP = -epsilon*(-1+a**2)*np.exp(-(a**2/2))*np.sqrt(np.pi/2);
    else:
        raise Exception('Error')
    return out_IP

def  inner_prod_rbf_2(freq_n, freq_m, epsilon, rbf_type):
    a = epsilon*np.log(freq_n/freq_m);
    if rbf_type == 'gaussian':
            out_IP = epsilon**3*(3-6*a**2+a**4)*np.exp(-(a**2/2))*np.sqrt(np.pi/2)
    else:
        raise Exception('Error')
    return out_IP

def g_i(freq_n, freq_m, epsilon, rbf_type, integration_algorithm):
    alpha = 2*np.pi*freq_n/freq_m;
    if rbf_type=='gaussian':
            rbf = lambda x: np.exp(-np.power(epsilon*x,2))
    else:
        raise Exception('Error')
    integrand_g_i = lambda x: np.multiply( np.power(1+np.power(alpha,2)*np.exp(2*x),-1),rbf(x))   
    out_val,err = integrate.quad(integrand_g_i, -np.inf, np.inf, epsabs=1e-10, epsrel=1e-10)
    return out_val

def g_ii(freq_n, freq_m, epsilon, rbf_type, integration_algorithm):
    alpha = 2*np.pi*freq_n/freq_m
    if rbf_type == 'gaussian':
            rbf = lambda x: np.exp(-np.power(epsilon*x,2))
    else:
        raise Exception('Error')
    integrand_g_ii = lambda x: np.multiply(alpha*np.power( (np.power(np.exp(x),-1) + (alpha**2) * np.exp(x)),-1),rbf(x))
    out_val,err = integrate.quad (integrand_g_ii, -np.inf, np.inf,epsabs=1e-10,epsrel=1e-10)#,'RelTol',1E-6,'AbsTol',1e-6)
    return out_val

def assemble_A_im(freq, epsilon, rbf_type, integration_algorithm,L):
    std_freq = np.std(np.diff(-np.log(freq)));
    mean_freq = np.mean(np.diff(-np.log(freq)));
    R=np.zeros((1,freq.size))
    C=np.zeros((freq.size,1))
    out_A_im_temp=np.zeros((freq.size, freq.size));
    out_A_im = np.zeros( (freq.size, freq.size+2) )
    if std_freq/mean_freq<1:  
        for iter_freq_n in range(freq.size):
            freq_n = freq[iter_freq_n]
            freq_m = freq[0]
            C[iter_freq_n, 0] = g_ii(freq_n, freq_m, epsilon, rbf_type, integration_algorithm);        
        for iter_freq_m in range(freq.size):
                freq_n = freq[0]
                freq_m = freq[iter_freq_m]
                R[0, iter_freq_m] = g_ii(freq_n, freq_m, epsilon, rbf_type, integration_algorithm)
        out_A_im_temp = toeplitz(C,R);
    else:
        for iter_freq_n in range(freq.size):
            for iter_freq_m in range(freq.size):
                freq_n = freq[iter_freq_n]
                freq_m = freq[iter_freq_m]
                out_A_im_temp[iter_freq_n, iter_freq_m] = g_ii(freq_n, freq_m, epsilon, rbf_type, integration_algorithm);      
    out_A_im[:, 2:] = out_A_im_temp
    if L==1:
        out_A_im[:,0] = -2*np.pi*freq
    return out_A_im


def assemble_A_re(freq, epsilon, rbf_type, integration_algorithm):
    std_freq = np.std(np.diff(-np.log(freq)));
    mean_freq = np.mean(np.diff(-np.log(freq)));
    R=np.zeros((1,freq.size))
    C=np.zeros((freq.size,1))
    out_A_re = np.zeros( (freq.size, freq.size+2))
    out_A_re_temp = np.zeros((freq.size,freq.size))
    if std_freq/mean_freq<1:  
        for iter_freq_n in range(freq.size):
            freq_n = freq[iter_freq_n]
            freq_m = freq[0]
            val = g_i(freq_n, freq_m, epsilon, rbf_type, integration_algorithm)
            C[iter_freq_n, 0] = val      
            for iter_freq_m in range(freq.size):
                freq_n = freq[0]
                freq_m = freq[iter_freq_m]
                R[0, iter_freq_m] = g_i(freq_n, freq_m, epsilon, rbf_type, integration_algorithm);
        out_A_re_temp= toeplitz(C,R);
    else:
        for iter_freq_n in range(freq.size):
            for iter_freq_m in range(freq.size):
                freq_n = freq[iter_freq_n]
                freq_m = freq[iter_freq_m]
                out_A_re_temp[iter_freq_n, iter_freq_m] = g_i(freq_n, freq_m, epsilon, rbf_type, integration_algorithm);
    out_A_re[:, 2:] = out_A_re_temp;
    out_A_re[:,1] = 1;
    return out_A_re

def assemble_M_im(freq, epsilon, rbf_type, der_used):
    std_freq = np.std(np.diff(-np.log(freq)));
    mean_freq = np.mean(np.diff(-np.log(freq)));
    R=np.zeros((1,freq.size))
    C=np.zeros((freq.size,1))
    out_M_im_temp = np.zeros((freq.size,freq.size));
    out_M_im = np.zeros((freq.size+2, freq.size+2))
    if der_used == '1st-order':
            if std_freq/mean_freq<1:#  
                for iter_freq_n in range(freq.size):
                        freq_n = freq[iter_freq_n]
                        freq_m = freq[0]
                        C[iter_freq_n, 0] = inner_prod_rbf(freq_n, freq_m, epsilon, rbf_type);
                for iter_freq_m in range(freq.size):
                        freq_n = freq[0]
                        freq_m = freq[iter_freq_m]
                        R[0, iter_freq_m] = inner_prod_rbf(freq_n, freq_m, epsilon, rbf_type);
                out_M_im_temp = toeplitz(C,R);
            else:
                for iter_freq_n in range(freq.size):
                    for iter_freq_m in range(freq.size):
                        freq_n = freq[iter_freq_n]
                        freq_m = freq[iter_freq_m]
                        out_M_im_temp[iter_freq_n, iter_freq_m] = inner_prod_rbf(freq_n, freq_m, epsilon, rbf_type);
    else:
            if std_freq/mean_freq<1: 
                for iter_freq_n in range(freq.size):
                        freq_n = freq[iter_freq_n]
                        freq_m = freq[0]
                        C[iter_freq_n, 0] = inner_prod_rbf_2(freq_n, freq_m, epsilon, rbf_type);
                for iter_freq_m in range(freq.size):
                        freq_n = freq[0]
                        freq_m = freq[iter_freq_m]
                        R[0, iter_freq_m] = inner_prod_rbf_2(freq_n, freq_m, epsilon, rbf_type);
                end
                out_M_im_temp = toeplitz(C,R);
            else: 
                for iter_freq_n in range(freq.size):
                    for iter_freq_m  in range(freq.size):
                        freq_n = freq[iter_freq_n]
                        freq_m = freq[iter_freq_m]
                        out_M_im_temp[iter_freq_n, iter_freq_m] = inner_prod_rbf_2(freq_n, freq_m, epsilon, rbf_type);
    out_M_im[2:, 2:] = out_M_im_temp;
    return out_M_im

def assemble_M_re(freq, epsilon, rbf_type, der_used):
    std_freq = np.std(np.diff(-np.log(freq)));
    mean_freq = np.mean(np.diff(-np.log(freq)));
    out_M_re = np.zeros((freq.size+2, freq.size+2))
    out_M_re_temp = np.zeros((freq.size,freq.size));
    R=np.zeros((1,freq.size));
    C=np.zeros((freq.size,1))
    if der_used == '1st-order':
            if std_freq/mean_freq<1:
                for iter_freq_n in range(freq.size):
                        freq_n = freq[iter_freq_n]
                        freq_m = freq[0]
                        C[iter_freq_n, 0] = inner_prod_rbf(freq_n, freq_m, epsilon, rbf_type);
                for iter_freq_m in range(freq.size):
                        freq_n = freq[0]
                        freq_m = freq[iter_freq_m]
                        R[0, iter_freq_m] = inner_prod_rbf(freq_n, freq_m, epsilon, rbf_type);
                out_M_re_temp = toeplitz(C,R);
            else:
                for iter_freq_n in range(freq.size):
                    for iter_freq_m in range(freq.size):
                        freq_n = freq[iter_freq_n]
                        freq_m = freq[iter_freq_m]
                        out_M_re_temp[iter_freq_n, iter_freq_m] = inner_prod_rbf(freq_n, freq_m, epsilon, rbf_type);
    else:
            if std_freq/mean_freq<1:  
                for iter_freq_n in range(freq.size):
                        freq_n = freq[iter_freq_n]
                        freq_m = freq[0]
                        C[iter_freq_n, 0] = inner_prod_rbf_2(freq_n, freq_m, epsilon, rbf_type);
                for iter_freq_m in range(freq.size):
                        freq_n = freq[0]
                        freq_m = freq[iter_freq_m]
                        R[0, iter_freq_m] = inner_prod_rbf_2(freq_n, freq_m, epsilon, rbf_type);
                out_M_re_temp = toeplitz(C,R);
            else:
                for iter_freq_n in range(freq.size):
                    for iter_freq_m in range(freq.size):
                        freq_n = freq[iter_freq_n]
                        freq_m = freq[iter_freq_m]
                        out_M_re_temp[iter_freq_n, iter_freq_m] = inner_prod_rbf_2(freq_n, freq_m, epsilon, rbf_type);
    out_M_re[2:, 2:] = out_M_re_temp
    return out_M_re

def map_array_to_gamma(freq_map, freq_coll, x, epsilon, rbf_type):   
    y0 = -np.log(freq_coll);   
    if rbf_type == 'gaussian':
            rbf = lambda y: np.exp(-np.power((epsilon*(y-y0)),2))
    else:
        raise Exception('Error')    
    out_gamma = np.zeros(freq_map.shape)
    for iter_freq_map in range(freq_map.size):
        freq_map_loc = freq_map[iter_freq_map];
        y = -np.log(freq_map_loc)
        out_gamma[iter_freq_map] = x.T@rbf(y)
    return out_gamma

class Data:
    freq = None

def drt(Z_exp,fr,lam = 1e-8,coeff = 7, L=0,abstol = 1e-15, reltol = 1e-10, maxiters = int(1e+5) ):
    data = Data()
    data.freq = fr;
    rbf_gaussian_4_FWHM = lambda x: np.exp(-np.power(x,2))-1/2
    rbf_C2_matern_4_FWHM = lambda x: np.multiply(np.exp(-np.abs(x)),(1+np.abs(x)))-1/2
    rbf_C4_matern_4_FWHM = lambda x: 1/3.*np.multiply(np.exp(-np.abs(x)),(3+3*np.abs(x)+np.power(np.abs(x),2)))-1/2;
    rbf_C6_matern_4_FWHM = lambda x: 1./15.*np.multiply(np.exp(-np.abs(x)),(15+15*np.abs(x)+6*np.power(abs(x),2)+np.power(np.abs(x),3)))-1/2;
    rbf_inverse_quadratic_4_FWHM = lambda x:  np.power(1+np.power(x,2),-1)-1/2;
    rbf_inverse_quadric_4_FWHM = lambda x:  np.power(sqrt(1+np.power(x,2)),-1)-1/2;
    rbf_cauchy_4_FWHM = lambda x:  np.power((1+np.abs(x)),-1)-1/2
    FWHM_gaussian = lambda x: 2*opt.fzero(rbf_gaussian_4_FWHM, 1);
    FWHM_C2_matern = lambda x:2*opt.fzero( rbf_C2_matern_4_FWHM, 1);
    FWHM_C4_matern = lambda x:2*opt.fzero(rbf_C4_matern_4_FWHM, 1);
    FWHM_C6_matern = lambda x:2*opt.fzero(rbf_C6_matern_4_FWHM, 1);
    FWHM_inverse_quadratic = lambda x:2*opt.fzero(rbf_inverse_quadratic_4_FWHM, 1);
    FWHM_inverse_quadric = lambda x:2*opt.fzero(rbf_inverse_quadric_4_FWHM, 1);
    FWHM_cauchy = lambda x:2*opt.fzero(rbf_cauchy_4_FWHM ,1);
    FWHM_coeff = FWHM_gaussian;
    rbf_type = 'gaussian';
    data_used = 'Combined Re-Im Data';
    integration_algorithm = 'integral';
    pcw = 0;
    shape_control = 'Shape Factor';
    der_used = '1st-order';
    data_exist=0;
    lb_im = np.zeros((data.freq.size+2,1))
    ub_im = np.inf*np.ones((data.freq.size+2,1))
    x_im_0 = np.ones(lb_im.shape)
    lb_re = np.zeros((data.freq.size+2,1))
    ub_re = np.inf*np.ones((data.freq.size+2,1))
    x_re_0 = np.ones(lb_re.shape)
    taumax=np.ceil(np.max(np.log10(np.power(data.freq,-1))))+1;  
    taumin=np.floor(np.min(np.log10(np.power(data.freq,-1))))-1;
    b_re = np.real(Z_exp);
    b_im = -np.imag(Z_exp);                      
    if shape_control == 'Coefficient to FWHM':
            delta = np.mean(np.diff(np.log(np.power(data.freq,-1)))) 
            epsilon  = coeff*FWHM_coeff/delta;
    else:
            epsilon = coeff;
    data.freq_out = np.logspace(-taumin, -taumax, 10*data.freq.size)
    A_re = assemble_A_re(data.freq, epsilon, rbf_type, integration_algorithm)
    A_im = assemble_A_im(data.freq, epsilon, rbf_type, integration_algorithm,L)
    M_re = assemble_M_re(data.freq, epsilon, rbf_type, der_used)
    M_im = assemble_M_im(data.freq, epsilon, rbf_type, der_used)
    H_re,f_re = quad_format(A_re, b_re, M_re, lam);
    H_im,f_im = quad_format(A_im, b_im, M_im, lam);
    H_combined,f_combined = quad_format_combined(A_re, A_im, b_re, b_im, M_re, M_im, lam);
    P = cvxopt.matrix(H_combined, tc='d')
    q = cvxopt.matrix(f_combined, tc='d')
    G = -1*np.eye(H_combined.shape[0])
    h = np.ones((1*(data.freq.size+2),1))
    h[:,0] = lb_re.squeeze()
    G = cvxopt.matrix(G, tc='d')
    h = cvxopt.matrix(h, tc='d')
    solvers.options['show_progress'] = False
    solvers.options['abstol'] = abstol
    solvers.options['reltol'] = reltol
    solvers.options['maxiters'] = maxiters 
    sol = solvers.qp(P,q,G,h)
    x_ridge_combined = sol['x']
    gamma_ridge_combined_fine = map_array_to_gamma(data.freq_out, data.freq, x_ridge_combined[2:], epsilon, rbf_type);
    gamma_ridge_combined_coarse = map_array_to_gamma(data.freq, data.freq, x_ridge_combined[2:], epsilon, rbf_type);
    rl=x_ridge_combined[:2]
    graph=gamma_ridge_combined_fine; 
    tau_res = np.power(data.freq_out,-1)
    return graph, tau_res