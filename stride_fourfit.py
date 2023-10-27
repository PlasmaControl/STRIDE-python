import numpy as np
from scipy.interpolate import CubicSpline, RectBivariateSpline
import scipy.optimize
import scipy.fft
import sys
import os
import copy



def make_metric(psi_grid, theta_grid, m_grid, ro, zo, rzphi):
    
    
    metric_names = ['g11','g22','g33','g23','g31','g12','jac','jac_prime']
    fourfit_metric_names = ['G11','G22','G33','G23','G31','G12','Jmat','Jmat_prime']
    metric = {name:np.zeros((len(psi_grid),len(theta_grid))) for name in metric_names}

    r = np.sqrt(rzphi['r_squared'](psi_grid,theta_grid,grid=True))
    eta = 2*np.pi*(theta_grid + rzphi['delta_eta'](psi_grid,theta_grid,grid=True))
    R = ro + r*np.cos(eta)
    jac = rzphi['jac'](psi_grid,theta_grid,grid=True)
    jac1 = rzphi['jac'](psi_grid,theta_grid,grid=True,dx=1)

    v = np.zeros((3,3,len(psi_grid),len(theta_grid)))
    """compute contravariant basis vectors"""
    v[0,0] = rzphi['r_squared'](psi_grid,theta_grid,grid=True,dx=1)/(2*r*jac)          # 0, 5
    v[0,1] = rzphi['delta_eta'](psi_grid,theta_grid,grid=True,dx=1)*2*np.pi*r/jac      # 0, 5
    v[0,2] = rzphi['delta_phi'](psi_grid,theta_grid,grid=True,dx=1)*R/jac              # 0, 5, 4
    v[1,0] = rzphi['r_squared'](psi_grid,theta_grid,grid=True,dy=1)/(2*r*jac)          # 1, 5
    v[1,1] = (1+rzphi['delta_eta'](psi_grid,theta_grid,grid=True,dy=1))*2*np.pi*r/jac  # 1, 5
    v[1,2] = rzphi['delta_phi'](psi_grid,theta_grid,grid=True,dy=1)*R/jac              # 1, 3, 5
    v[2,2] = 2*np.pi*R/jac                                                             # 2, 3, 4
    """compute metric tensor components"""
    metric['g11'] = np.sum(v[0,:]**2,axis=0)*jac      # g11
    metric['g22'] = np.sum(v[1,:]**2,axis=0)*jac      # g22
    metric['g33'] = v[2,2]**2*jac                     # g33
    metric['g23'] = v[1,2]*v[2,2]*jac                 # g23
    metric['g31'] = v[2,2]*v[0,2]*jac                 # g31
    metric['g12'] = np.sum(v[0,:]*v[1,:],axis=0)*jac  # g12
    metric['jac'] = jac
    metric['jac_prime'] = jac1

    delta_m = (m_grid[np.newaxis,:] - m_grid[:,np.newaxis]).T
    fourfit_metric = {name1: scipy.fft.fft(metric[name2][:,:-1]/(len(theta_grid)-1), axis=1)[:,delta_m] 
                      for name1, name2 in zip(fourfit_metric_names, metric_names)}
    
    return metric, fourfit_metric
    
def make_primitive_matrices(psi_grid,m_grid,nn,psio,profiles,fourfit_metric,**kwargs):
    
    G11 = fourfit_metric['G11']
    G22 = fourfit_metric['G22']
    G33 = fourfit_metric['G33']
    G23 = fourfit_metric['G23']
    G31 = fourfit_metric['G31']
    G12 = fourfit_metric['G12']
    Jmat = fourfit_metric['Jmat']
    Jmat_prime = fourfit_metric['Jmat_prime']
    
    chi_prime = 2*np.pi*psio
    M = np.diag(m_grid)
    mpert = len(m_grid)
    Imat = np.array([np.eye(mpert) for psi in psi_grid])
    P_prime = profiles['P'](psi_grid,1)[:,np.newaxis,np.newaxis]
    q_prime = profiles['q'](psi_grid,1)[:,np.newaxis,np.newaxis]
    q = profiles['q'](psi_grid)[:,np.newaxis,np.newaxis]
    f_prime = profiles['F'](psi_grid,1)[:,np.newaxis,np.newaxis]
    Q = np.stack([np.diag(m_grid-nn*qq) for qq in q[:,0,0]],axis=0)

    A = (2*np.pi)**2*(nn*(nn*G22 + np.matmul(G23,M)) + np.matmul(M,(nn*G23 + np.matmul(G33,M))))
    B = -2*np.pi*1j*chi_prime*(nn*(G22 + q*G23) + np.matmul(M,G23 + q*G33))
    C = -2*np.pi*1j*chi_prime*q_prime*(nn*G23 + np.matmul(M,G33)) \
        - (2*np.pi)**2*chi_prime*np.matmul((nn*G12 + np.matmul(M,G31)),Q) \
         + 2*np.pi*1j*(f_prime*Q - nn*P_prime/chi_prime*Jmat)
    D = chi_prime**2*((G22 + q*G23) + q*(G23 + q*G33))
    E = chi_prime*(chi_prime*q_prime*(G23 + q*G33)) \
        - 2*np.pi*1j*chi_prime**2*(G12 + q*G31) + P_prime*Jmat
    H = (q_prime*chi_prime)**2*G33 + 2*np.pi*1j*chi_prime**2*q_prime*(np.matmul(M,G31)-np.matmul(G31,M)) \
        + (2*np.pi*chi_prime)**2*np.matmul(np.matmul(Q,G11),Q) + P_prime*Jmat_prime - f_prime*q_prime*chi_prime*Imat

    primitive_matrices = {'A':A, 'B':B, 'C':C, 'D':D, 'E':E, 'H':H, 'Q':Q, 'M':M}
    
    return primitive_matrices
    
def make_coefficient_matrices(psi_grid,m_grid,nn,psio,profiles,fourfit_metric,
                              primitive_matrices=None,splinetype1d=CubicSpline,**kwargs):
 
    G11 = fourfit_metric['G11']
    G22 = fourfit_metric['G22']
    G33 = fourfit_metric['G33']
    G23 = fourfit_metric['G23']
    G31 = fourfit_metric['G31']
    G12 = fourfit_metric['G12']
    Jmat = fourfit_metric['Jmat']
    Jmat_prime = fourfit_metric['Jmat_prime']
    
    if primitive_matrices is None:
        primitive_matrices = make_primitive_matrices(psi_grid,m_grid,nn,psio,profiles,fourfit_metric)
    A = primitive_matrices['A']
    B = primitive_matrices['B']
    C = primitive_matrices['C']
    D = primitive_matrices['D']
    E = primitive_matrices['E']
    H = primitive_matrices['H']
    M = primitive_matrices['M']
    Q = primitive_matrices['Q']

    chi_prime = 2*np.pi*psio
    mpert = len(m_grid)
    Imat = np.array([np.eye(mpert) for psi in psi_grid])
    P_prime = profiles['P'](psi_grid,1)[:,np.newaxis,np.newaxis]
    q_prime = profiles['q'](psi_grid,1)[:,np.newaxis,np.newaxis]
    q = profiles['q'](psi_grid)[:,np.newaxis,np.newaxis]
    f_prime = profiles['F'](psi_grid,1)[:,np.newaxis,np.newaxis]
    
    Fbar = (chi_prime/nn)**2*(G33 - np.matmul((nn*G23 + np.matmul(G33,M))*(2*np.pi)**2, 
                                              np.linalg.solve(A,(nn*G23 + np.matmul(M,G33)))))
    Kbar = chi_prime/nn*(2*np.pi*1j*np.matmul(nn*G23 + np.matmul(G33,M), np.linalg.solve(A,C)) \
                         - (chi_prime*q_prime*G33 - 2*np.pi*1j*chi_prime*np.matmul(G31,Q) - f_prime*Imat))
    G = H - np.matmul(C.conj().transpose(0,2,1),np.linalg.solve(A,C))

    Fbar = (Fbar + Fbar.conj().transpose(0,2,1))/2
    G = (G + G.conj().transpose(0,2,1))/2

    F = np.matmul(np.matmul(Q,Fbar),Q)
    K = np.matmul(Q,Kbar)

    coeff_matrices = {'F':F,
                     'Fbar':Fbar,
                     'G':G,
                     'K':K,
                     'Kbar':Kbar,
                     'Q':Q}
    
    coeff_splines = {key + '_spline': splinetype1d(psi_grid,val,axis=0) for key,val in coeff_matrices.items()}
    
    return coeff_matrices, coeff_splines