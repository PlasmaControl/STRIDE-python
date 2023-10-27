import numpy as np
import scipy.fft
from gfile_helpers import read_gfile



colorblind_colors = [(0.0000, 0.4500, 0.7000), # blue
                     (0.8359, 0.3682, 0.0000), # vermillion
                     (0.0000, 0.6000, 0.5000), # bluish green
                     (0.9500, 0.9000, 0.2500), # yellow
                     (0.3500, 0.7000, 0.9000), # sky blue
                     (0.8000, 0.6000, 0.7000), # reddish purple
                     (0.9000, 0.6000, 0.0000), # orange
                     (0.5000, 0.5000, 0.5000)] # grey

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams, cycler
matplotlib.rcdefaults()
rcParams['font.family'] = 'DejaVu Serif'
rcParams['mathtext.fontset'] = 'cm'
rcParams['font.size'] = 10
rcParams['figure.facecolor'] = (1,1,1,1)
rcParams['figure.figsize'] = (8,6)
rcParams['figure.dpi'] = 141
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
rcParams['axes.labelsize'] =  'small'
rcParams['axes.titlesize'] = 'medium'
rcParams['lines.linewidth'] = 1.5
rcParams['lines.solid_capstyle'] = 'round'
rcParams['lines.dash_capstyle'] = 'round'
rcParams['lines.dash_joinstyle'] = 'round'
rcParams['xtick.labelsize'] = 'x-small'
rcParams['ytick.labelsize'] = 'x-small'
rcParams['legend.fontsize'] = 'small'
color_cycle = cycler(color=colorblind_colors)
rcParams['axes.prop_cycle'] =  color_cycle

labelsize=10
ticksize=8






def plot_profiles(psi_grid, F, P, jac, q, figsize=(8,6), **kwargs):
    
    fig,ax = plt.subplots(2,2, figsize=figsize)
    ax[0,0].plot(psi_grid,F)
    ax[0,0].set_xlabel('$\psi$')
    ax[0,0].set_ylabel('$F$')

    ax[0,1].plot(psi_grid,P)
    ax[0,1].set_xlabel('$\psi$')
    ax[0,1].set_ylabel('$\mu_0 P$')

    ax[1,0].plot(psi_grid,jac)
    ax[1,0].set_xlabel('$\psi$')
    ax[1,0].set_ylabel('$\mathcal{J}$')

    ax[1,1].plot(psi_grid,q)
    ax[1,1].set_xlim((0,None))
    ax[1,1].set_xlabel('$\psi$')
    ax[1,1].set_ylabel('$q$')

    plt.subplots_adjust(wspace=.3)
    
    return fig, ax
    
    
def plot_straight_field_line_coords(psi_grid,theta_grid,r_squared,delta_eta,delta_phi,jac,figsize=(8,6),levels=30, **kwargs):

    fig, ax = plt.subplots(2,2, figsize=figsize)

    im1 = ax[0,0].contourf(theta_grid,psi_grid,r_squared,levels=levels)
    fig.colorbar(im1,ax=ax[0,0], shrink=0.9)
    ax[0,0].set_xlabel('$\\theta$', fontsize=10)
    ax[0,0].set_ylabel('$\psi$', fontsize=10)
    ax[0,0].set_title('$r^2$', fontsize=10)

    im2 = ax[0,1].contourf(theta_grid,psi_grid,delta_eta,levels=levels)
    fig.colorbar(im2,ax=ax[0,1], shrink=0.9)
    ax[0,1].set_xlabel('$\\theta$', fontsize=10)
    ax[0,1].set_ylabel('$\psi$', fontsize=10)
    ax[0,1].set_title('$\Delta \eta$', fontsize=10)

    im3 = ax[1,0].contourf(theta_grid,psi_grid,delta_phi,levels=levels)
    fig.colorbar(im3,ax=ax[1,0], shrink=0.9)
    ax[1,0].set_xlabel('$\\theta$', fontsize=10)
    ax[1,0].set_ylabel('$\psi$', fontsize=10)
    ax[1,0].set_title('$\Delta \phi$', fontsize=10)

    im4 = ax[1,1].contourf(theta_grid,psi_grid,jac,levels=levels)
    fig.colorbar(im4,ax=ax[1,1], shrink=0.9)
    ax[1,1].set_xlabel('$\\theta$', fontsize=10)
    ax[1,1].set_ylabel('$\psi$', fontsize=10)
    ax[1,1].set_title('$\mathcal{J}$', fontsize=10)

    plt.subplots_adjust(hspace=.3)
    
    return fig, ax


def plot_metric(psi_grid,theta_grid,metric,figsize=None,levels=30,fft=False,**kwargs):

    metric_labels = {'g11':'$g_{11} = |\\nabla\\theta \\times \\nabla\zeta |^2$',
                     'g22':'$g_{22} = |\\nabla\zeta \\times \\nabla\psi |^2$',
                     'g33':'$g_{33} = |\\nabla\psi \\times \\nabla\\theta |^2$',
                     'g23':'$g_{23} = (\\nabla\zeta \\times \\nabla\psi) \cdot (\\nabla\psi \\times \\nabla\\theta)$',
                     'g31':'$g_{31} = (\\nabla\psi \\times \\nabla\\theta) \cdot (\\nabla\\theta \\times \\nabla\zeta)$',
                     'g12':'$g_{12} = (\\nabla\\theta \\times \\nabla\zeta) \cdot (\\nabla\zeta \\times \\nabla\psi)$',
                     'jac':'$\mathcal{J}$',
                     'jac_prime':'$d\mathcal{J}/d\psi$'}

    fourfit_metric_labels = {'g11':'$G_{11} = \mathcal{F}[|\\nabla\\theta \\times \\nabla\zeta |^2]$',
                             'g22':'$G_{22} = \mathcal{F}[|\\nabla\zeta \\times \\nabla\psi |^2]$',
                             'g33':'$G_{33} = \mathcal{F}[|\\nabla\psi \\times \\nabla\\theta |^2]$',
                             'g23':'$G_{23} = \mathcal{F}[(\\nabla\zeta \\times \\nabla\psi) '
                                 + '\cdot (\\nabla\psi \\times \\nabla\\theta)]$',
                             'g31':'$G_{31} = \mathcal{F}[(\\nabla\psi \\times \\nabla\\theta) '
                                 + '\cdot (\\nabla\\theta \\times \\nabla\zeta)]$',
                             'g12':'$G_{12} = \mathcal{F}[(\\nabla\\theta \\times \\nabla\zeta) '
                                 + '\cdot (\\nabla\zeta \\times \\nabla\psi)]$',
                             'jac':'$\mathcal{F}[\mathcal{J}]$',
                             'jac_prime':'$\mathcal{F}[d\mathcal{J}/d\psi]$'}
    
    if not fft:
        figsize=(8,8) if figsize is None else figsize
        fig, ax = plt.subplots(4,2,figsize=figsize)
        ax = ax.flatten()
        for i, key in enumerate(list(metric_labels.keys())):
            cm = ax[i].contourf(theta_grid,psi_grid,metric[key],levels=levels)  
            ax[i].set_xlabel('$\\theta$', fontsize=10)
            ax[i].set_ylabel('$\psi$', fontsize=10)
            ax[i].set_title(metric_labels[key],fontsize=10)
            ax[i].tick_params(axis='both', which='both', labelsize=6)
            cb = fig.colorbar(cm, ax=ax[i])
            cb.ax.tick_params(labelsize=6)
        plt.subplots_adjust(hspace=.6)

    else:
        figsize=(8,16) if figsize is None else figsize
        cbar_ticks = [-np.pi,-np.pi/2,0,np.pi/2,np.pi]
        cbar_labels = ['$-\pi$','$-\pi/2$','$0$','$+\pi/2$','$+\pi$',]
        
        fig, ax = plt.subplots(8,2,figsize=figsize)
        for i, key in enumerate(list(metric_labels.keys())):
            x = scipy.fft.fft(metric[key][:,:-1]/(len(theta_grid)-1), axis=1)
            x = scipy.fft.fftshift(x,axes=1)
            m_grid = np.arange(-x.shape[1]/2,x.shape[1]/2)
            cm = ax[i,0].contourf(m_grid,psi_grid,np.abs(x),levels=levels)  
            ax[i,0].set_xlabel('$m$', fontsize=10)
            ax[i,0].set_ylabel('$\psi$', fontsize=10)
            ax[i,0].set_title('Amp. ' + fourfit_metric_labels[key],fontsize=10)
            ax[i,0].tick_params(axis='both', which='both', labelsize=6)
            cb = fig.colorbar(cm, ax=ax[i,0])
            cb.ax.tick_params(labelsize=6)
            
            cm = ax[i,1].contourf(m_grid,psi_grid,np.angle(x),levels=levels)  
            ax[i,1].set_xlabel('$m$', fontsize=10)
            ax[i,1].set_ylabel('$\psi$', fontsize=10)
            ax[i,1].set_title('Arg. ' + fourfit_metric_labels[key],fontsize=10)
            ax[i,1].tick_params(axis='both', which='both', labelsize=6)
            cb = fig.colorbar(cm, ax=ax[i,1],ticks=cbar_ticks)
            cb.ax.tick_params(labelsize=6)
            cb.ax.set_yticklabels(cbar_labels)
        plt.subplots_adjust(hspace=.6)
        

    return fig, ax