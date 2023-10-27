import numpy as np
from scipy.interpolate import CubicSpline, RectBivariateSpline
import scipy.optimize


class Bfield():
    def __init__(self,psirz,F,P,psio):
        self.psirz = psirz
        self.psio = psio
        self.F_func = F
        self.P_func = P
    def psi(self,r,z, dr=0,dz=0,grid=False):
        return self.psirz(r,z,dx=dr,dy=dz,grid=grid)
    def F(self,r,z, dpsi=0, grid=False):
        return self.F_func(1-self.psi(r,z,grid=grid)/self.psio, nu=dpsi)
    def F_prime(self,r,z, dpsi=0, grid=False):
        return self.F_func(1-self.psi(r,z,grid=grid)/self.psio,nu=dpsi+1)
    def P(self,r,z, dpsi=0, grid=False):
        return self.P_func(1-self.psi(r,z,grid=grid)/self.psio, nu=dpsi)
    def P_prime(self,r,z, dpsi=0, grid=False):
        return self.P_func(1-self.psi(r,z,grid=grid)/self.psio,nu=dpsi+1)
    def psi_r(self,r,z,dr=0,dz=0,grid=False):
        return self.psirz(r,z, dx=dr+1, dy=dz, grid=grid)
    def psi_z(self,r,z,dr=0,dz=0,grid=False):
        return self.psirz(r,z, dx=dr, dy=dz+1, grid=grid)
    def Br(self,r,z,dr=0,dz=0,grid=False):
        return self.psi_z(r,z,dr=dr,dz=dz,grid=grid)/r
    def Bz(self,r,z,dr=0,dz=0,grid=False):
        return -self.psi_r(r,z,dr=dr,dz=dz,grid=grid)/r
    def psi_rr(self,r,z,dr=0,dz=0,grid=False):
        return self.psirz(r,z,dx=dr+2,dy=dy,grid=grid)
    def psi_rz(self,r,z,dr=0,dz=0,grid=False):
        return self.psirz(r,z,dx=dr+1,dy=dz+1, grid=grid)
    def psi_zz(self,r,z,dr=0,dz=0,grid=False):
        return self.psirz(r,z,dx=dr,dy=dz+2,grid=grid)
    def Br_r(self,r,z,dr=0,dz=0,grid=False):
        return (self.psi_rz(r,z,dr=dr,dz=dz,grid=grid) - self.Br(r,z,dr=dr,dz=dz,grid=grid))/r
    def Br_z(self,r,z,dr=0,dz=0,grid=False):
        return -self.psi_zz(r,z,dr=dr,dz=dz,grid=grid)/r
    def Bz_r(self,r,z,dr=0,dz=0,grid=False):
        return -(self.psi_rr(r,z,dr=dr,dz=dz,grid=grid) + self.Bz(r,z,dr=dr,dz=dz,grid=grid))/r
    def Bz_z(self,r,z,dr=0,dz=0,grid=False):
        return -self.psi_rz(r,z,dr=dr,dz=dz,grid=grid)/r
    def Bp(self,r,z,dr=0,dz=0,grid=False):
        return np.sqrt(self.Br(r,z,dr=dr,dz=dz,grid=grid)**2 + self.Bz(r,z,dr=dr,dz=dz,grid=grid)**2)
    def Bt(self,r,z,grid=False):
        return self.F(r,z,dpsi=0,grid=grid)/r
    def B(self,r,z,grid=False):
        return np.sqrt(self.Bp(r,z,grid=grid)**2 + self.Bt(r,z,grid=grid)**2)

def find_O_point(bf,guess=None):
    eps = 1e-6
    nnstep = 2048
    
    rmin = bf.psirz.get_knots()[0].min()
    rmax = bf.psirz.get_knots()[0].max()
    zmin = bf.psirz.get_knots()[1].min()
    zmax = bf.psirz.get_knots()[1].max()
    if guess is None:
        """find where Bz changes sign along midplane"""

        rgrid = bf.psirz.get_knots()[0]
        z = bf.psirz.get_knots()[1].mean()
        Bz = bf.Bz(rgrid,z)
        sign_change = Bz[1:]*Bz[:-1]
        sign_change_idx = np.where(sign_change<0)[0][0]
        r = rgrid[sign_change_idx]

    else:
        r = guess[0]
        z = guess[1]
    
    def Bsquared(x):
        return(bf.Br(x[0],x[1])**2 + bf.Bz(x[0],x[1])**2)
    
    out = scipy.optimize.minimize(Bsquared,(r,z),bounds=[(rmin,rmax),(zmin,zmax)])
    
    if out.success:
        return out.x
    else:
        raise Exception('Could not find O point')
        
def find_X_points(bf,ro,zo):
    
    rmin = bf.psirz.get_knots()[0].min()
    rmax = bf.psirz.get_knots()[0].max()
    
    def psi(r):
        return bf.psi(r,zo)
    # inboard
    r = (rmin + ro)/2
    bracket = (rmin,ro) if psi(rmin)*psi(ro)<0 else None
    x1 = r + 1e-4 if bracket is None else None
    out = scipy.optimize.root_scalar(psi,x0=r,x1=x1,bracket=bracket,xtol=1e-12,rtol=1e-12)
    if out.converged:
        r_sep_in = out.root
    else:
        raise Exception('Could not find inboard LCFS')

    # outboard
    r = (rmax + ro)/2
    bracket = (ro,rmax) if psi(ro)*psi(rmax)<0 else None
    x1 = r + 1e-4 if bracket is None else None
    out = scipy.optimize.root_scalar(psi,x0=r,x1=x1,bracket=bracket,xtol=1e-12,rtol=1e-12)
    if out.converged:
        r_sep_out = out.root
    else:
        raise Exception('Could not find inboard LCFS')
    return r_sep_in, r_sep_out


def direct_fl_der(eta,y,bf,ro,zo,power_bp,power_b,power_r):
       
    dy = np.zeros_like(y)
    
    cos_eta = np.cos(eta)
    sin_eta = np.sin(eta)
    
    r = y[1]
    R = ro + r*cos_eta 
    Z = zo + r*sin_eta
    
    Br = bf.Br(R,Z)
    Bz = bf.Bz(R,Z)
    Bp = bf.Bp(R,Z)
    Bt = bf.Bt(R,Z)
    B = bf.B(R,Z)
    jac = Bp**power_bp * B**power_b / R**power_r

    dy[0] = r/(Bz*cos_eta - Br*sin_eta)
    dy[1] = dy[0]*(Br*cos_eta + Bz*sin_eta)
    dy[2] = dy[0]/(R**2)
    dy[3] = dy[0]*jac
    return dy


def direct_fl_int(psi_n,bf,ro,zo,rs2,psio,power_bp,power_b,power_r):

    psi = psio*(1-psi_n)
    r = ro + np.sqrt(psi_n)*(rs2-ro)
    z = zo
    def psi_diff(r):
        return psi - bf.psi(r,z)

    out = scipy.optimize.root_scalar(psi_diff,x0=r,bracket=(ro,rs2),xtol=1e-12,rtol=1e-12)
    if out.converged:
        r = out.root
    else:
        raise Exception('Could not find starting pt on flux surface')

    y0 = np.zeros(4)
    y0[1] = np.sqrt((r-ro)**2 + (z-zo)**2)
    
    out = scipy.integrate.solve_ivp(direct_fl_der,
                                    [0,2*np.pi],
                                    y0, 
                                    method='RK45',
                                    vectorized=True,
                                    rtol=1e-12,
                                    atol=1e-12,
                                    args=(bf,ro,zo,power_bp,power_b,power_r))
    if out.success:
        return (out.t,out.y)
    else:
        raise Exception('Could not integrate along flux surface')
        


def direct_fl_der_vectorized(eta,y,bf,ro,zo,power_bp,power_b,power_r):
        
    y = y.reshape((-1,4))
    dy = np.zeros_like(y)
    
    cos_eta = np.cos(eta)
    sin_eta = np.sin(eta)
    
    r = y[:,1]
    R = ro + r*cos_eta
    Z = zo + r*sin_eta

    Br = bf.Br(R,Z,grid=False)
    Bz = bf.Bz(R,Z,grid=False)
    
    Bp = np.sqrt(Br**2 + Bz**2) if power_bp else 1
    Bt = bf.Bt(R,Z,grid=False) if power_b else 1
    B = np.sqrt(Bp**2 + Bt**2) if power_b else 1
    jac=Bp**power_bp * B**power_b / R**power_r
    
    dy[:,0] = r/(Bz*cos_eta - Br*sin_eta)
    dy[:,1] = dy[:,0]*(Br*cos_eta + Bz*sin_eta)
    dy[:,2] = dy[:,0]/(R**2)
    dy[:,3] = dy[:,0]*jac
    
    return dy.flatten()


def direct_fl_int_vectorized(psi_grid,bf,ro,zo,r_sep_out,psio,power_bp,power_b,power_r):

    psi = psio*(1-psi_grid)
    r = ro + np.sqrt(psi_grid)*(r_sep_out-ro)
    z = zo
    
    def psi_diff(r):
        return psi - bf.psi(r,z)

    out = scipy.optimize.root(psi_diff,x0=r,tol=1e-12)
    if out.success:
        r = out.x
    else:
        raise Exception('Could not find starting pt on flux surface')

    y0 = np.zeros((len(psi_grid),4))
    y0[:,1] = np.sqrt((r-ro)**2 + (z-zo)**2)
    
    out = scipy.integrate.solve_ivp(direct_fl_der_vectorized,
                                    [0,2*np.pi],
                                    y0.flatten(), 
                                    method='RK45',
                                    vectorized=True,
                                    rtol=1e-12,
                                    atol=1e-12,
                                    args=(bf,ro,zo,power_bp,power_b,power_r))
    if out.success:
        return (out.t,out.y.T.reshape(out.t.size,-1,4))
    else:
        raise Exception('Could not integrate along flux surface')
        
        
def convert_efit_equilibrium(g,mpsi=129,mtheta=129,psilow=0.01,psihigh=0.98, return_arrs=False, splinetype1d=CubicSpline):

    mu0 = np.pi*4e-7

    R_grid = np.linspace(g['rleft'],g['rleft']+g['rdim'],g['nw'])
    Z_grid = np.linspace(-g['zdim']/2,g['zdim']/2,g['nh'])
    psio = g['boundary_flux'] - g['axis_flux']
    psigrid_in = np.linspace(0,1,g['nw'])
    fpol = np.abs(g['fpol'])
    pres = np.maximum(g['pres']*mu0,0)
    qpsi = g['qpsi']
    psirz_arr = g['boundary_flux'] - g['psirz']
    if psio<0:
        psio = -psio
        psirz_arr = -psirz_arr

    direct_out = process_direct_equilibrium(R_grid, Z_grid, psirz_arr, 
                                            psigrid_in, pres, fpol,
                                            psilow, psihigh, mpsi, 
                                            mtheta, psio, return_arrs, splinetype1d)
    
    psi_grid = direct_out[0]
    theta_grid = direct_out[1]
    straight_field_line_coords = direct_out[2]
    profiles = direct_out[3]
    ro = direct_out[4]
    zo = direct_out[5]
    if return_arrs:
        straight_field_line_coords_arrs = direct_out[6]
        profiles_arrs = direct_out[7]
        bf = direct_out[8]
        temp_data = direct_out[9]
        
    if return_arrs:
        return psi_grid, theta_grid, straight_field_line_coords, profiles, \
                ro, zo, psio, straight_field_line_coords_arrs, profiles_arrs, bf, temp_data
    else:
        return psi_grid, theta_grid, straight_field_line_coords, profiles, ro, zo, psio

    
def process_direct_equilibrium(R_grid, Z_grid, psirz_arr, psigrid_in, pres,fpol, 
                               psilow, psihigh, mpsi, mtheta, psio, return_arrs=False,
                               splinetype1d=CubicSpline):

    psirz = RectBivariateSpline(R_grid,Z_grid,psirz_arr.T)
    P = splinetype1d(psigrid_in,pres)
    F = splinetype1d(psigrid_in,fpol)

    psi_grid = psilow + (psihigh-psilow)*np.sin(np.linspace(0,1,mpsi)*np.pi/2)**2
    theta_grid = np.linspace(0,1,mtheta)
    straight_field_line_coords_arrs = {'r_squared':np.zeros((mpsi,mtheta)),
                                       'delta_eta':np.zeros((mpsi,mtheta)),
                                       'delta_phi':np.zeros((mpsi,mtheta)),
                                       'jac':np.zeros((mpsi,mtheta))}

    profiles_arrs = {'F':np.zeros(mpsi),
                     'P':np.zeros(mpsi),
                     'jac': np.zeros(mpsi),
                     'q': np.zeros(mpsi)}

    bf = Bfield(psirz,F,P,psio)

    ro,zo = find_O_point(bf)
    r_sep_in, r_sep_out = find_X_points(bf,ro,zo)
    
    
    power_bp = 0
    power_b = 0
    power_r = 0
    
    eta, y_out = direct_fl_int_vectorized(psi_grid,bf,ro,zo,r_sep_out,psio,power_bp,power_b,power_r)
    
    theta_n = y_out[:,:,3]/y_out[-1,:,3]
    r_squared = y_out[:,:,1]**2
    delta_eta = eta[:,np.newaxis]/(2*np.pi) - theta_n
    delta_phi = F(psi_grid).squeeze()*(y_out[:,:,2] - theta_n*y_out[-1,:,2])
    jac = y_out[:,:,0]/y_out[-1,:,0] - theta_n

    profiles_arrs['F'] = F(psi_grid).squeeze()*2*np.pi
    profiles_arrs['P'] = P(psi_grid).squeeze()
    profiles_arrs['jac'] = y_out[-1,:,0]*2*np.pi*psio
    profiles_arrs['q'] = y_out[-1,:,2]*F(psi_grid).squeeze()/(2*np.pi)
    
    temp_data = np.stack([r_squared,delta_eta,delta_phi,jac],axis=2)
    temp_data[-1,:,:] = temp_data[0,:,:]
    
    for ipsi in range(mpsi):

        temp_spline = splinetype1d(theta_n[:,ipsi],temp_data[:,ipsi,:],axis=0,bc_type='periodic')
        temp_out = temp_spline(theta_grid)
        temp_out_der = temp_spline(theta_grid, nu=1)
        straight_field_line_coords_arrs['r_squared'][ipsi,:] = temp_out[:,0]
        straight_field_line_coords_arrs['delta_eta'][ipsi,:] = temp_out[:,1]
        straight_field_line_coords_arrs['delta_phi'][ipsi,:] = temp_out[:,2]
        straight_field_line_coords_arrs['jac'][ipsi,:] = (1 + temp_out_der[:,3])*y_out[-1,ipsi,0]*2*np.pi*psio


    straight_field_line_coords = {key:RectBivariateSpline(psi_grid,theta_grid,val) 
                                  for key,val in straight_field_line_coords_arrs.items()}
    profiles = {key:splinetype1d(psi_grid,val) for key,val in profiles_arrs.items()}

    straight_field_line_coords_arrs['psi_grid'] = psi_grid
    straight_field_line_coords_arrs['theta_grid'] = theta_grid
    profiles_arrs['psi_grid'] = psi_grid

    
    if return_arrs:
        return psi_grid, theta_grid, straight_field_line_coords, profiles, \
                ro, zo, straight_field_line_coords_arrs, profiles_arrs, bf, temp_data
    else:
        return psi_grid, theta_grid, straight_field_line_coords, profiles, ro, zo
