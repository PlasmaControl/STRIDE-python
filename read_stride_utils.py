import numpy as np
import os
import sys
import subprocess
from gfile_helpers import write_gfile
import time

def percent_err(x,y):
    return 100*np.abs(x-y)/(np.abs(x) + np.abs(y) + np.finfo(np.float32).eps)

def flatten(x):
    """Flattens a nested list or tuple
    Args:
        x (list or tuple): nested list or tuple of lists or tuples to flatten
    Returns:
        x (list): flattened input
    """
    
    if isinstance(x, list) or isinstance(x, tuple):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]
    
def str2complex(s):
    """parses strings of the form (re,im) into complex numbers"""
    
    re = float(s[1:-1].split(',')[0])
    im = float(s[1:-1].split(',')[1])
    return complex(re,im)

def str2complex2(s):
    """parses strings of the form re +i*im into complex numbers"""
    
    for i, char in enumerate(s):
        if char == 'i' or char == 'j':
            idx = i
    sign = s[idx-1]
    real = float(s[:idx-1])
    imag = float(s[idx+2:]) * (1 if sign == '+' else -1)
    return complex(real,imag)






def parse_raw_output(raw_output, return_timing=True, return_interval_details=True):
    """reads terminal output of original code, to get locations of singularities and other related quantities"""

    if isinstance(raw_output,list):
        f = flatten(raw_output)
    else:
        f = raw_output.split('\n')
    psio = float([foo.split()[-1] for foo in f if 'psio = ' in foo][0])
    mpsi = int([foo.split()[-1] for foo in f if 'mpsi = ' in foo][0])
    mtheta = int([foo.split()[-1] for foo in f if 'mtheta = ' in foo][0])
    mdata = [foo for foo in f if "mband" in foo][0].split()
    mlow = int([mdata[i+2] for i in range(len(mdata)-2) if 'mlow' in mdata[i]][0][:-1])
    mhigh = int([mdata[i+2] for i in range(len(mdata)-2) if 'mhigh' in mdata[i]][0][:-1])
    nn = int([mdata[i+2] for i in range(len(mdata)-2) if 'nn' in mdata[i]][0][:-1])
    singid = []
    singpsi = []
    singm = []
    
    if return_interval_details:
        raw_interval_details = [foo for foo in f if foo.startswith('Interval')]
        interval_details = {'interval': np.array([int(foo.split()[1]) for foo in raw_interval_details]),
                            'nsteps': np.array([int(foo.split()[3]) for foo in raw_interval_details]),
                            'nfevals': np.array([int(foo.split()[5]) for foo in raw_interval_details]),
                            'startpsi': np.array([float(foo.split()[7]) for foo in raw_interval_details]),
                            'endpsi': np.array([float(foo.split()[9]) for foo in raw_interval_details]),
                            'interval_time': np.array([float(foo.split()[11]) for foo in raw_interval_details]),
                            }
        interval_details['flops'] = 38/3*interval_details['nfevals'] + 4
    else:
        interval_details = None
        
    if return_timing:
        times = {'equil': float([foo for foo in f if '*** equil-input time=' in foo][0].split()[-1]),
                 'locstab': float([foo for foo in f if '*** locstab time=' in foo][0].split()[-1]),
                 'sing': float([foo for foo in f if '*** sing time=' in foo][0].split()[-1]),
                 'fourfit': float([foo for foo in f if '*** fourfit-tot time=' in foo][0].split()[-1]),
                 'threadalloc': float([foo for foo in f if '*** thread-alloc time=' in foo][0].split()[-1]),
                 'integration': float([foo for foo in f if '*** ode-parallel-integration time=' in foo][0].split()[-1]),
                 'propagation': float([foo for foo in f if '*** ode-propagateLR time=' in foo][0].split()[-1]),
                 'wpmodes': float([foo for foo in f if '*** calc-modes-for-wp time=' in foo][0].split()[-1]),
                 'wpcalc': float([foo for foo in f if '*** wp-calc time=' in foo][0].split()[-1]),
                 'free': float([foo for foo in f if '*** free-run time=' in foo][0].split()[-1]),
                 'parallel': float([foo for foo in f if '*** tot-parallel time=' in foo][0].split()[-1]),
                 'total': float([foo for foo in f if '*** complete-run time=' in foo][0].split()[-1]),
                }
    else:
        times = None
        
    for i, line in enumerate(f):
        if 'Sing# = ' in line:
            singid.append(int(f[i].split()[-1]))
            singpsi.append(float(f[i+3].split()[-1]))
            singm.append(int(f[i+2].split()[-1]))

    singinterval = [line for line in f if ' sing # ' in line]
    singstarts = [float(foo.split()[-2]) for foo in singinterval]
    singends = [float(foo.split()[-1]) for foo in singinterval]

    m_grid = np.arange(mlow,mhigh+1)
    sing = np.array([singstarts,singends,singm]).T
    sing_loc = np.array([0,*singpsi,1])
    return sing, sing_loc, m_grid, mpsi, mtheta, nn, psio, times, interval_details


def readWmat(filename,M):
    """Reads results of original code"""
    
    with open(filename,'r') as foo:
        f = list(foo)
    for i, line in enumerate(f):
        if isinstance(line,str):
            f[i] = line.strip().split()
    f = flatten(f)


    temp = np.array([str2complex(elem) for elem in f])
    W_dcon = temp.reshape((M,M),order='f')
    return W_dcon

def readLmat(filename):
    """Reads values of psi and ODE RHS matrix L(psi) from text file generated by original code"""

    with open(filename,'r') as foo:
        f = list(foo)
    for i, line in enumerate(f):
        if isinstance(line,str):
            f[i] = line.strip().split()
    f = flatten(f)

    psi_idx = []
    for i, line in enumerate(f):
        if '(' not in line:
            psi_idx.append(i)
    psi_idx = np.array(psi_idx)
    nsteps = len(psi_idx)
    np.testing.assert_almost_equal(np.std(np.diff(psi_idx)),0,err_msg='Unequal matrix sizes at different psi locations!')
    N = np.sqrt(np.diff(psi_idx)[0]-1)
    np.testing.assert_almost_equal(N,int(N),err_msg='Non square matrices!')
    N = int(N)
    M = int(N/2)

    psi = np.full((nsteps,),np.nan,dtype=np.float64)
    Ldata = np.full((nsteps,N,N),np.nan,dtype=np.complex128)

    for i, idx in enumerate(psi_idx):
        psi[i] = float(f[idx])
        temp = f[idx+1:idx+N**2+1]
        temp = np.array([str2complex(elem) for elem in temp])
        Ldata[i] = temp.reshape((N,N),order='f')
    assert not np.any(np.isnan(psi))
    assert not np.any(np.isnan(Ldata))

    idx = np.argsort(psi)
    psi = psi[idx]
    Ldata = Ldata[idx]
    
    L = {'psi_grid':psi,
         'L':Ldata}
    
    return L


def read_primitive_matrices(directory,M):
    matrices = {}
    filenames = [foo + 'mat.out' for foo in 'abcdeh']
    for filename in filenames:
       
        with open(directory + '/' + filename,'r') as foo:
            f = list(foo)
        for i, line in enumerate(f):
            if isinstance(line,str):
                f[i] = line.strip().split()
        f = flatten(f)

        psi_idx = []
        for i, line in enumerate(f):
            if '(' not in line:
                psi_idx.append(i)
        psi_idx = np.array(psi_idx)
        nsteps = len(psi_idx)
        np.testing.assert_almost_equal(np.std(np.diff(psi_idx)),0,err_msg='Unequal matrix sizes at different psi locations!')

        mat_id = filename[0].upper()
        matrices[mat_id] = np.full((nsteps,M,M),np.nan,dtype=np.complex128)
        matrices['psi_' + mat_id] = np.full((nsteps,),np.nan,dtype=np.float64)

        offset = np.diff(psi_idx)[0] 

        for i, idx in enumerate(psi_idx):
            matrices['psi_' + mat_id][i] = float(f[idx])
            temp = f[idx+1:idx+offset]
            temp = np.array([str2complex(elem) for elem in temp])
            iqty = 0
            for jpert in range(0,M):
                for ipert in range(0,M):
                    matrices[mat_id][i,ipert,jpert] = temp[iqty]
                    iqty +=1
        assert not np.any(np.isnan(matrices['psi_' + mat_id]))
        assert not np.any(np.isnan(matrices[mat_id]))


        idx = np.argsort(matrices['psi_' + mat_id])
        matrices['psi_' + mat_id] = matrices['psi_' + mat_id][idx]
        matrices[mat_id] = matrices[mat_id][idx]

    psis = np.stack([matrices['psi_' + mat_id] for mat_id in 'ABCDEH'])
    assert np.allclose(np.std(psis,axis=0),0), "different psi grids!"
    fourfit_mats = {mat_id:matrices[mat_id] for mat_id in 'ABCDEH'}
    fourfit_mats['psi_grid'] = psis[0]
    
    return fourfit_mats


def read_metric(filename,mpsi,mtheta):
    
    with open(filename,'r') as foo:
        f = list(foo)
    for i, line in enumerate(f):
        if isinstance(line,str):
            f[i] = line.strip().split()
    f = flatten(f)
    f = np.array(f)
    metric = np.full((mpsi,mtheta,8),np.nan)
    psi_metric = np.full(mpsi,np.nan)
    psi_idx = np.arange(0,len(f),8*mtheta+1)
    for i, idx in enumerate(psi_idx):
        psi_metric[i] = float(f[idx])
        for j in range(8):
            metric[i,:,j] = np.array([float(foo) for foo in f[idx+mtheta*j+1:idx+mtheta*(j+1)+1]])

    assert not np.any(np.isnan(psi_metric))
    assert not np.any(np.isnan(metric))
    idx = np.argsort(psi_metric)
    psi_metric = psi_metric[idx]
    metric = metric[idx]
    
    theta_grid = np.linspace(0,1,mtheta)
    out_metric = {'psi_grid':psi_metric,
                  'theta_grid':theta_grid,
                  'g11':metric[:,:,0],
                  'g22':metric[:,:,1],
                  'g33':metric[:,:,2],
                  'g23':metric[:,:,3],
                  'g31':metric[:,:,4],
                  'g12':metric[:,:,5],
                  'jac':metric[:,:,6],
                  'jac_prime':metric[:,:,7]}

    return out_metric


def read_fourfit_metric(directory,M):
    matrices = {}
    filenames = ['fourG11.out','fourG22.out','fourG33.out','fourG23.out',
                 'fourG31.out','fourG12.out','fourjac.out','fourjac1.out']
    matnames = ['G11','G22','G33','G23','G31','G12','Jmat','Jmat_prime']
    for filename,matname in zip(filenames,matnames):


        with open(directory + '/' + filename,'r') as foo:
            f = list(foo)
        for i, line in enumerate(f):
            if isinstance(line,str):
                f[i] = line.strip().split()
        f = flatten(f)


        psi_idx = np.array([i for i, line in enumerate(f) if '(' not in line])
        nsteps = len(psi_idx)
        np.testing.assert_almost_equal(np.std(np.diff(psi_idx)),0,err_msg='Unequal matrix sizes at different psi locations!')

        offset = np.diff(psi_idx)[0] 
        M = int(offset/2)

        matrices[matname] = np.full((nsteps,M,M),np.nan,dtype=np.complex128)
        matrices['psi_' + matname] = np.full((nsteps,),np.nan,dtype=np.float64)

        for k, idx in enumerate(psi_idx):
            matrices['psi_' + matname][k] = float(f[idx])
            temp = f[idx+1:idx+offset]
            temp = np.array([str2complex(elem) for elem in temp])
            for ii in range(M):
                for jj in range(M):
                    matrices[matname][k,ii,jj] = temp[jj-ii+M-1]

        assert not np.any(np.isnan(matrices['psi_' + matname]))
        assert not np.any(np.isnan(matrices[matname]))


        idx = np.argsort(matrices['psi_' + matname])
        matrices['psi_' + matname] = matrices['psi_' + matname][idx]
        matrices[matname] = matrices[matname][idx]

    psis = np.stack([matrices['psi_' + matname] for matname in matnames])
    assert np.allclose(np.std(psis,axis=0),0), "different psi grids!"

    fourfit_metric_mats = {matname:matrices[matname] for matname in matnames}
    fourfit_metric_mats['psi_grid'] = psis[0]
    
    return fourfit_metric_mats






def readFbarmat(filename,M):
    """Reads bar(F) from text files. Note that bar(F) must be multiplied by Q to get F used in L matrix"""
    
    with open(filename,'r') as foo:
        f = list(foo)
    for i, line in enumerate(f):
        if isinstance(line,str):
            f[i] = line.strip().split()
    f = flatten(f)


    psi_idx = []
    for i, line in enumerate(f):
        if '(' not in line:
            psi_idx.append(i)
    psi_idx = np.array(psi_idx)
    nsteps = len(psi_idx)
    np.testing.assert_almost_equal(np.std(np.diff(psi_idx)),0,err_msg='Unequal matrix sizes at different psi locations!')

    psi = np.full((nsteps,),np.nan,dtype=np.float64)
    Fdata = np.full((nsteps,M,M),0,dtype=np.complex128)

    offset = np.diff(psi_idx)[0]

    for i, idx in enumerate(psi_idx):
        psi[i] = float(f[idx])
        temp = f[idx+1:idx+offset]
        temp = np.array([str2complex(elem) for elem in temp])
        iqty = 0
        for jpert in range(0,M):
            for ipert in range(jpert,M):
                Fdata[i,ipert,jpert] = temp[iqty]
                iqty +=1
        # F is stored as lower cholesky factored Fbar, need to square and also multiply by q^2
        Fdata[i] = np.matmul(Fdata[i],Fdata[i].conj().T)
    assert not np.any(np.isnan(psi))
    assert not np.any(np.isnan(Fdata))

    Fdata = np.array([(F + F.conj().T)/2 for F in Fdata])
    idx = np.argsort(psi)
    psi_f = psi[idx]
    Fbardata = Fdata[idx]
    
    return psi_f, Fbardata


def readKbarmat(filename,M):

    with open(filename,'r') as foo:
        f = list(foo)
    for i, line in enumerate(f):
        if isinstance(line,str):
            f[i] = line.strip().split()
    f = flatten(f)


    psi_idx = []
    for i, line in enumerate(f):
        if '(' not in line:
            psi_idx.append(i)
    psi_idx = np.array(psi_idx)
    nsteps = len(psi_idx)
    np.testing.assert_almost_equal(np.std(np.diff(psi_idx)),0,err_msg='Unequal matrix sizes at different psi locations!')

    psi = np.full((nsteps,),np.nan,dtype=np.float64)
    Kdata = np.full((nsteps,M,M),np.nan,dtype=np.complex128)

    offset = np.diff(psi_idx)[0] 

    for i, idx in enumerate(psi_idx):
        psi[i] = float(f[idx])
        temp = f[idx+1:idx+offset]
        temp = np.array([str2complex(elem) for elem in temp])
        iqty = 0
        for jpert in range(0,M):
            for ipert in range(0,M):
                Kdata[i,ipert,jpert] = temp[iqty]
                iqty +=1
    assert not np.any(np.isnan(psi))
    assert not np.any(np.isnan(Kdata))


    idx = np.argsort(psi)
    psi_k = psi[idx]
    Kbardata = Kdata[idx]
    

    
    return psi_k, Kbardata 



def readGmat(filename,M):

    with open(filename,'r') as foo:
        f = list(foo)
    for i, line in enumerate(f):
        if isinstance(line,str):
            f[i] = line.strip().split()
    f = flatten(f)


    psi_idx = []
    for i, line in enumerate(f):
        if '(' not in line:
            psi_idx.append(i)
    psi_idx = np.array(psi_idx)
    nsteps = len(psi_idx)
    np.testing.assert_almost_equal(np.std(np.diff(psi_idx)),0,err_msg='Unequal matrix sizes at different psi locations!')

    psi = np.full((nsteps,),np.nan,dtype=np.float64)
    Gdata = np.full((nsteps,M,M),np.nan,dtype=np.complex128)

    offset = np.diff(psi_idx)[0] 

    for i, idx in enumerate(psi_idx):
        psi[i] = float(f[idx])
        temp = f[idx+1:idx+offset]
        temp = np.array([str2complex(elem) for elem in temp])
        iqty = 0
        for jpert in range(0,M):
            for ipert in range(jpert,M):
                Gdata[i,ipert,jpert] = temp[iqty]
                Gdata[i,jpert,ipert] = temp[iqty].conj()
                iqty +=1
    assert not np.any(np.isnan(psi))
    assert not np.any(np.isnan(Gdata))

    Gdata = np.array([(G + G.conj().T)/2 for G in Gdata])

    idx = np.argsort(psi)
    psi_g = psi[idx]
    Gdata = Gdata[idx]
    
    return psi_g, Gdata




def readQmat(filename,M):
    """Reads Q = m-nq from text file"""
    
    with open(filename,'r') as foo:
        f = list(foo)
    for i, line in enumerate(f):
        if isinstance(line,str):
            f[i] = line.strip().split()
    f = flatten(f)
    
    psi_idx = np.arange(len(f))[::M+1]
    psi_q = np.full(len(psi_idx),np.nan)
    Qdata = np.full((len(psi_idx),M,M),np.nan)
    
    
    for i, idx in enumerate(psi_idx):
        psi_q[i] = float(f[idx])
        temp = np.array([float(foo) for foo in f[idx+1:idx+M+1]])
        Qdata[i,:,:] = np.diag(temp) 
        
    assert not np.any(np.isnan(psi_q))
    assert not np.any(np.isnan(Qdata))
        
        
    idx = np.argsort(psi_q)
    psi_q = psi_q[idx]
    Qdata = Qdata[idx]
    
    return psi_q, Qdata


def read_coeff_mats(directory,M):
    
    psi_q, Q = readQmat(directory + '/qmat.out',M)
    psi_f, Fbar = readFbarmat(directory + '/fmat.out',M)
    psi_g, G = readGmat(directory + '/gmat.out',M)
    psi_k, Kbar = readKbarmat(directory + '/kmat.out',M)

    
    psis = np.stack([psi_q,psi_f,psi_g,psi_k])
    assert np.allclose(np.std(psis,axis=0),0), "different psi grids!"
    
    F = np.matmul(np.matmul(Q,Fbar),Q)
    K = np.matmul(Q,Kbar)
    
    coeff_mats = {'psi_grid':psis[0],
                 'F':F,
                 'Fbar':Fbar,
                 'G':G,
                 'K':K,
                 'Kbar':Kbar,
                 'Q':Q}
    
    return coeff_mats



def read_sq_mat(filename):
    
    with open(filename,'r') as foo:
        f = list(foo)
    for i, line in enumerate(f):
        if isinstance(line,str):
            f[i] = line.strip().split()
    f = flatten(f)

    psi_sq = np.array([float(foo) for foo in f[::5]])
    F = np.array([float(foo) for foo in f[1::5]]) 
    P = np.array([float(foo) for foo in f[2::5]]) 
    jac = np.array([float(foo) for foo in f[3::5]]) 
    q = np.array([float(foo) for foo in f[4::5]]) 
    
    sq = {'psi_grid':psi_sq,
         'F':F,
         'P':P,
         'jac':jac,
         'q':q}
    
    return sq


def read_rzphi_mat(filename):
    with open(filename,'r') as foo:
        f = list(foo)
    for i, line in enumerate(f):
        if isinstance(line,str):
            f[i] = line.strip().split()
    f = flatten(f)

    mpsi = None
    mtheta = None

    for i, line in enumerate(f):
        if float(line) == 0 and mpsi is None:
            mpsi = i
        if float(line) == 1 and mtheta is None:
            mtheta = i-mpsi + 1

    psi_grid = np.array([float(foo) for foo in f[:mpsi]])
    theta_grid = np.array([float(foo) for foo in f[mpsi:mpsi+mtheta]])
    griddata = f[mpsi+mtheta:]

    r2 = np.full((mpsi,mtheta),np.nan)
    deta = np.full((mpsi,mtheta),np.nan)
    dphi = np.full((mpsi,mtheta),np.nan)
    jac = np.full((mpsi,mtheta),np.nan)

    k = 0
    for i in range(mpsi):
        for j in range(mtheta):
            r2[i,j] = float(griddata[k])
            deta[i,j] = float(griddata[k+1])
            dphi[i,j] = float(griddata[k+2])
            jac[i,j] = float(griddata[k+3])
            k += 4

    assert not np.any(np.isnan(r2))
    assert not np.any(np.isnan(deta))
    assert not np.any(np.isnan(dphi))
    assert not np.any(np.isnan(jac))
    
    rzphi_arrs = {'psi_grid':psi_grid,
                  'theta_grid':theta_grid,
                  'r_squared':r2,
                  'delta_eta':deta,
                  'delta_phi':dphi,
                  'jac':jac}
    
    return rzphi_arrs


def run_stride(gfile, return_WpWv=True, return_metric=True, return_direct=True, return_coeffs=True,
               return_interval_details=True, return_timing=True, ninters=50, nthreads=1, verbose=True):

    assert os.path.exists('./stride'), "stride program not found in cwd"
    
    dst = 'g'
    if os.path.exists(dst):
        os.remove(dst)
    if isinstance(gfile,str):
        gfile_path = gfile
        os.symlink(gfile_path, dst)
    else:
        write_gfile('g',**gfile)
        gfile_path = 'g' + str(gfile['shot']) + '.' + str(gfile['time'])
        
    try:
        raw_output = subprocess.run(['./stride',str(ninters),str(nthreads)],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    universal_newlines=True)
    except:
        raise
    if raw_output.returncode != 0:
        print('failed for gfile: ',gfile_path)
        raise Exception(raw_output.stderr, raw_output.stdout)
    elif verbose:
        print('run suceeded for gfile: ',gfile_path)
    time.sleep(2)
    # get locations of singular surfaces, mode numbers, etc
    sing, sing_loc, m_grid, mpsi, mtheta, nn, psio, times, interval_details = parse_raw_output(raw_output.stdout, 
                                                                                               return_timing, 
                                                                                               return_interval_details)
    
    M = len(m_grid)
    N = 2*M
    
    data = {'sing': sing,
            'sing_loc': sing_loc,
            'm_grid': m_grid,
            'mpsi':mpsi,
            'mtheta':mtheta,
            'nn':nn,
            'psio': psio,
            'M': M,
            'N': N}
    
    if return_interval_details:
        data.update({'interval_details':interval_details})

    if return_timing:
        data.update({'times':times})
    
    if return_WpWv:
        if os.path.exists('wp.out'):
            Wp_dcon = readWmat('wp.out',M)
            os.remove('wp.out')
        else:
            Wp_dcon = None
        if os.path.exists('wv.out'):
            Wv_dcon = readWmat('wv.out',M)
            os.remove('wv.out')
        else:
            Wv_dcon = None
        if os.path.exists('delta_prime.out'):
            delta_prime = readWmat('delta_prime.out',2*len(sing))
            os.remove('delta_prime.out')
        else:
            delta_prime = None
        data.update({'Wp_dcon': Wp_dcon,
                     'Wv_dcon': Wv_dcon,
                     'delta_prime':delta_prime})

    if return_coeffs:
        if all([os.path.exists(foo + 'mat.out') for foo in 'abcdeh']):
            primitive_mats = read_primitive_matrices('./',M)
            _ = [os.remove(foo + 'mat.out') for foo in 'abcdeh']
        else:
            primitive_mats = None
        if all([os.path.exists(foo + 'mat.out') for foo in 'fgkq']):
            coeff_mats = read_coeff_mats('./',M) 
            _ = [os.remove(foo + 'mat.out') for foo in 'fgkq']
        else:
            coeff_mats = None
        data.update({'coefficient_matrices':coeff_mats,
                     'primitive_matrices':primitive_mats})
        
    if return_metric:
        if all([os.path.exists('four' + foo + '.out') for foo in [
                'G11','G22','G33','G23','G31','G12','jac','jac1']]):
            fourfit_metric = read_fourfit_metric('./',M)
            _ = [os.remove('four' + foo + '.out') for foo in [
                'G11','G22','G33','G23','G31','G12','jac','jac1']]
        else:
            fourfit_metric = None
        if os.path.exists('metric.out'):
            metric = read_metric('metric.out',mpsi+1,mtheta+1) 
            os.remove("metric.out")
        else:
            metric = None
        data.update({'fourfit_metric':fourfit_metric,
                     'metric':metric})
    
    if return_direct:
        if os.path.exists('sqmat.out'):
            profiles = read_sq_mat('sqmat.out')
            os.remove('sqmat.out')
        else:
            profiles = None
        if os.path.exists('rzphimat.out'):
            rzphi = read_rzphi_mat('rzphimat.out') 
            os.remove('rzphimat.out')
        else:
            rzphi = None
        data.update({'profiles':profiles,
                     'straight_field_line_coords':rzphi})

            
    return data
