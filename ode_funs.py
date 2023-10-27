import numpy as np
import scipy.integrate
import scipy.linalg


def ode_itime(coeffs, pt1, sing_loc, pt2=None):
    """Computes estimated cost to integrate from pt1 to pt2 using adaptive method (ie, number of steps to take over the interval)"""
    
    a_axis = coeffs[0]
    b_axis = coeffs[1]
    a_sing = coeffs[2]
    b_sing = coeffs[3]
    a_edge = coeffs[4]
    b_edge = coeffs[5]
    
    itime = np.zeros_like(pt1)
            
    if pt2 is not None:
    # estimate integration time of the whole interval.
        itime += (a_axis/b_axis) * np.abs(np.log(1+b_axis*np.abs(pt2-sing_loc[0])) - np.log(1+b_axis*np.abs(pt1-sing_loc[0])))
        itime += (a_edge/b_edge) * np.abs(np.log(1+b_edge*np.abs(pt2-sing_loc[-1])) - np.log(1+b_edge*np.abs(pt1-sing_loc[-1])))
        for s in sing_loc[1:-1]:
            itime += (a_sing/b_sing) * np.abs(np.log(1+b_sing*np.abs(pt2-s)) - np.log(1+b_sing*np.abs(pt1-s)))      
    else:
    # estimate instantaneous time of current location
        itime +=  a_axis/(1+b_axis*np.abs(pt1-sing_loc[0]))
        itime +=  a_edge/(1+b_edge*np.abs(pt1-sing_loc[-1]))
        for s in sing_loc[1:-1]:
            itime +=  a_sing/(1+b_sing*np.abs(pt1-s))
    return itime


def set_intervals(sing,itime_coeffs,start,end,nInters,method='naive'):
    """divides domains into subintervals for integration in parallel"""
    
    # make sure there are enough intervals so that each will touch at most 1 singularity
    assert nInters >= len(sing)*3+2
    # each row defines one interval: [start, stop, sing_left, sing_right, direction, q]
    inters = np.full((len(sing)*2+2,6),np.nan)  
    sing_loc = np.array([0,*np.mean(sing[:,:2],axis=1),1])

    #  first set minimal intervals defined by axis and singular surfaces.
    temp = np.concatenate(([start],sing[:,:2].flatten(),[end])).reshape((len(sing)+1,2))
    for i, elem in enumerate(temp):
        inters[2*i,0] = elem[0]
        inters[2*i,1] = elem[0] + (elem[1] - elem[0])/2
        inters[2*i,2:] = [1,0,1,0]
        inters[2*i+1,0] = elem[0] + (elem[1] - elem[0])/2
        inters[2*i+1,1] = elem[1]
        inters[2*i+1,2:] = [0,1,-1,0]
        
    # evenly subdivide intervals further
    while len(inters) < nInters-len(sing):
        if method=='naive':
            # find the largest interval
            interval_lengths = inters[:,1] - inters[:,0]
            max_ind = np.argmax(interval_lengths)
            split_pt = inters[max_ind,0]+interval_lengths[max_ind]/2
            # divide that interval, keeping track of where singularities are
        elif method=='sing':
            # find the interval that will take the most steps
            interval_times= ode_itime(itime_coeffs,inters[:,0],sing_loc,inters[:,1])
            max_ind = np.argmax(interval_times)
            # find where to split to make each half take the same time
            s1 = ode_itime(itime_coeffs,inters[max_ind,0],sing_loc)
            s2 = ode_itime(itime_coeffs,inters[max_ind,1],sing_loc)
            alpha = (2*s1 - np.sqrt(2*s1**2 + 2*s2**2))/(2*(s1 - s2))
            split_pt = alpha*inters[max_ind,1] + (1-alpha)*inters[max_ind,0]
        # divide that interval, keeping track of where singularities are
        if inters[max_ind][2] == 1: # singularity to the left of interval, split to the right
            temp = np.array([[split_pt,inters[max_ind,1],0,0,inters[max_ind,-2],0]])
            inters[max_ind,1] = split_pt
            inters = np.concatenate((inters[:max_ind+1],temp,inters[max_ind+1:]),axis=0)
        elif inters[max_ind][3] == 1: # singularity to the right, split to the left
            temp = np.array([[inters[max_ind,0],split_pt,0,0,inters[max_ind,-2],0]])
            inters[max_ind,0] = split_pt
            inters = np.concatenate((inters[:max_ind],temp,inters[max_ind:]),axis=0)
        else: # no singularity on either side, doesnt matter which way you split
            temp = np.array([[split_pt,inters[max_ind,1],0,0,inters[max_ind,-2],0]])
            inters[max_ind,1] = split_pt
            inters = np.concatenate((inters[:max_ind+1],temp,inters[max_ind+1:]),axis=0)
    
     # insert singular intervals   
    for s in sing:
        idx = np.searchsorted(inters[:,0],s[0])
        inters = np.insert(inters,idx,np.array([s[0],s[1],1,1,0,s[2]]),axis=0)
    
    
    return inters


def solve(inters,L,mpert,method):
    """integrates each subinterval seperately"""
    
    nInters = len(inters)
    M = len(mpert)
    N = 2*M
    def fun(t,y):
        return np.matmul(L(t),y.reshape((N,N))).flatten()
    
    def fun_lsoda(t,y):
        y_re = y[:N**2].reshape((N,N))
        y_im = y[N**2:].reshape((N,N))
        y1 = np.matmul(L(t),y_re + 1j*y_im)
        return np.concatenate([y1.real.flatten(),y1.imag.flatten()])
    
    
    all_soln = np.full((nInters,N,N),np.nan,dtype=np.complex128)
    status = np.full((nInters),0)
    nfev = np.full((nInters),0)
    njev = np.full((nInters),0)
    nlu = np.full((nInters),0)
    nmm = np.full((nInters),0)
    nsteps = np.full((nInters),0)

    for i, interval in enumerate(inters):


        if interval[-1] != 0: # singualr surface, just take 1 step and zero resonant mode
            # find which mode is resonant
            resm = np.where(mpert==interval[-1])[0][0]
            y0 = np.eye(N,dtype=np.complex128)
            # zero resonant mode
            y0[:,resm] = 0
            y0[resm,:] = 0
            y0[:,resm + M] = 0
            y0[resm + M,:] = 0
            # step forward
            y0 += np.matmul(L(interval[0]),y0)*(interval[1]-interval[0])
            # reset resoannt modes to identity
            y0[:,resm] = 0
            y0[resm,:] = 0
            y0[:,resm + M] = 0
            y0[resm + M,:] = 0
            y0[resm,resm] = 1
            y0[resm + M, resm + M] = 1
            # store soln
            all_soln[i,:] = y0
            
            status[i] = 0
            nfev[i] = 1
            njev[i] = 0
            nlu[i] = 0
            nmm[i] = 1
            nsteps[i] = 1 
            
            
        else:        
            if interval[-2] == 1: # integrate forward
                t0 = interval[0]
                tf = interval[1]
            elif interval[-2] == -1: # integrate backward
                t0 = interval[1]
                tf = interval[0]



            if method in ['RK45','RK23','Radau','BDF']:
                y0 = np.eye(N, dtype=np.complex128).flatten()
                out = scipy.integrate.solve_ivp(fun,(t0,tf),y0,method=method)

                all_soln[i,:] = out.y[:,-1].reshape(N,N)
                status[i] = out.status
                nfev[i] = out.nfev
                njev[i] = out.njev
                nlu[i] = out.nlu
                nmm[i] = out.nfev
                nsteps[i] = out.t.size - 1
                
            elif method == 'LSODA':
                y0 = np.concatenate([np.eye(N).flatten(),np.zeros((N,N)).flatten()])
                out = scipy.integrate.solve_ivp(fun_lsoda,(t0,tf),y0,method=method)

                all_soln[i,:] = out.y[:N**2,-1].reshape(N,N) + 1j*out.y[N**2:,-1].reshape(N,N)
                status[i] = out.status
                nfev[i] = out.nfev
                njev[i] = out.njev
                nlu[i] = out.nlu
                nmm[i] = out.nfev
                nsteps[i] = out.t.size - 1

            elif method == 'expm':
                teval = np.mean([t0,tf])
                all_soln[i] = scipy.linalg.expm(L(teval)*(tf-t0))
                
                theta13 = 5.371920351148152
                Lnorm = np.linalg.norm(L(teval)-np.eye(N)*np.trace(L(teval))/N,1)
                             
                status[i] = 0
                nfev[i] = 1
                njev[i] = 0
                nlu[i] = 1
                nmm[i] = 6 + int(np.log2(Lnorm/theta13))
                nsteps[i] = 1 
                
            elif method == 'eig_expm':
                # eigendecomp in LAPACK for general nonsymmetric matrix: 26.33*N^3 flops
                # https://www.netlib.org/lapack/lug/node71.html
                
                teval = np.mean([t0,tf])
                Lt = L(teval)
                dt = tf-t0
                
                D,V = np.linalg.eig(Lt)
                Vinv = np.linalg.inv(V)
                D = np.diag(np.exp(D*dt))
                
                all_soln[i] = np.matmul(np.matmul(V,D),Vinv)
                             
                status[i] = 0
                nfev[i] = 1
                njev[i] = 0
                nlu[i] = 1
                nmm[i] = 26.33/2 + 1
                nsteps[i] = 1 
                
            elif method == 'magnus':
                # approximate L on interval as
                # L = L0 + L1*t
                # Omega1 = A0 (tf-t0) + 1/2 A1 (tf^2-t0^2)
                # Omega2 = A0 A1 (1/12 t0^3 - 1/4 t0^2 tf + 1/4 t0 tf^2 - 1/12 tf^3)
                #         + A1 A0 (-1/12 t0^3 + 1/4 t0^2 tf - 1/4 t0 tf^2 + 1/12 t_f^3)
                
                teval = (tf + t0)/2
                dt = tf-t0
                
                Lt0 = L(t0)
                Ltf = L(tf)
                
                L0 = L(teval)
                L1 = L.derivative(1)(teval)

                Omega1 = L0*(tf-t0) + 1/2*L1*(tf**2 - t0**2)
                Omega2 = np.matmul(L0,L1)*(1/12*t0**3 - 1/4*t0**2*tf + 1/4*t0*tf**2 - 1/12*tf**3) \
                         + np.matmul(L1,L0)*(-1/12*t0**3 + 1/4*t0**2*tf - 1/4*t0*tf**2 + 1/12*tf**3)
                Omega = Omega1 + Omega2
                
                all_soln[i] = scipy.linalg.expm(Omega)
                                            
                status[i] = 0
                nfev[i] = 2
                njev[i] = 0
                nlu[i] = 1
                nmm[i] = 26.33/2 + 1 + 2
                nsteps[i] = 1 
                
            elif method == 'rk4':
                y0 = np.eye(N).flatten()
                h = tf-t0
                k1 = h*fun(t0, y0)
                k2 = h*fun(t0 + h/2, y0 + k1/2)
                k3 = h*fun(t0 + h/2, y0 + k2/2)
                k4 = h*fun(t0 + h, y0 + k3)
                
                all_soln[i,:] = (y0 + 1/6*(k1 + 2*k2 + 2*k3 + k4)).reshape((N,N))
                
                status[i] = 0
                nfev[i] = 4
                njev[i] = 0
                nlu[i] = 0
                nmm[i] = 4
                nsteps[i] = 1   
                
            elif method == 'rk2':
                y0 = np.eye(N).flatten()
                h = tf-t0
                
                
                all_soln[i,:] = (y0 + h*fun(t0+h/2,y0+h/2*fun(t0, y0))).reshape((N,N))
                
                status[i] = 0
                nfev[i] = 2
                njev[i] = 0
                nlu[i] = 0
                nmm[i] = 2
                nsteps[i] = 1
            
            elif method == 'trapz':
                # y1 = y0 + h/2*(L(t0)y0  + L(t1)y1)
                # y1 = y0 + h/2*L(t0)*y0 + h/2*L(t1)*y1
                # (I-h/2*L(t1))y1 = (I+h/2*L(t0)y0
                
                h = (tf-t0)
                Lt0 = L(t0)
                Ltf = L(tf)
                
                all_soln[i,:] = np.linalg.solve(np.eye(N) - h/2*Ltf, np.eye(N) + h/2*Lt0)

                status[i] = 0
                nfev[i] = 2
                njev[i] = 0
                nlu[i] = 1
                nmm[i] = 0
                nsteps[i] = 1  
                
            elif method == 'implicit_midpoint':
                # y1 = y0 + h*L(t/2)(y0 + y1)/2
                # y1 = y0 + h*L*y0/2 + h*L*y1/2
                # (I-h/2*L)y1 = (I+h/2*L)y0
                
                teval = (t0 + tf)/2
                h = (tf-t0)
                Lt = L(teval)
               
                all_soln[i,:] = np.linalg.solve(np.eye(N) - h/2*Lt, np.eye(N) + h/2*Lt)

                status[i] = 0
                nfev[i] = 1
                njev[i] = 0
                nlu[i] = 1
                nmm[i] = 0
                nsteps[i] = 1  

            elif method == 'forward_euler':      
                teval = t0
                h = (tf-t0)
                Lt = L(teval)
               
                all_soln[i,:] = np.eye(N) + h*Lt

                status[i] = 0
                nfev[i] = 1
                njev[i] = 0
                nlu[i] = 0
                nmm[i] = 0
                nsteps[i] = 1  
                
            elif method == 'backward_euler':
                teval = tf
                h = (tf-t0)
                Lt = L(teval)
               
                all_soln[i,:] = np.linalg.inv(np.eye(N) - h*Lt)

                status[i] = 0
                nfev[i] = 1
                njev[i] = 0
                nlu[i] = 1
                nmm[i] = 0
                nsteps[i] = 1  
            else:
                raise Exception("method not implemented")
                
                
    return all_soln, status, nfev, njev, nlu, nmm, nsteps


def fixup(uT):
    """maintains linear independence of solns by triangularizing"""
    N = uT.shape[0]
    M = int(N/2)
    # calculate and sort the Euclidean norms of the propagator's columns
    # u = [q; p] (at the axis)
    # Look at the q(axis) = 0 modes only = RHS of uT
    uT2 = uT[:,M:]
    # Get the norm of the "q" halves of the propagator's columns
    unorm = np.linalg.norm(uT2[:M,:],2, axis=0)
    # Sort the columns by their norms => store in "index"
    index = np.argsort(unorm)[::-1]
    # triangularize primary solutions--(take linear combos of columns)
    mask = np.full((2,M),True)
    for isol in range(M):
        ksol=index[isol] # ksol = largest remaining mode (column)
        mask[1,ksol] = False
        kpert = np.arange(M)[mask[0,:]][np.argmax(np.abs(uT2[:M,ksol])[mask[0,:]])]  # kpert = largest unmasked row element in ksol  
        mask[0,kpert]= False
        for jsol in range(M):
            if mask[1,jsol]:
                if (uT2[kpert,ksol] == 0):
                    if (uT2[kpert,jsol] !=0):
                        #There is an all zero column.
                        raise Exception("Unable to Gauss-reduce!")
                else:
                    # The actual linear combination step.
                    # Remove from all other (as yet unmaksed) modes the projection of the largest mode on them.
                    uT2[:,jsol] = uT2[:,jsol] - uT2[:,ksol] * uT2[kpert,jsol] / uT2[kpert,ksol]
                    uT2[kpert,jsol] = 0 # Just 0's an element that is already meant to be exactly 0.

    uT[:, :M] = 0.0
    uT[:, M:] = uT2
    return uT


def propagate(all_soln,inters,psio):
    """combines individual solutions to propagate IC across whole domain, calculates plasma response matrix"""
    
    N = all_soln.shape[-1]
    M = int(N/2)
    
    # IC = [0,0 ; 0,I]
    uAxis = np.zeros_like(all_soln[0])
    for i in range(M):
        uAxis[M+i,M+i] = 1
        
    nInters = len(inters)
    sing_intervals = np.where(inters[:,-1] != 0)[0]
    directions = inters[:,-2]
    # create subpropagators: wherever the direction of integration changes, start a new one 
    # first shoot out from axis to halfway between axis and first singular surface
    for i, mat in enumerate(all_soln):
        uAxis = np.matmul(mat,uAxis)
        uAxis = fixup(uAxis)
        if directions[i] != directions[i+1] or i+1==len(directions):
            break
    
    
    Lshoot = [np.eye(N) for foo in sing_intervals]
    Rshoot = [np.eye(N) for foo in sing_intervals]
    # shoot solutions out from each singular interval
    for i, sing in enumerate(sing_intervals):
        # shoot left:
        for k in range(sing-1,0,-1):
            Lshoot[i] = np.matmul(all_soln[k],Lshoot[i])
            if directions[k] != directions[k-1] or k-1==0:
                break
        # shoot right:
        for k in range(sing+1,nInters):
            Rshoot[i] = np.matmul(all_soln[k],Rshoot[i])
            if directions[k] != directions[k+1] or k+1==nInters:
                break
    # shoot backwards from the edge
    uEdge = np.eye(N)
    for i in range(nInters-1,0,-1):
        uEdge = np.matmul(all_soln[i],uEdge)
        if directions[i] != directions[i-1] or i-1==0:
            break

    # connect solutions across singular surfaces
    soln = uAxis
    for i, sing in enumerate(sing_intervals):
        
        soln = np.linalg.solve(Lshoot[i],soln)
        soln = fixup(soln)
        
        soln = np.matmul(all_soln[sing],soln)
        soln = fixup(soln)

        soln = np.matmul(Rshoot[i],soln)
        soln = fixup(soln)

    # connect final subinterval to the edge
    soln = np.linalg.solve(uEdge,soln)
    soln = fixup(soln)
    
    # solve for plasma response matrix from full solution
    phi_pp = soln[M:,M:]
    phi_qp = soln[:M,M:]
    Wp = np.matmul(phi_pp,np.linalg.inv(phi_qp))
    
    # make sure its hermitian and scale it
    Wp = (Wp + Wp.conj().T)/(2*psio**2)

    return Wp, soln


def wrapper(nInters, ode_method, interval_method, start, end, sing, mpert, L, psio,itime_coeffs):
    """wrapper function that computes intervals, integrates ODE, comb """
    inters = set_intervals(sing,itime_coeffs,start,end,nInters,method=interval_method)
    all_soln, status, nfev, njev, nlu, nmm, nsteps = solve(inters,L,mpert,ode_method)
    Wp, soln = propagate(all_soln,inters,psio)
    stats = {'nfev':nfev,
            'njev':njev,
            'nlu':nlu,
            'nmm':nmm+1,
            'nsteps':nsteps,
            'flops':38/3*nfev + 8/3*nlu + 2*(nmm+1)} # 1 extra matmul per interval for propagation
    
    return Wp, inters, stats
    
    