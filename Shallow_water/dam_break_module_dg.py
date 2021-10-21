
from numpy import *
from time import perf_counter
from scipy.sparse import csr_matrix
#from matplotlib.pyplot import*
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import special
import copy

def Legendre_deriv(Q, x):
    
    '''
    This function compute the Legendre polynomial and its derivative
    
    Inputs:
    -------
            Q  : Integration order(N+1: for exact, N: for inexact integration)
            x  : value of x at which the polynomial is evaluated
            
    Outputs:
    -------
           L1  : Value of the polynomial at x
           dL1 : First derivative
           ddLi: Second derivative
    '''
    
    L0 = 1; dL0 = 0; ddL0 = 0
    L1 = x; dL1 = 1; ddL1 = 0
    if(Q == 0):
        return L0,dL0,ddL0
    elif(Q == 1):
        return L1,dL1,ddL1
    else:
        for i in range(2, Q+1):

            Li = ((2*i-1)/i)*x*L1 - ((i-1)/i)*L0  # iteration of the polynomials
            dLi = i*L1 + x*dL1
            ddLi = (i+1.0)*dL1 + x*ddL1

            L0,L1 = L1,Li

            dL0,dL1 = dL1,dLi

            ddL0,ddL1 = ddL1,ddLi

        return L1, dL1, ddL1

def Lobatto_deriv(Q, x):
    
    '''
    This function compute the Lobatto polynomial and its derivative
    
    Inputs:
    -------
            Q  : Integration order(N+1: for exact, N: for inexact integration)
            x  : value of x at which the polynomial is evaluated
            
    Outputs:
    -------
           B  : Value of the polynomial at x
           dB : First derivative
    '''
    
    L,dL, ddL = Legendre_deriv(Q-1, x)
    B = (1.0-x**2)*dL                      # lobatto polynomial
    dB = -2.0*x*dL + (1.0-x**2)*ddL        # derivative of lobatto polynomial   
    
    return B, dB

def Lobatto_p(Q):
    
    '''
    This function compute the Lobatto points
    
    Input:
    -------
            Q  : Integration order(N+1: for exact, N: for inexact integration)
            
    Output:
    -------
           X: array containing the Lobatto points
    '''
    
    X = []                                      # Array that contains legendre points
    K = 100                                     # Order of approximation of Newton method
    e = 1e-20                                   # tolerance
    for i in range(Q+1):
        xik = cos(((2*i+1)/(2*(Q+1)-1))*pi)         # Chebchev points

        for k in range(K):
            out1, out2 = Lobatto_deriv(Q+1, xik)
            xikk = xik - out1/out2              # approximation of the solution using Newton

            if abs(xikk-xik) < e:

                break

            xik = xikk

        X.append(xikk)
        
    return array(X[::-1])

# Lagrange basis for single value x
def LagrangeBasis(N, i, xl, Xr):
    
    '''
    This function compute the Lagrange polynomial(basis function) and its derivative
    
    Inputs:
    -------
            N  : polynomial order
            i  : ith polynomial 
            xl : values at which the polynomial is evaluated
            Xr : Lobatto points or the roots of the generating polynomial used to construct the basis function
            
    Outputs:
    -------
           L   : Value of the polynomial
           dL  : Derivative
    '''
    L = 1; dL = 0
        
    for j in range(N+1):
            
        prod = 1
        
        if (j != i):
            L = L*(xl-Xr[j])/(Xr[i]-Xr[j])
                
            for k in range(N+1):
                if (k!=i  and k!=j):
                    
                    prod = prod*(xl-Xr[k])/(Xr[i]-Xr[k])
        
            dL = dL+prod/(Xr[i]-Xr[j])
            
    return L, dL

# Lagrange basis for an array that contains value of x
def LagrangeBasis_deriv(N,Q,Xn, Xq):

    l_basis = zeros((N+1,Q+1))
    dl_basis = zeros((N+1,Q+1))
    
    for k in range(Q+1):
        xl = Xq[k]
        
        for i in range(N+1):
            # Call of LagrangeBasis function
            l_basis[i,k], dl_basis[i,k] = LagrangeBasis(N, i, xl, Xn)
            
    return l_basis, dl_basis

# intma function
def intma_cdg(N, Ne, method_type):
    
    '''
    This function compute the intma array for the CG or DG
    
    Inputs:
    -------
            N          : polynomial order
            Ne         : number of element
            method_type: CG or DG
            
    Output:
    -------
           intma: (matrix) that contains intma values
    '''
    
    intma = zeros((N+1,Ne))
    
    # intma for CG
    if (method_type == 'cg'):
        for e in range(1,Ne+1):
        
            t = (e-1)*N
            r = N*e
            intmm = []
            for s in range(t, r+1):
                intmm.append(s)
            intma[:,e-1] = array(intmm)
    
    # intma for DG
    if (method_type == 'dg'):
        for e in range(1,Ne+1):
        
            t = int((e-1)*N)
            r = int(e*N)

            intmm = []
            for s in range(t, r+1):
                it = e-1+s
                intmm.append(it)
            intma[:,e-1] = array(intmm)
        
    return intma


#funtion that compute weight values based on quadrature rule
def weight(Q):
    
    '''
    This function compute the weight for the integration
    
    Inputs:
    -------
            Q : Integration order(N+1: for exact, N: for inexact integration)
            
    Output:
    -------
           w : (array) that contains the weight values
    '''
    
    xi = Lobatto_p(Q)
    w = zeros(Q+1)
    for i in range(Q+1):
        
        out1, out2, out3 = Legendre_deriv(Q, xi[i])
        w[i] = 2/(Q*(Q+1)*(out1)**2)
        
    return w 

# grid points
def grid_dg(N,Ne, xe, ax, bx):
    
    '''
    This function compute the weight for the integration
    
    Inputs:
    -------
            Q     : Integration order(N+1: for exact, N: for inexact integration)
            Ne    : Number of elements in the domain
            xe    : grid points within the element
            ax, bx: boundaries
            
    Output:
    -------
           grid: (matrix) that contains all the grid points
    '''
    
    grid = zeros((N+1,Ne))

    xel = linspace(ax,bx,Ne+1)

    for e in range(1,Ne+1):
        
        ae = xel[e-1] ; be = xel[e]

        xsi = ((be-ae)/2)*(xe-1) + be
        
        for i in range(N+1):

            grid[i,e-1] = xsi[i]
            
    return grid

# Element mass matrix

def Element_matrix(N,Q, wght,l_basis):
    
    '''
    This function compute the element mass matrix
    
    Inputs:
    -------
            Q      : Integration order(N+1: for exact, N: for inexact integration)
            N      : Polynomial order
            wght   : weights
            l_basis: basis function values
            
    Output:
    -------
           Me: Element mass matrix
    '''
    
    Me = zeros((N+1, N+1))       # initialisation of the matrix
    
    for k in range(Q+1):
        wk = wght[k]
        
        for i in range(N+1):
            xi = l_basis[i,k]
            
            for j in range(N+1):
                xj = l_basis[j,k]
                
                Me[i,j] = Me[i,j] + wk*xi*xj

    Me = (1/2)*Me

    return Me

# Element mass matrix for Dg
def Element_matrix_inv_DG(coord,Ne,N,Q, wght,l_basis):
    
    '''
    This function compute the element mass matrix
    
    Inputs:
    -------
            Q      : Integration order(N+1: for exact, N: for inexact integration)
            N      : Polynomial order
            wght   : weights
            l_basis: basis function values
            
    Output:
    -------
           Me: Element mass matrix
    '''
    
    Me = zeros((Ne,N+1, N+1))       # initialisation of the matrix
    Me_inv = zeros((Ne,N+1, N+1)) 
    for e in range(1,Ne+1):
        x = coord[:,e-1]
        dx=x[-1]-x[0]
    
        for k in range(Q+1):
            wk = wght[k]

            for i in range(N+1):
                xi = l_basis[i,k]

                for j in range(N+1):
                    xj = l_basis[j,k]

                    Me[e-1,i,j] = Me[e-1,i,j] + wk*xi*xj

        Me[e-1,:,:] = (dx/2)*Me[e-1,:,:]
        
        Me_inv[e-1,:,:] = linalg.inv(Me[e-1,:,:])

    return Me, Me_inv

# DSS operator

def DSS_operator(A, Ne, N, Np, intma, coord, iperiodic, mtype = None):
    
    '''
    This function is the Direct Stiffness Summation operator
    
    Inputs:
    -------
            A       : Matrix ( Me or De, ...)
            N       : Polynomial order
            Ne      : Number of elements
            Np      : Number of global grid points
            intma   : intma array
            periodic: Array that helps to deal with the boundaries and periodicity
            coord   : all the grid points
            mtype   : method used (CG or DG)
            
    Output:
    -------
           M: Global matrix
    '''
    
    M = zeros((Np, Np))
    
    for e in range(1,Ne+1):
        x = coord[:,e-1]
        dx=x[-1]-x[0]
        
        for j in range(N+1):
            
            J = int(intma[j,e-1])
            
            for i in range(N+1):
                
                I = int(intma[i, e-1])
                
                # diff = differentiation matrix
                if (mtype == 'diff'):
                    M[I,J] = M[I,J] + A[i,j]
                
                elif(mtype == "Lmatrix"):
                    M[I,J] = M[I,J] + (1/dx)*A[i,j]
                    
                else:
                    
                    M[I,J] = M[I,J] + dx*A[i,j]
                    
    return M
      
# Exact solution
def exact_solution(coord,Ne,N,time, case,ax,bx,h_init,hu_init):

    #Set some constants
    #xc = 0
    amp = 0.1
    strength = 8
    #Initialize
    qe = zeros((Ne,2, N+1))
    
    #Generate Grid Points
    for e in range(1,Ne+1):
        for i in range(N+1):
            x = coord[i,e-1]
            #xbar = x - xc
            
            if(case == 1):  #Gaussian wave with flat bathymetry
                gravity = 10
                hmean = 0.01
                
                hh = amp*exp(-strength*x**2)
                qe[e-1,0,i] = hh
                qe[e-1,1,i] = 0
              

            elif(case == 2): #Gaussian Wave with linear bathymetry
                gravity = 10
                hmean = 0.2
                hh = amp*exp(-strength*x**2)
                hb = hmean - 0.1*(1 - (x-ax)/(bx-ax))
                qe[e-1,0,i] = hh+0*hmean
                qe[e-1,1,i] = 0
            
                
            elif(case == 3):  # Dam break
                gravity = 1
                qe[e-1,0,i] = h_init(x)
                qe[e-1,1,i] = hu_init(x)
            
            elif(case == 4):
                gravity = 1
                qe[e-1,0,i] = (1/2)*cos(2*pi*x)*cos(2*pi*time)
                qe[e-1,1,i] = ((1/2)*cos(2*pi*x)*cos(2*pi*time)+1)*(1/2)*sin(2*pi*x)*sin(2*pi*time)
                
            elif(case == 5):  
                
                gravity = 1
                qe[e-1,0,i] = 1 + (2/5)*exp(-5*x**2)
                qe[e-1,1,i] = 0
                
                
    return qe, gravity

# Runge Kutta method
def RKstages(ik, kstages):
    
    # Stages for Runge Kutta method
    if(kstages == 1):
        if(ik == 1):
            a0 = 1; a1 = 0; beta = 1
    elif (kstages == 2):
        if(ik == 1):
            a0 = 1; a1 = 0; beta = 1
        elif(ik == 2):
            a0 = 0.5; a1 = 0.5; beta = 0.5
    elif(kstages == 3):
        if(ik == 1):
            a0=1; a1=0; beta=1

        if(ik == 2):
            a0=3/4; a1=1/4; beta=1/4 
        if(ik == 3):
            a0 = 1/3; a1 = 2/3; beta = 2/3

    elif(kstages == 4):
        if(ik == 1):
            a0=1; a1=0; beta=1/2

        if(ik == 2):
            a0=0; a1=1; beta=1/2

        if(ik == 3):
            a0 = 2/3; a1 = 1/3; beta = 1/6

        if (ik == 4):
            a0 = 0; a1 = 1; beta = 1/2

    return a0, a1, beta

def compute_ti_aux(stages):

    M = stages + 1
    MM = M+1
    alpha = zeros((MM, MM))

    alpha[2,0] = 0
    alpha[2,1] = 1

    for m in range(3,MM):
        
        alpha[m,m-1] = (2/m)*alpha[m-1,m-2]
        
        for k in range(1, m-2):
            
            alpha[m,k] = (2/k)*alpha[m-1,k-1]
        
        sum_temp = 0
        for i in range(1,m):
            sum_temp = sum_temp + alpha[m,i]
    
        alpha[m,0] = 1 - sum_temp
        
    return alpha

def massInit(wnq, Ne, N, Q, l_basis, qe, qb, coord):
    
    mass0 = 0
    for e in range(1, Ne+1):
        
        x = coord[:,e-1]
        dx=x[-1]-x[0]
        for k in range(Q+1):
            
            wq = wnq[k]*(dx/2)
            
            for j in range(N+1):
                
                h_k = l_basis[j,k]
                mass0 = mass0 + wq*(qe[e-1,0,j] + qb[j,e-1])*l_basis[j,k]
                
                
    return mass0


def create_rhs(qp,coord,Ne,N,Q,wnq,l_basis,dl_basis,h_eps,gravity):

    #Initialize
    rhs = zeros((Ne,2, N+1))

    #Integrate Divergence of Flux (Weak Form)
    for e in range(1,Ne+1):

       #Store Coordinates
        
        h_e = qp[e-1,0,:]
        hu_e = qp[e-1,1,:]

        #Jacobians
        x = coord[:,e-1]
        dx = x[-1]-x[0]
        jac = dx/2
        ksi_x = 2/dx

        #LGL Integration
        for k in range(Q+1):
            wq = wnq[k]*jac

            #Form Derivative
            h_k = 0
            hu_k = 0
            for j in range(N+1):

                l_k  = l_basis[j,k]
                hu_k = hu_k + hu_e[j]*l_k
                h_k = h_k + h_e[j]*l_k

            flux_h = hu_k
            flux_hu = hu_k**2/h_k + 0.5*gravity*h_k**2

            #Form RHS
            for i in range(N+1):
                l_i = l_basis[i,k]
                dldx_i = dl_basis[i,k]*ksi_x
                rhs[e-1,0,i] = rhs[e-1,0,i] + wq*dldx_i*flux_h
                rhs[e-1,1,i] = rhs[e-1,1,i] + wq*dldx_i*flux_hu

    return rhs
                

def applyFilter(q,f,intma,Ne,N):
    
    for e in range(1,Ne+1):
        
        q[e-1,0,:] = f@q[e-1,0,:]
        q[e-1,1,:] = f@q[e-1,1,:]
        
    return q

def filter_func(N,xn,xmu,filter_type,weight_type):
    
    n = N-1
    nh = floor((n+1)/2)
    alpha = 17
    order = 18

    #Initialize
    leg = zeros((N+1,N+1))
    leg2 = zeros((N+1,N+1))
    f = zeros((N+1,N+1))

    #Compute Legendre Polynomial Matrix
    for i in range(N+1):
        x = xn[i]
        
        for j in range(N+1):
            L0,L0_1,L0_2 = Legendre_deriv(j,x)
            
            leg[i,j] = L0
            
    #Construct Hierarchical Modal Legendre Basis
    leg2 = copy.deepcopy(leg)
    if(filter_type == 1):
        for i in range(N+1):
            x = xn[i]
            
            leg2[i,0] = 0.5*(1 - x)
            leg2[i,1] = 0.5*(1 + x)
            for j in range(2,N+1):
                leg2[i,j] = leg[i,j] - leg[i,j-2]
                

        leg = leg2
    
    #Compute Inverse
    leg_inv = linalg.inv(leg)
    
    #Compute Weight(quadratic)
    if(weight_type == 0): 
        weight = zeros(N+1)
        p = N

        for i in range(N):
            
            weight[i] = exp(-alpha*((i/p)**order))
        #print(weight)                    
    #Compute Weight(erfc-log)                
    if(weight_type == 1):        
                            
        weight = zeros(N+1)
        ibeg = round((2/3)*N)

        iend = N
        F = 12
        for i in range(ibeg):
                            
            weight[i] = 1

        for i in range(ibeg,iend+1):

            x = xn[i]
            Omega = abs(x) - 1/2
            lOm = 1-4*Omega**2
            if (lOm == 0):
                lOm = 1
            #print(Omega)
            lo = log(lOm)/(4*Omega**2)
            #print(lo)
            weight[i] = (1/2)*special.erfc(2*sqrt(F)*Omega)#*sqrt(-lo))
    
    
    #Construct 1D Filter Matrix
    for i in range(N+1):
        for j in range(N+1):
            sum1 = 0
            for k in range(N+1):
                sum1 = sum1 + leg[i,k]*weight[k]*leg_inv[k,j]
          
            f[i,j] = xmu*sum1
       
        f[i,i] = f[i,i] + (1-xmu)
    
    return f
      
def productInvMass_rhs(M_inv,N,Ne,rhs):
    
    for e in range(1,Ne+1):
        
        rhs[e-1,0,:] = M_inv[e-1,:,:]@rhs[e-1,0,:]
        rhs[e-1,1,:] = M_inv[e-1,:,:]@rhs[e-1,1,:]
        
    return rhs


def rusanov_flux(hl,ul,hr,ur,gravity,diss):

    # Compute big U
    Ul = hl*ul
    Ur = hr*ur
    
    # Compute Wave Speeds 
    lamb_l = (abs(ul) + sqrt(gravity*hl))
    lamb_r = (abs(ur) + sqrt(gravity*hr))
    lamb = max(lamb_l,lamb_r)
    
    # Mass Flux
    fl = Ul
    fr = Ur
    
    flux_h = 0.5*(fl + fr - diss*abs(lamb)*(hr - hl))
    
    #Momentum Flux
    fl = Ul**2/hl + 0.5*gravity*hl**2
    fr = Ur**2/hr + 0.5*gravity*hr**2

    flux_U = 0.5*(fl + fr - diss*abs(lamb)*(Ur - Ul))
    
    return flux_h, flux_U

def roe_flux(hl,ul,hr,ur,gravity,diss):

    # Compute big U
    Ul = hl*ul
    Ur = hr*ur
    
    # Mass Flux
    fh_l = Ul
    fh_r = Ur
    
    #Momentum Flux
    fU_l = Ul**2/hl + 0.5*gravity*hl**2
    fU_r = Ur**2/hr + 0.5*gravity*hr**2
    
    fl = array([fh_l,fU_l])
    fr = array([fh_r,fU_r])
    h_tilde = sqrt(hl*hr)
    Dh = hr - hl
    Du = ur - ul
    u_tilde = (sqrt(hl)*ul + sqrt(hr)*ur)/(sqrt(hl)+sqrt(hr))
    c_tilde = sqrt(gravity*(hl+hr)/2)
    lamb1 = u_tilde - c_tilde
    lamb2 = u_tilde + c_tilde
    
    k1 = array([1,lamb1])
    k2 = array([1,lamb2])
    
    alpha1 = 0.5*(Dh - h_tilde*Du/c_tilde)
    alpha2 = 0.5*(Dh + h_tilde*Du/c_tilde)
    
    flux = 0.5*(fl+fr - diss*(alpha1*abs(lamb1)*k1 + alpha2*abs(lamb2)*k2))
    flux_h = flux[0]
    flux_U = flux[1]
    
    return flux_h, flux_U

def flux_dg(rhs,qp,Ne,N,diss,h_eps,gravity,time,flux_type):

    #Integrate Flux Terms
    for e in range(1,Ne+2):
        el = e-2
        er = e-1

        #Left Variables
        if(e > 1):
            hl = qp[el,0,N]
            hul = qp[el,1,N]
            ul  = hul/hl
    
        #Right Variables
        if(e < Ne+1):
            hr = qp[er,0,0]
            hur = qp[er,1,0]
            ur  = hur/hr
        
        #Left NFBC
        if(e == 1):
            hl  = hr
            ul  = ur

        #Right NFBC
        if (e == Ne+1):
            hr  = hl
            ur  = ul
        
        # Apply rusanouv flux
        if(flux_type == 'rusa'):
            flux_h, flux_hu = rusanov_flux(hl,ul,hr,ur,gravity,diss)
        elif(flux_type == 'roe'):
            flux_h, flux_hu = roe_flux(hl,ul,hr,ur,gravity,diss)
        
        # Add RHS to Left
        if(e > 1):
            rhs[el,0,N] = rhs[el,0,N] - flux_h
            rhs[el,1,N] = rhs[el,1,N] - flux_hu
       
        # Add RHS to Right
        if(e < Ne+1):
            rhs[er,0,0] = rhs[er,0,0] + flux_h
            rhs[er,1,0] = rhs[er,1,0] + flux_hu
            
    return rhs



def Shu_Positivity_Preserving_limiter(qp,Ne,l_basis,N,Q,wq):
    
    qm = zeros((Ne,2, N+1))
    eps = 1e-3
    
    for e in range(1,Ne+1):
        
        Ub = zeros(2)
        
        qm[e-1,0,:] = qp[e-1,0,:]
        qm[e-1,1,:] = qp[e-1,1,:]
        
        Uq = qm[e-1,:,:]@l_basis
        m = min(Uq[0,:])
        
        for i in range(Q+1):
            Ub[0] = Ub[0] + Uq[0,i]*wq[i]
            Ub[1] = Ub[1] + Uq[1,i]*wq[i]
            
        Ub = Ub/2
        
        theta = min(1, Ub[0]/(abs(Ub[0]-m)+eps))
        #print(theta)
        for i in range(N+1):
            
            qm[e-1,:,i] = theta*(qm[e-1,:,i]-Ub) + Ub
            if(m == 0 and m == Ub[0]):
                
                qm[e-1,:,i] = 0
    return qm

def courant(qp,intma,Ne,N,gravity,dt,dx):

    u_max = -100000

    c_max = -100000

    for e in range(1,Ne+1):
        for i in range(N+1):
            
            h = qp[e-1,0,i]
            lam = sqrt(gravity*h)
            u = qp[e-1,1,i]/qp[e-1,0,i]

            u_max = max(u_max, u)
            c_max = max(c_max, u + lam)

    courant_u = u_max*dt/dx
    courant_c = c_max*dt/dx
    
    return courant_u, courant_c


def cg_dgSolver(N,Q,nel, Np, ax, bx, integration_type, u, CFL, Tfinal, kstages, icase, \
                delta_nl,h_eps,plot_movie,h_init,hu_init,xmu,ifilter,diss,method_type,\
                limiter,filter_type,weight_type,flux_type):

    '''
    This function is CG/DG solver for 1D wave equation
    
    Inputs:
    -------
            N              : Polynomial order
            Q              : Integration order(N+1: for exact, N: for inexact integration)
            nel            : Number of element
            nel0           : The first number of element 
            Np             : Global grid points(nel*N+1 for CG, nel*(N+1) for DG)
            ax, bx         : Left and right boundaries of the physical domain
            intergatio_type: Exact or Inexact integration
            method_type    : CG or DG
            icase          : For the initial condition type(icase = 1, for gaussian or 2, for sine)
            diss           : 0(centered flux), 1(Rusanov flux)
            u              : velocity
            Courant_max    : CFL
            Tfinal         : Ending time for the computation
            kstages        : Type for the time integration(2 = RK2, 3 = RK3, 4 = RK4)
            time_step      : function that compute the time step and number of time( double time per element)
    Outputs:
    --------
    
            T         : Temperature
            S         : Salinity
            coord     : All grid points
            intma     : Intma(CG/DG)
    '''

    # Compute Interpolation and Integration Points
    
    t0 = perf_counter()
    xgl = Lobatto_p(N)
    wgl = weight(N)
    
    xnq = Lobatto_p(Q)
    wnq = weight(Q)

    f = filter_func(N,xgl,xmu,filter_type,weight_type)    
    
    # Create intma
    intma = intma_cdg(N, nel, method_type)
    
    # Create Grid and space stuff
    
    coord = grid_dg(N, nel,xgl, ax, bx)
    
    dx = coord[1,0] - coord[0,0]
    
    # Lagrange basis and its derivatives
    l_basis, dl_basis = LagrangeBasis_deriv(N,Q,xgl, xnq)
    
    # Compute Initial Solutions
    t_time = 0
    qe,gravity = exact_solution(coord,nel,N,t_time,icase,ax,bx, h_init, hu_init)

    # Initial mass
        
    mass,mass_inv = Element_matrix_inv_DG(coord,nel,N,Q, wnq,l_basis)

    t1 = perf_counter()
    
    time_f = t1-t0
    
    # time stuff
    dx = (bx-ax)/Np
    dt_est = CFL*dx/u
    ntime = int(floor(Tfinal/dt_est))
    dt = Tfinal/ntime
    
    print("N = {:d}, nel = {:d}, Np = {}".format(N,nel,Np))
    print("\tdt = {:.4e}".format(dt))
    print("\tNumber of time steps = {}".format(ntime))
    
    np = nel*(N+1)
    
    if(plot_movie == True):
         
        x_sol = zeros(np)
        qe_p = zeros(np)
        qe_u = zeros(np)
        ip = 0
        for e in range(1,nel+1):
            
            for i in range(N+1):
                ip = ip+1

                x_sol[ip-1] = coord[i,e-1]
                qe_p[ip-1] = qe[e-1,0,i]
                qe_u[ip-1] = qe[e-1,1,i]
    
    
        fig = plt.figure(1)
        
        plt.subplot(211)
        movie1, = plt.plot(x_sol,qe_p,'b-',markersize=10)

        Title = 't = {:.4f}'
        htitle = plt.title(Title.format(t_time),fontsize=18)
        plt.ylabel('h', fontsize = 15)

        plt.subplot(212)
        movie2, = plt.plot(x_sol,qe_u,'r--',markersize=10)
        
        plt.xlabel('x', fontsize = 10)
        plt.ylabel('U', fontsize = 10)

    
    # Initialize for temperature
    q1 = qe
    q = qe
    qp = qe
    #Time Integration
    
    #ntime = 1
    for itime in range(ntime):
        
        #if(ifilter > 0):
        #    qp = applyFilter(qp,f,intma,nel,N)
        
        t_time = t_time + dt_est
        
        courant_u, courant_c = courant(qp,intma,nel,N,gravity,dt,dx)
        #print(courant_u,courant_c)
        
        for ik in range(1,kstages+1):
            
            rhs = create_rhs(qp,coord,nel,N,Q,wnq,l_basis,dl_basis,h_eps,gravity)
            
            rhs = flux_dg(rhs,qp,nel,N,diss,h_eps,gravity,t_time,flux_type)

            rhs = productInvMass_rhs(mass_inv,N,nel,rhs)

            # RK coefficients
            a0, a1, beta = RKstages(ik, kstages)
            
            dtt = dt*beta
            
            # Compute the new solution
            qp = a0*q + a1*q1 + dtt*rhs
            
            # Apply limiter
            if(limiter == 1):
                
                qp = Shu_Positivity_Preserving_limiter(qp,nel,l_basis,N,Q,wnq)
                
    
            #Update
            # Apply filter
            if(ifilter > 0):
                qp = applyFilter(qp,f,intma,nel,N)
                
            q1 = qp
        
        #Update q
        
        q = qp
    
        #t3 = perf_counter()
    
        # Plot movie
        if(plot_movie == True):
        
            #Compute a gridpoint solution

            q_p = zeros(np)
            q_u = zeros(np)
            
            ip = 0

            for e in range(1,nel+1):
                for i in range(N+1):
                    ip = ip+1
                    q_p[ip-1] = q[e-1,0,i]
                    q_u[ip-1] = q[e-1,1,i]

            plt.subplot(211)

            movie1.set_ydata(q_p)
            htitle.set_text(Title.format(t_time))
            #plt.ylim([-1.0,1])
            plt.pause(0.1)

            plt.subplot(212)
            movie2.set_ydata(q_u)
            
            #plt.ylim([-1.0,1])
            plt.pause(0.1)

            fig.canvas.draw()
    
    if(plot_movie == False):
        
        # compute the exact solution
        qe,gravity = exact_solution(coord,nel,N,t_time,icase,ax,bx, h_init, hu_init)

        q_p = zeros(np)
        q_u = zeros(np)
        qe_p = zeros(np)
        qe_u = zeros(np)
        x_sol = zeros(np)

        ip = 0

        for e in range(1,nel+1):
            for i in range(N+1):
                ip = ip+1
                q_p[ip-1] = q[e-1,0,i]
                q_u[ip-1] = q[e-1,1,i]

                qe_p[ip-1] = qe[e-1,0,i]
                qe_u[ip-1] = qe[e-1,1,i]
                x_sol[ip-1] = coord[i,e-1]

    return q_p,q_u, x_sol, coord, intma
