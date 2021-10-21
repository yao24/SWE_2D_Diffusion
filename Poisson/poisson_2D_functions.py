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

def map_deriv(l_basis,dl_basis,z,N,Q):
    
    z_ksi = zeros((Q+1,Q+1))
    z_eta = zeros((Q+1,Q+1))

    for l in range(Q+1):
        for k in range(Q+1):

            sum_ksi = 0
            sum_eta = 0

            for j in range(N+1):
                for i in range(N+1):
                    
                    z_ksi[k,l] += dl_basis[i,k]*l_basis[j,l]*z[i,j]
                    z_eta[k,l] += l_basis[i,k]*dl_basis[j,l]*z[i,j]
        
    return z_ksi, z_eta

def metrics(coord,intma,l_basis,dl_basis,wnq,Ne,N,Q):

    #Initialize Global Arrays
    ksi_x = zeros((Ne,Q+1,Q+1))
    ksi_y = zeros((Ne,Q+1,Q+1))
    eta_x = zeros((Ne,Q+1,Q+1))
    eta_y = zeros((Ne,Q+1,Q+1))
    jac = zeros((Ne,Q+1,Q+1))

    #Initialize Local Arrays
    x_ksi = zeros((Q+1,Q+1))
    x_eta = zeros((Q+1,Q+1))
    y_ksi = zeros((Q+1,Q+1))
    y_eta = zeros((Q+1,Q+1))
    x = zeros((N+1,N+1))
    y = zeros((N+1,N+1))

    #loop thru the elements
    for e in range(1,Ne+1):

        #Store Element Variables
        for j in range(N+1):
            for i in range(N+1):
                ip = int(intma[e-1,i,j])
                #print(ip)
                x[i,j] = coord[ip,0]
                y[i,j] = coord[ip,1]
        #print(x)
        #Construct Mapping Derivatives: dx/dksi, dx/deta, dy/dksi, dy/deta
        x_ksi,x_eta = map_deriv(l_basis,dl_basis,x,N,Q)
        y_ksi,y_eta = map_deriv(l_basis,dl_basis,y,N,Q)
        #print(y_ksi)
        #Construct Inverse Mapping: dksi/dx, dksi/dy, deta/dx, deta/dy
        for j in range(Q+1):
            for i in range(Q+1):
                xjac = x_ksi[i,j]*y_eta[i,j] - x_eta[i,j]*y_ksi[i,j]
                ksi_x[e-1,i,j] = 1/xjac*y_eta[i,j]
                ksi_y[e-1,i,j] = -1/xjac*x_eta[i,j]
                eta_x[e-1,i,j] = -1/xjac*y_ksi[i,j]
                eta_y[e-1,i,j] = +1/xjac*x_ksi[i,j]
                jac[e-1,i,j]   = wnq[i]*wnq[j]*abs(xjac)
                
    return ksi_x,ksi_y,eta_x,eta_y,jac

def grid_2D(Np, Ne, Nbound, Nex, Ney, N, Q, Xn, ax, bx):

    #Initialize Global Arrays
    coord = zeros((Np,2))
    intma = zeros((Ne,N+1,N+1))
    bsido = zeros(Nbound)

    #Initialize Local Arrays
    node = zeros((Np,Np))

    #Set some constants
    dx = (bx-ax)/Nex 
    dy = (bx-ax)/Ney
    
    Nx = Nex*N + 1
    Ny = Ney*N + 1
    
    #GENERATE COORD
    
    ip = 0
    jj = 0
    
    for ey in range(1,Ney+1):
        
        y0 = ax + (ey-1)*dy

        if(ey == 1): 
            l1 = 0
        else:
            l1 = 1

        for l in range(l1,N+1):
            
            y = (Xn[l]+1)*dy/2 + y0
            ii = 0

            for ex in range(1,Nex+1):
                
                x0 = ax + (ex-1)*dx

                if(ex == 1):
                    j1 = 0
                else:
                    j1 = 1

                for j in range(j1,N+1):
                    x = (Xn[j]+1)*dx/2 + x0
                    coord[ip,0] = x
                    coord[ip,1] = y
                    node[ii,jj] = ip
                    ii = ii + 1
                    ip = ip + 1 
            jj = jj + 1
     
    # GENERATE INTMA
    e = 0
    for ey in range(1,Ney+1):
        
        for ex in range(1,Nex+1):
            
            for l in range(N+1):
                
                jj = N*(ey-1) + l
                
                for j in range(N+1):
                    ii = N*(ex-1) + j
                    ip = node[ii,jj]
                    intma[e,j,l] = ip   
            e = e+1
    #Generate BSIDO
    ib = 0
    #Bottom Boundary
    for ex in range(1,Nx+1):

        ip1 = node[ex-1,0]
        bsido[ib] = ip1
        ib = ib + 1
        
    #Right Boundary
    for ey in range(2,Ny+1):

        ip1 = node[Nx-1,ey-1]
        bsido[ib] = ip1
        ib = ib+1       
    #Top Boundary
    for ex in range(Nx-1,0,-1):

        ip1 = node[ex-1,Ny-1]
        bsido[ib] = ip1
        ib = ib+1
    #Left Boundary
    for ey in range(Ny-1,1,-1):

        ip1 = node[0,ey-1]
        bsido[ib] = ip1
        ib = ib+1
        
    return coord, intma, bsido


def create_Mmatrix(jac,intma,l_basis,Np,Ne,N,Q):

    #Initialize
    Mmatrix = zeros((Np,Np))

    for e in range(1,Ne+1):

       #Do LGL Integration
        for l in range(Q+1):
            for k in range(Q+1):
                
                wq = jac[e-1,k,l]
                
                for j in range(N+1):
                    for i in range(N+1):
                        
                        jp = int(intma[e-1,i,j])
                        h_j = l_basis[i,k]*l_basis[j,l]
                        
                        for n in range(N+1):
                            for m in range(N+1):
                                
                                ip = int(intma[e-1,m,n])
                                h_i = l_basis[m,k]*l_basis[n,l]
                                Mmatrix[ip,jp] += wq*h_i*h_j

    return Mmatrix

def create_Lmatrix(intma,jac,ksi_x,ksi_y,eta_x,eta_y,l_basis,dl_basis,Np,Ne,N,Q):

    #Initialize
    Lmatrix = zeros((Np,Np))
    inode = zeros((N+1,N+1))

    for e in range(1,Ne+1):

       #Do LGL Integration
        for l in range(Q+1):
            for k in range(Q+1):
                wq = jac[e-1,k,l]
                e_x = ksi_x[e-1,k,l]
                e_y = ksi_y[e-1,k,l]
                n_x = eta_x[e-1,k,l]
                n_y = eta_y[e-1,k,l]
                
            
                #Loop through I points
                for j in range(N+1):
                    for i in range(N+1):
                        jp = int(intma[e-1,i,j])
                        h_e = dl_basis[i,k]*l_basis[j,l]
                        h_n = l_basis[i,k]*dl_basis[j,l]
                        dhdx_j = h_e*e_x + h_n*n_x
                        dhdy_j = h_e*e_y + h_n*n_y
                    
                        #Interpolate Derivatives onto Quadrature Points

                        for n in range(N+1):
                            for m in range(N+1):
                                ip = int(intma[e-1,m,n])
                                h_e = dl_basis[m,k]*l_basis[n,l]
                                h_n = l_basis[m,k]*dl_basis[n,l]
                                dhdx_i = h_e*e_x + h_n*n_x
                                dhdy_i = h_e*e_y + h_n*n_y

                                Lmatrix[ip,jp] = Lmatrix[ip,jp] - wq*(dhdx_i*dhdx_j + dhdy_i*dhdy_j)

            

    return Lmatrix

def create_side(intma,bsido,Np,Ne,Nbound,Nside,N):
    Q = N+1
    # global arrays
    iside = -ones((Nside,4))
    psideh = -ones((Nside,4))

    # local arrays
    lwher = zeros(Np)
    lhowm = zeros(Np)
    icone = zeros(5*Np)
    inode = zeros(4)
    jnode = zeros(4)

    # Fix lnode
    inode[0] = 0
    inode[1] = Q-1
    inode[2] = Q-1
    inode[3] = 0
    jnode[0] = 0
    jnode[1] = 0
    jnode[2] = Q-1
    jnode[3] = Q-1

    # Count how many elements own each node
    for i in range(4):
        ii = int(inode[i])
        jj = int(jnode[i])
        for ie in range(1,Ne+1):
            ip = int(intma[ie-1,jj,ii])          
            lhowm[ip] = lhowm[ip] + 1
    # Track elements owning each node   
    lwher[0] = 0
    for ip in range(1,Np):        
        lwher[ip] = lwher[ip-1] + lhowm[ip-1]   
    # another tracker array
    lhowm = zeros(Np)   
    for i in range(4):       
        ii = int(inode[i])
        jj = int(jnode[i])       
        for ie in range(1,Ne+1):            
            ip = int(intma[ie-1,jj,ii])
            lhowm[ip] = lhowm[ip] + 1
            jloca = int(lwher[ip] + lhowm[ip])
            icone[jloca-1] = ie
    
    #LOOP OVER THE NODES
    iloca = 0
    for ip in range(Np):
        iloc1 = iloca
        iele = int(lhowm[ip])  
        
        if(iele != 0):
            iwher = int(lwher[ip])
            #LOOP OVER THOSE ELEMENTS SURROUNDING NODE IP
            ip1 = ip 
            for iel in range(iele):
                ie = int(icone[iwher+iel])
                #find out position of ip in intma
                for i in range(4):                    
                    in1 = i
                    ii = int(inode[i])
                    jj = int(jnode[i])
                    ipt = int(intma[ie-1,jj,ii])                   
                    if(ipt == ip): 
                        break 

                #Check Edge of Element IE which claims IP
                j = 0
                for jnod in range(1,4,2):
                    iold = 0
                    j = j+1
                    in2 = i+jnod
                    if(in2 > 3): 
                        in2 = in2-4
                    ip2 = int(intma[ie-1,int(jnode[in2]),int(inode[in2])])
                    if(ip2 >= ip1): 
                        # Check whether side is old or new
                        if(iloca != iloc1):
                            for iin in range(iloc1+1,iloca+1):
                                jloca = iin
                                if(int(iside[iin-1,1]) == ip2): 
                                    iold = 1
                                    break
                        if(iold == 0):
                            #NEW SIDE
                            iloca = iloca + 1
                            iside[iloca-1,0] = ip1
                            iside[iloca-1,1] = ip2
                            iside[iloca-1,1+j] = ie-1
                            
                        elif(iold == 1):   
                            #OLD SIDE
                            iside[jloca-1,1+j] = ie-1
                            
            #Perform some Shifting to order the nodes of a side in CCW direction
            for iis in range(iloc1+1,iloca+1):           
                if(iside[iis-1,2] == -1):
                    iside[iis-1,2] = iside[iis-1,3]
                    iside[iis-1,3] = -1
                    iside[iis-1,0] = iside[iis-1,1]
                    iside[iis-1,1] = ip1
            
            
    if(iloca != Nside): 
        print('Error in SIDE. iloca nside = ')
        print(iloca)
        print(Nside)
        #pause
     
    '''
    #RESET THE BOUNDARY MARKERS
    for iis in range(Nside):
        if(int(iside[iis,3]) == -1): 
            il = int(iside[iis,0])
            ir = int(iside[iis,1])
            ie = int(iside[iis,2])
            for ib in range(Nbound):
                ibe = int(bsido[ib,2])
                ibc = int(bsido[ib,3])
                
                if(ibe == ie):
                    
                    ilb = int(bsido[ib,0])
                    irb = int(bsido[ib,1])
                    if(ilb == il and irb == ir):
                        iside[iis,3] =- ibc
                        break
    '''
    #FORM ELEMENT/SIDE CONNECTIVITY ARRAY
    #loop thru the sides
    for i in range(Nside):

        ip1 = int(iside[i,0])
        ip2 = int(iside[i,1])
        iel = int(iside[i,2])
        ier = int(iside[i,3])

        #check for position on Left Element
        for j in range(4):
            
            j1 = j
            j2 = j+1
            
            if(j2 > 3):
                j2 = 0

            jp1 = int(intma[iel,int(jnode[j1]),int(inode[j1])])
            jp2 = int(intma[iel,int(jnode[j2]),int(inode[j2])])
            
            if(ip1 == jp1 and ip2 == jp2):
                
                psideh[i,0] = j
                break
        
        #check for position on Right Element
        if(ier > 0):
            for j in range(4):
                j1 = j
                j2 = j+1
                if(j2 > 3):
                    j2 = 0

                jp1 = intma[ier,int(jnode[j1]),int(inode[j1])]
                jp2 = intma[ier,int(jnode[j2]),int(inode[j2])]

                if(ip1 == jp2 and ip2 == jp1): 
                    psideh[i,1] = j
                    break

        #Store Elements into PSIDEH
        psideh[i,2] = iel
        psideh[i,3] = ier

    return iside,psideh

def compute_normals(psideh,intma,coord,Nside,N,Q,wnq,l_basis,dl_basis):

    #global arrays
    nx = zeros((Nside,Q+1)) 
    ny = zeros((Nside,Q+1))
    nz = zeros((Nside,Q+1))
    jac_side = zeros((Nside,Q+1))

    #local arrays
    x = zeros((N+1,N+1))
    y = zeros((N+1,N+1))
    x_ksi = zeros((Q+1,Q+1))
    x_eta = zeros((Q+1,Q+1))
    y_ksi = zeros((Q+1,Q+1)) 
    y_eta = zeros((Q+1,Q+1))

    #loop thru the sides
    for iis in range(Nside):

        #Store Left and Right Elements
        ilocl = int(psideh[iis,0])
        iel   = int(psideh[iis,2])

        #Store Element Variables
        for j  in range(N+1):
            for i in range(N+1):
                ip = int(intma[iel,j,i])
                x[i,j] = coord[ip,0]
                y[i,j] = coord[ip,1] 

        #Construct Mapping Derivatives: dx/dksi, dx/deta, dy/dksi,dy/deta
        x_ksi,x_eta = map_deriv(l_basis,dl_basis,x,N,Q)
        y_ksi,y_eta = map_deriv(l_basis,dl_basis,y,N,Q)
        
        #Compute Normals 
        for l in range(Q+1):
            wq = wnq[l]
            if(ilocl == 0):
                #Side 0: eta=-1
                i = l
                j = 0
                nx[iis,l] =+ y_ksi[i,j]
                ny[iis,l] =- x_ksi[i,j]
                jac_side[iis,l] = wq*sqrt(x_ksi[i,j]**2 + y_ksi[i,j]**2)

            elif(ilocl == 1):
                #Side 1: ksi=+1
                i = Q
                j = l
                nx[iis,l] =+ y_eta[i,j]
                ny[iis,l] =- x_eta[i,j]
                jac_side[iis,l] = wq*sqrt(x_eta[i,j]**2 + y_eta[i,j]**2)

            elif(ilocl == 2):
                #Side 2: eta=+1
                i = Q-l
                j = Q
                nx[iis,l] =- y_ksi[i,j]
                ny[iis,l] =+ x_ksi[i,j]
                jac_side[iis,l] = wq*sqrt(x_ksi[i,j]**2 + y_ksi[i,j]**2)

            elif(ilocl == 3):
                #Side 3: ksi=-1
                i = 0
                j = Q-l
                nx[iis,l] =- y_eta[i,j]
                ny[iis,l] =+ x_eta[i,j]
                jac_side[iis,l] = wq*sqrt(x_eta[i,j]**2 + y_eta[i,j]**2)  

        #Normalize Norms
        for l in range(Q+1):
            rnx = sqrt(nx[iis,l]**2 + ny[iis,l]**2)
            nx[iis,l] = nx[iis,l]/rnx
            ny[iis,l] = ny[iis,l]/rnx
        
    return nx,ny,jac_side

def exact_solution(coord,Np,c,icase):

    #Initialize
    qe = zeros(Np)
    fe = zeros(Np)
    cc = pi*c

    #Generate Grid Points
    if(icase == 1):
        for i in range(Np):
            x = coord[i,0]
            y = coord[i,1]
            qe[i] = sin(cc*x)*sin(cc*y)
            fe[i] = -2*cc**2*sin(cc*x)*sin(cc*y)
    elif(icase == 2): 
        for i in range(Np):
            x = coord[i,0]
            y = coord[i,1]
            qe[i] = cos(cc*x)*cos(cc*y)
            fe[i] = -2*cc**2*cos(cc*x)*cos(cc*y)
            
    elif(icase == 3):
        
        for i in range(Np):
            x = coord[i,0]
            y = coord[i,1]
            qe[i] = -sin(5*x)
            fe[i] = 10*exp(-(x-0.5)**2 + (y-0.5)**2/0.02)
    
    return qe,fe

def apply_Dirichlet_BC(Lmatrix,Rvector,bsido,qe,Nbound):

    for i in range(Nbound):
            ip = int(bsido[i])
            Lmatrix[ip,:] = 0
            Lmatrix[ip,ip] = 1
            Rvector[ip] = qe[ip]

    return Lmatrix,Rvector


def apply_Neumann_BC(nx,ny,jac_side,iside,l_basis,dq,Np,Q,Nside,Nbound):
    
    B = zeros(Np)
    
    for n in range(Nbound):
        ip = int(iside[n,2])
        
        for j in range(Nside):
            for i in range(Q+1):
                
                #wq = jac_side[j,i]
                
                
                il = imapl[0,i,j,iface]
                jl = imapl[1,i,j,iface]
                
                ip = intma[il,jl,iel]
                nx = normal_vector[0,i,j,iface]
                ny = normal_vector[1,i,j,iface]
             
                ndp = nx*gradp[0,ip] + ny*gradp[1,ip]

                B[ip] = wq*ndp
                
    return B
      


def poisson_solver(N,Q,Ne, Np, ax, bx, Nelx, Nely, Nx, Ny, Nbound,Nside,c,icase):
    
    # Compute Interpolation and Integration Points
    
    #t0 = perf_counter()
    xgl = Lobatto_p(N)
    wgl = weight(N)
    
    xnq = Lobatto_p(Q)
    wnq = weight(Q)
    
    # Lagrange basis and its derivatives
    l_basis, dl_basis = LagrangeBasis_deriv(N,Q,xgl, xnq)
    
    coord, intma, bsido = grid_2D(Np, Ne, Nbound, Nelx, Nely, N, Q, xgl, ax, bx)
    dx = coord[1,0]-coord[0,0]
    #print(coord)
    ksi_x,ksi_y,eta_x,eta_y,jac = metrics(coord,intma,l_basis,dl_basis,wnq,Ne,N,Q)
    #print(jac)
    iside,psideh = create_side(intma,bsido,Np,Ne,Nbound,Nside,N)
    #print(iside)
    nx,ny,jac_side = compute_normals(psideh,intma,coord,Nside,N,Q,wnq,l_basis,dl_basis)
    
    qe,fe = exact_solution(coord,Np,c,icase)
    
    Mmatrix = create_Mmatrix(jac,intma,l_basis,Np,Ne,N,Q)
    #print(Mmatrix)
    Lmatrix = create_Lmatrix(intma,jac,ksi_x,ksi_y,eta_x,eta_y,l_basis,dl_basis,Np,Ne,N,Q)
    #print(Lmatrix)
    #Impose Homogeneous Dirichlet Boundary Conditions

    Rvector = Mmatrix@fe
    #print(Rvector)
    
    Lmatrix, Rvector = apply_Dirichlet_BC(Lmatrix,Rvector,bsido,qe,Nbound)
    #B = apply_Neumann_BC(jac_side,l_basis,bsido,qe,Np,Q,Nside,Nbound)
    
    #Solve System 
    #Lmatrix_inv = linalg.inv(Lmatrix)
    #print(Lmatrix_inv)
    q0 = linalg.solve(Lmatrix,Rvector)
    #q0 = Lmatrix_inv@Rvector
    
    
    return qe, q0, coord, intma, Mmatrix