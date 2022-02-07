from numpy import *
from time import perf_counter
from scipy.sparse import csr_matrix
#from matplotlib.pyplot import*
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import special
import copy
from scipy.interpolate import griddata
from scipy.optimize import fsolve

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
    
    #X = []                                      # Array that contains legendre points
    X = zeros(Q+1)
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

        X[i] = xikk#.append(xikk)
        
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
                    
                    sum_ksi += dl_basis[i,k]*l_basis[j,l]*z[i,j]
                    sum_eta += l_basis[i,k]*dl_basis[j,l]*z[i,j]
                    
            z_ksi[k,l] = sum_ksi
            z_eta[k,l] = sum_eta
            
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
                ksi_x[e-1,i,j] = +(1/xjac)*y_eta[i,j]
                ksi_y[e-1,i,j] = -(1/xjac)*x_eta[i,j]
                eta_x[e-1,i,j] = -(1/xjac)*y_ksi[i,j]
                eta_y[e-1,i,j] = +(1/xjac)*x_ksi[i,j]
                jac[e-1,i,j]   = wnq[i]*wnq[j]*abs(xjac)
                
    return ksi_x,ksi_y,eta_x,eta_y,jac

def grid_2D(Np, Ne, Nbound, Nex, Ney, N, Q, Xn, ax, bx,bound_index):

    #Initialize Global Arrays
    coord = zeros((Np,2))
    intma = zeros((Ne,N+1,N+1))
    bsido = zeros(Nbound)
    nboun = 2*Nex + 2*Ney
    face = zeros((nboun,4))

    #Initialize Local Arrays
    node = zeros((Np,Np))

    #Set some constants
    dx = (bx-ax)/(Nex) 
    dy = (bx-ax)/(Ney)
    
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
        
    #Generate Face
    ib = 0
    #Bottom Boundary
    for ex in range(1,Nex+1):
        ie = ex
        i1 = (ex-1)*N + 1
        i2 = (ex-1)*N + N + 1
        
        ip1 = node[i1-1,0]
        ip2 = node[i2-1,0]
        
        face[ib,0] = ip1
        face[ib,1] = ip2
        face[ib,2] = ie-1
        face[ib,3] = bound_index[0]
        ib = ib + 1

    #Right Boundary
    for ey in range(1,Ney+1):
        ie = Nex*ey
        i1 = (ey-1)*N + 1
        i2 = (ey-1)*N + N+1 
        
        ip1 = node[Nx-1,i1-1]
        ip2 = node[Nx-1,i2-1]
        face[ib,0] = ip1
        face[ib,1] = ip2
        face[ib,2] = ie-1
        face[ib,3] = bound_index[1]
        ib = ib+1
    #Top Boundary
    for ex in range(Nex,0,-1):
        ie = Ne - (Nex - ex)
        i1 = (ex-1)*N + N+1
        i2 = (ex-1)*N + 1
        ip1 = node[i1-1,Ny-1]
        ip2 = node[i2-1,Ny-1]
        
        face[ib,0] = ip1
        face[ib,1] = ip2
        face[ib,2] = ie-1
        face[ib,3] = bound_index[2]
        ib = ib+1
    #Left Boundary
    for ey in range(Ney,0,-1):
        ie = (Nex)*(ey-1) + 1
        
        i1 = (ey-1)*N + N+1
        i2 = (ey-1)*N + 1
        ip1 = node[0,i1-1]
        ip2 = node[0,i2-1]
        face[ib,0] = ip1
        face[ib,1] = ip2
        face[ib,2] = ie-1
        face[ib,3] = bound_index[3]
        ib = ib+1
    
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
    #bsido = face[:,0]    
    return coord, intma, bsido,face


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
                #print(wq)
                e_x = ksi_x[e-1,k,l]
                e_y = ksi_y[e-1,k,l]
                n_x = eta_x[e-1,k,l]
                n_y = eta_y[e-1,k,l]
                
            
                #Loop through I points
                for j in range(N+1):
                    for i in range(N+1):
                        jp = int(intma[e-1,i,j])
                        #print('jp = ',jp)
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

                                Lmatrix[ip,jp] -= wq*(dhdx_i*dhdx_j + dhdy_i*dhdy_j)

            

    return Lmatrix

def create_side(intma,face,Np,Ne,Nbound,Nside,N):
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
    inode[1] = N
    inode[2] = N
    inode[3] = 0
    jnode[0] = 0
    jnode[1] = 0
    jnode[2] = N
    jnode[3] = N
    
    # Count how many elements own each node
    for i in range(4):
        ii = int(inode[i])
        jj = int(jnode[i])
        for ie in range(1,Ne+1):
            ip = int(intma[ie-1,ii,jj])          
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
            ip = int(intma[ie-1,ii,jj])
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
                    ipt = int(intma[ie-1,ii,jj])                   
                    if(ipt == ip): 
                        break 

                #Check Edge of Element IE which claims IP
                j = 0
                for jnod in range(1,4,2):
                    iold = 0
                    j = j+1
                    #J.append(j)
                    #print(j)
                    in2 = i+jnod
                    if(in2 > 3): 
                        in2 = in2-4
                    ip2 = int(intma[ie-1,int(inode[in2]),int(jnode[in2])])
                    if(ip2 >= ip1): 
                        #print(ip1+1, ip2+1)
                        # Check whether side is old or new
                        if(iloca != iloc1):
                            #print(ip1+1,ip2+1,iloc1, iloca)
                            for iin in range(iloc1+1,iloca+1):
                                #print(iin)
                                jloca = iin
                                #print(iside[iin-1,1]+1)
                                if(int(iside[iin-1,1]) == ip2):
                                    #print(ip1+1,ip2+1,iloc1, iloca)
                                    iold = 1
                                    #print(iold)
                                    break
                        if(iold == 0):
                            #NEW SIDE
                            iloca = iloca + 1
                            #print(iloca)
                            iside[iloca-1,0] = ip1
                            iside[iloca-1,1] = ip2
                            iside[iloca-1,1+j] = ie-1
                            #print(iold,ip1+1,ip2+1,j,jnod)
                        elif(iold == 1):   
                            #OLD SIDE
                            iside[jloca-1,1+j] = ie-1
                            #print(2+j)
                            #print(iold,ip1+1,ip2+1,j,jnod)
                    #j = j+1
            #Perform some Shifting to order the nodes of a side in CCW direction
            
            for iis in range(iloc1+1,iloca+1):           
                if(iside[iis-1,2] == -1):
                    iside[iis-1,2] = iside[iis-1,3]
                    iside[iis-1,3] = -1
                    iside[iis-1,0] = iside[iis-1,1]
                    iside[iis-1,1] = ip1
            
    #print(iside+1)       
    if(iloca != Nside): 
        print('Error in SIDE. iloca nside = ')
        print(iloca)
        print(Nside)
        #pause
     
    
    #RESET THE BOUNDARY MARKERS
    
    for iis in range(Nside):
        if(int(iside[iis,3]) == -1): 
            il = int(iside[iis,0])
            ir = int(iside[iis,1])
            ie = int(iside[iis,2])
            for ib in range(Nbound):
                ibe = int(face[ib,2])
                ibc = int(face[ib,3])
                
                if(ibe == ie):
                    
                    ilb = int(face[ib,0])
                    irb = int(face[ib,1])
                    if(ilb == il and irb == ir):
                        iside[iis,3] =- ibc
                        break
    
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

def create_face(iside,intma,Nside,N):
    
    psideh = -ones((Nside,4))
    imapl = zeros((N+1,4,2))
    imapr = zeros((N+1,4,2))
    
    inode = zeros(4)
    jnode = zeros(4)

    # Fix lnode
    inode[0] = 0
    inode[1] = N
    inode[2] = N
    inode[3] = 0
    jnode[0] = 0
    jnode[1] = 0
    jnode[2] = N
    jnode[3] = N
    
    
    for l in range(N+1):

        #eta=-1
        imapl[l,0,0] = l
        imapl[l,0,1] = 0
        imapr[l,0,0] = N-l
        imapr[l,0,1] = 0

        #ksi=+1
        imapl[l,1,0] = N
        imapl[l,1,1] = l
        imapr[l,1,0] = N
        imapr[l,1,1] = N-l

        #eta=+1
        imapl[l,2,0] = N-l
        imapl[l,2,1] = N
        imapr[l,2,0] = l
        imapr[l,2,1] = N

        #ksi=-1     
        imapl[l,3,0] = 0
        imapl[l,3,1] = N-l
        imapr[l,3,0] = 0
        imapr[l,3,1] = l
    
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

            jp1 = int(intma[iel,int(inode[j1]),int(jnode[j1])])
            jp2 = int(intma[iel,int(inode[j2]),int(jnode[j2])])
            
            if(ip1 == jp1 and ip2 == jp2):
                
                psideh[i,0] = j
                break
        
        #check for position on Right Element
        if(ier >= 0):
            for j in range(4):
                j1 = j
                j2 = j+1
                if(j2 > 3):
                    j2 = 0

                jp1 = int(intma[ier,int(inode[j1]),int(jnode[j1])])
                jp2 = int(intma[ier,int(inode[j2]),int(jnode[j2])])

                if(ip1 == jp2 and ip2 == jp1): 
                    psideh[i,1] = j
                    break

        #Store Elements into PSIDEH
        psideh[i,2] = iel
        psideh[i,3] = ier

    return psideh, imapl, imapr

def compute_normals(psideh,intma,coord,Nside,N,Q,wnq,l_basis,dl_basis):

    #global arrays
    nx = zeros((Nside,Q+1)) 
    ny = zeros((Nside,Q+1))
    
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
                ip = int(intma[iel,i,j])
                x[i,j] = coord[ip,0]
                y[i,j] = coord[ip,1] 

        #Construct Mapping Derivatives: dx/dksi, dx/deta, dy/dksi,dy/deta
        x_ksi,x_eta = map_deriv(l_basis,dl_basis,x,N,Q)
        y_ksi,y_eta = map_deriv(l_basis,dl_basis,y,N,Q)
        #print(x_ksi, x_eta)
        
        #Compute Normals 
        for l in range(Q+1):
            wq = wnq[l]
            if(ilocl == 0):
                #Side 0: eta=-1
                i = l
                j = 0
                xjac = x_ksi[i,j]*y_eta[i,j] - x_eta[i,j]*y_ksi[i,j]
                nx[iis,l] =+ y_ksi[i,j]
                ny[iis,l] =- x_ksi[i,j]
                jac_side[iis,l] = wq*sqrt((x_ksi[i,j])**2 + (y_ksi[i,j])**2)
         
            elif(ilocl == 1):
                #Side 1: ksi=+1
                i = Q
                j = l
                xjac = x_ksi[i,j]*y_eta[i,j] - x_eta[i,j]*y_ksi[i,j]
                nx[iis,l] =+ y_eta[i,j]
                ny[iis,l] =- x_eta[i,j]
                jac_side[iis,l] = wq*sqrt(x_eta[i,j]**2 + y_eta[i,j]**2)
                
            elif(ilocl == 2):
                #Side 2: eta=+1
                i = Q-l
                j = Q
                xjac = x_ksi[i,j]*y_eta[i,j] - x_eta[i,j]*y_ksi[i,j]
                nx[iis,l] =- y_ksi[i,j]
                ny[iis,l] =+ x_ksi[i,j]
                jac_side[iis,l] = wq*sqrt(x_ksi[i,j]**2 + y_ksi[i,j]**2)
                
            elif(ilocl == 3):
                #Side 3: ksi=-1
                i = 0
                j = Q-l
                xjac = x_ksi[i,j]*y_eta[i,j] - x_eta[i,j]*y_ksi[i,j]
                nx[iis,l] =- y_eta[i,j]
                ny[iis,l] =+ x_eta[i,j]
                jac_side[iis,l] = wq*sqrt(x_eta[i,j]**2 + y_eta[i,j]**2)  
        
        #Normalize Norms
        for l in range(Q+1):
            rnx = sqrt(nx[iis,l]**2 + ny[iis,l]**2)
            
            nx[iis,l] = nx[iis,l]/rnx
            ny[iis,l] = ny[iis,l]/rnx
        
    return nx,ny,jac_side


# Exact solution, source term and derivatives for Neumann BCs
def exact_solution(coord,Np,icase):

    #Initialize
    
    gradq = zeros((Np,2))
    qe = zeros(Np)
    
    #Generate Grid Points
    x = coord[:,0]
    y = coord[:,1]
    
    if(icase == 1):
        
        qe = y*(1-y)*x**3
        fe = 6*x*y*(1-y) - 2*x**3
        gradq[:,0] = 3*y*(1-y)*x**2
        gradq[:,1] = (1-2*y)*x**3
            
    elif(icase == 2):
        
        qe = (1-x**2)*(2*y**3-3*y**2+1)
        fe = -2*(2*y**3-3*y**2+1) + 6*(1-x**2)*(2*y-1)
        gradq[:,0] = -2*x*(2*y**3-3*y**2+1)
        gradq[:,1] = (1-x**2)*(6*y**2-6*y)
        
    elif(icase == 3):

        qe = sin(pi*x)*sin(pi*y)
        fe = -2*(pi**2)*sin(pi*x)*sin(pi*y)
        gradq[:,0] = pi*cos(pi*x)*sin(pi*y)
        gradq[:,1] = pi*sin(pi*x)*cos(pi*y)
        
    elif(icase == 4):

        qe = sin(2*pi*x)
        fe = -4*(pi**2)*sin(2*pi*x)
        gradq[:,0] = 2*pi*cos(2*pi*x)
        gradq[:,1] = 0
    
    elif(icase == 5):
        
        qe = cos(pi*x)*cos(pi*y)
        fe = -2*pi**2*cos(pi*x)*cos(pi*y)
        gradq[:,0] = -pi*sin(pi*x)*cos(pi*y)
        gradq[:,1] = -pi*cos(pi*x)*sin(pi*y)
            
    return fe,qe,gradq

def apply_Dirichlet_BC(Lmatrix,bsido,Nbound):

    for i in range(Nbound):
            ip = int(bsido[i])
            Lmatrix[ip,:] = 0
            Lmatrix[ip,ip] = 1

    return Lmatrix

def apply_Dirichlet_BC_vec(Rvector,bsido,qe,Nbound):

    for i in range(Nbound):
            ip = int(bsido[i])
            Rvector[ip] = qe[ip]

    return Rvector

def apply_Dirichlet_BC_vec1(Rvector,qe,intma,jac_side,imapl,psideh,Np,N,Q,Ne,Nelx,Nside,mixed):
    
    for n in range(Nside):
        
        el = int(psideh[n,2])
        iloc = int(psideh[n,0])
        er = int(psideh[n,3])
        
        flag = 0
        
        if(er < 0):
        
            flag = er%-10
        
        if(flag == -5 and mixed == True):

            il1 = int(imapl[0,0,0])
            jl1 = int(imapl[N,0,0])
            ip1 = int(intma[0,0,0])
            ip2 = int(intma[Ne-Nelx,il1,jl1])
            
        if(flag == -5):
            #                       print("Dirichlet ",flag)
            for i in range(Q+1):

                il = int(imapl[i,iloc,0])
                jl = int(imapl[i,iloc,1])
                ip = int(intma[el,il,jl])
                
                if(mixed == True and (ip == ip1 or ip == ip2)):
                    
                    Rvector[ip] = Rvector[ip]
                else:
                    Rvector[ip] = qe[ip]
        
    return Rvector

def apply_Dirichlet_BC_matrix(Matrix,intma,imapl,psideh,Np,N,Q,Nside):
    
    for n in range(Nside):
        
        el = int(psideh[n,2])
        iloc = int(psideh[n,0])
        er = int(psideh[n,3])
        
        flag = 0
        
        if(er < 0):
            flag = er%-10
        
        if(flag == -5):
            for i in range(Q+1):

                il = int(imapl[i,iloc,0])
                jl = int(imapl[i,iloc,1])
                ip = int(intma[el,il,jl])

                Matrix[ip,:] = 0
                Matrix[ip,ip] = 1
                
    return Matrix

def apply_Neumann_BC(q,nx,ny,intma,jac_side,l_basis,imapl,psideh,gradq,Np,N,Q,Nside):
    
    B = zeros(Np)
    
    for n in range(Nside):
        
        el = int(psideh[n,2])
        iloc = int(psideh[n,0])
        er = int(psideh[n,3])
        
        flag = 0
        
        if(er < 0):
            
            flag = er%-10
        
        if(flag == -4):
            
            for i in range(Q+1):

                wq = jac_side[n,i]
                
                nxl = nx[n,i]
                nyl = ny[n,i]
                
                il = int(imapl[i,iloc,0])
                jl = int(imapl[i,iloc,1])
                ip = int(intma[el,il,jl])

                ndp = nxl*gradq[ip,0] + nyl*gradq[ip,1]
                
                B[ip] += wq*ndp
                
                #print(ndp)
                
    q -= B
        
    return q

def extract_ice_bc(qp,intma,imapl,psideh,Np,N,Q,Nside):
    
    bc_ice1 = zeros(Np)
    
    for n in range(Nside):
        
        el = int(psideh[n,2])
        iloc = int(psideh[n,0])
        er = int(psideh[n,3])
        
        flag = 0
        
        if(er < 0):
            flag = er%-10
        
        if(flag == -7):
            for i in range(Q+1):

                il = int(imapl[i,iloc,0])
                jl = int(imapl[i,iloc,1])
                ip = int(intma[el,il,jl])

                bc_ice1[ip] = qp[ip]
                
    bc_ice = bc_ice1[where(bc_ice1!=0)]
                
    return bc_ice


def poisson_solver(N,Q,Ne, Np, ax, bx,ay,by, Nelx, Nely, Nx, Ny, Nbound,Nside,icase,x_boundary,y_boundary):
    
    # Compute Interpolation and Integration Points
    
    t0 = perf_counter()
    
    xgl = Lobatto_p(N)    # Compute Lobatto points
    xnq = Lobatto_p(Q)   # Compute Lobatto points
    wnq = weight(Q)      # Compute the weight values

    # Lagrange basis and its derivatives
    l_basis, dl_basis = LagrangeBasis_deriv(N,Q,xgl, xnq)
    
    # Put the type of grid into array for boundary conditions purpose
    bound_index = zeros(4)
    bound_index[0] = y_boundary[0]
    bound_index[2] = y_boundary[1]
    
    bound_index[1] = x_boundary[1]
    bound_index[3] = x_boundary[0]
    
    # Check whether the boundary conditions are mixed or not
    mixed = False
    if len(set(bound_index)) != 1:
        
        mixed = True
    
    # grid points, intma, face points ...
    coord, intma, bsido,face = grid_2D(Np, Ne, Nbound, Nelx, Nely, N, Q, xgl, ax, bx,bound_index)

    # metrics terms
    ksi_x,ksi_y,eta_x,eta_y,jac = metrics(coord,intma,l_basis,dl_basis,wnq,Ne,N,Q)
    
    iside,psideh = create_side(intma,face,Np,Ne,Nbound,Nside,N)
    
    psideh, imapl, imapr = create_face(iside,intma,Nside,N)
    
    # Compute the normal vectors and face jacobian
    nx,ny,jac_side = compute_normals(psideh,intma,coord,Nside,N,Q,wnq,l_basis,dl_basis)

    # Compute the mass matrix
    Mmatrix = create_Mmatrix(jac,intma,l_basis,Np,Ne,N,Q)
    
    # Compute the Laplacian matrix
    Lmatrix = create_Lmatrix(intma,jac,ksi_x,ksi_y,eta_x,eta_y,l_basis,dl_basis,Np,Ne,N,Q)
      
    # Apply Dirichlet boundary condition on the matrices
    Lmatrix = apply_Dirichlet_BC_matrix(Lmatrix,intma,imapl,psideh,Np,N,Q,Nside)
    Mmatrix = apply_Dirichlet_BC_matrix(Mmatrix,intma,imapl,psideh,Np,N,Q,Nside)
    
    print("N = {:d}, nel = {:d}, Np = {}".format(N,Ne,Np))
             
    
    #Mmatrix_inv = csr_matrix(Mmatrix_inv)
    
    fe,qe,gradq = exact_solution(coord,Np,icase)
    
    Rhs = Mmatrix@fe
    
    # Apply boundary conditions
    
    Rhs =  apply_Dirichlet_BC_vec1(Rhs,qe,intma,jac_side,imapl,psideh,Np,N,Q,Ne,Nelx,Nside,mixed)
    
    Rhs = apply_Neumann_BC(Rhs,nx,ny,intma,jac_side,l_basis,imapl,psideh,gradq,Np,N,Q,Nside)
    
    # Do this for Neumann condition in all directions because Lmatrix is singular
    if (bound_index == 4).all():
        
        Lmatrix[0,:] = 0
        Lmatrix[0,0] = 1
        Rhs[0] = qe[0]
        
    # Solve for the numerical soultion
    qp = linalg.solve(Lmatrix,Rhs)
    
    #bc = extract_ice_bc(qp,intma,imapl,psideh,Np,N,Q,Nside)
            
    
    return qe, qp, coord, intma

