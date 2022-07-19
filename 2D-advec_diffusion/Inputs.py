
from matplotlib.pylab import*                                         

from scipy.interpolate import griddata
from matplotlib import cm
import pandas as pd
from time import perf_counter
#from sympy import*

from Module_2D import*


# Domain of integration, look into func_2D module to add your own problem or initial condition

def domain(icase):
    
    
    if(icase == 1):
        ax = -1 ; bx = 1 ; coeff = 1.0
    elif(icase == 2):
        ax = -1 ; bx = 1 ; coeff = 1.0
    
    
    print("====================================")
    print("Problem: Diffusion")
    print("Domain: [{}, {}]".format(ax,bx)) 
    print("Diffusivity: {}".format(coeff)) 
    #print("====================================\n")
    
    return ax, bx, coeff

# Boundary conditions

def bc_type(x_boundary,y_boundary):
    
    bound = array([x_boundary,y_boundary])
    
    sbound = ["Dirichlet", "Neumann", "Robin"]
    nbound = [5,4,8]
    
    st = ""
    
    for i in range(3):
        
        if(nbound[i] in bound):
            st += sbound[i] + " & "
            
    print("Boundary conditions: ",st)
    
    print("==========================================\n")
       
        
def simulation(N,Nv,time_method,kstages,integration_type,icase,Tfinal,alpha,beta,\
               x_boundary,y_boundary,ax,bx,c,cfl):
    
    dt_data = dtype([('N',int),('M',int),('cfl','d'),('2-norm','d')])
    

    if (integration_type == 1):
        Q = N
    elif (integration_type == 2):
        Q = N+1

    wall = 0
    

    for e, nel in enumerate(Nv):
            
        Nelx = nel; Nely = nel
        Nx = Nelx*N+1
        Ny = Nely*N+1
        Np = Nx*Ny
        Ne = Nelx*Nely
        Nbound = 2*Nx + 2*(Ny-2)
        Nside = 2*Ne + Nelx + Nely
        
        tic = perf_counter()
        
    
        qe, q,coord,intma,ntime,tf = AdvectionDiffusion_solver(N,Q,Ne, Np, ax, bx, Nelx, Nely, Nx, Ny, Nbound,Nside,\
                                             icase,Tfinal,c,cfl,kstages,time_method,alpha,beta,\
                                             x_boundary,y_boundary)
        
        top = sum((q - qe)**2)
        bot = sum(qe**2)

        e2 = sqrt(top/bot)
        
        print("\tl2_norm = {:.4e}".format(e2))
        print("\twalltime: {}".format(tf))
        
    return qe, q,coord

def Visualisation(order,Nv,time_method,kstages,integration_type,icase,Tfinal,alpha,beta,x_boundary,y_boundary):
    
    # Domain and diffusion coefficient
    
    #ax,bx,c = domain(icase)
    ax,bx,c = domain(icase)
    #bc_type(alpha,beta)
    bc_type(x_boundary,y_boundary)
    
    for iN,N in enumerate(order):
        
        qe,q,coord = simulation(N,Nv,time_method,kstages,integration_type,icase,Tfinal,alpha,beta,\
               x_boundary,y_boundary,ax,bx,c,cfl)
    
    return qe,q,coord
    
    
    
    
def figures(coord, qe,q):

    xmin = min(coord[:,0])
    xmax = max(coord[:,0])
    ymin = min(coord[:,1])
    ymax = max(coord[:,1])
    xe = coord[:,0]
    ye = coord[:,1]
    nx = 200
    ny = 200
    dx = (xmax-xmin)/nx
    dy = (ymax-ymin)/ny
    x1 = arange(xmin,xmax+dx,dx)
    y1 = arange(ymin,ymax+dy,dy)
    xi,yi = meshgrid(x1,y1)
    
    q_2d = griddata((xe,ye),q,(xi,yi), method='cubic')

    fig = figure(1, figsize=(9,4))
    fig.suptitle("Numerical", fontsize=14)
    fx = fig.add_subplot(121, projection='3d')
    d3_surf = fx.plot_surface(xi,yi,q_2d,rstride = 1, cmap=cm.coolwarm,cstride = 1,antialiased=False)
    fig.colorbar(d3_surf, pad=0.1,shrink=0.75)

    xlabel("x")
    ylabel("y")
    
    fig.add_subplot(122)
    surf = imshow(q_2d, extent=[xmin, xmax, ymin, ymax],origin='lower',cmap=cm.coolwarm) # ocean, Blues, terrain, cm.coolwarm
    colorbar(surf,shrink=0.75)
    #clim(q.min(), q.max())
    xlabel("x")
    ylabel("y")
    
    fig.tight_layout()
    #subplot_tool()
    show()
    

    
    qe_2d = griddata((xe,ye),qe,(xi,yi), method='cubic')

    fig = figure(2, figsize=(9,4))
    fig.suptitle("Exact solution", fontsize=14)
    fx = fig.add_subplot(121, projection='3d')
    d3_surf = fx.plot_surface(xi,yi,qe_2d,rstride = 1, cmap=cm.coolwarm,cstride = 1,antialiased=False)
    fig.colorbar(d3_surf, pad=0.1,shrink=0.75)

    xlabel("x")
    ylabel("y")
    
    fig.add_subplot(122)
    surf = imshow(qe_2d, extent=[xmin, xmax, ymin, ymax],origin='lower',cmap=cm.coolwarm) # ocean, Blues, terrain, cm.coolwarm
    colorbar(surf,shrink=0.75)
    #clim(qe.min(), qe.max())
    xlabel("x")
    ylabel("y")
    
    fig.tight_layout()
    #subplot_tool()
    show()
        