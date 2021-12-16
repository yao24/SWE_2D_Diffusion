
from matplotlib.pylab import*                                         

from scipy.interpolate import griddata
from matplotlib import cm
import pandas as pd
from time import perf_counter
#from sympy import*

#from func2 import*

from module_2D_diffusion_bdf2_3 import*


# Domain of integration, look into func_2D module to add your own problem or initial condition

def domain(icase):
    
    
    if(icase == 1):
        ax = -1 ; bx = 1 ; coeff = 1.0
    elif(icase == 2):
        ax = 0 ; bx = 2*pi ; coeff = 1.0
    elif(icase == 3):
        ax = -1 ; bx = 1 ; coeff = 1.0
    elif(icase == 4):
        ax = -1 ; bx = 1 ; coeff = 0.01
    elif(icase == 5):
        ax = 0 ; bx = 3 ; coeff = 1/25
    
    print("====================================")
    print("Problem: Diffusion")
    print("Domain: [{}, {}]".format(ax,bx)) 
    print("Diffusivity: {}".format(coeff)) 
    #print("====================================\n")
    
    return ax, bx, coeff

# Boundary conditions
def bc_type(alpha,beta):
    
    if(alpha == 0 and beta == 1):
        print("Boundary conditions: Dirichlet")
    elif(alpha == 1 and beta == 0):
        print("Boundary conditions: Neumann")
    elif(alpha == 1 and beta != 0):
        print("Boundary conditions: Robin")
    print("====================================\n") 
    
dt_data = dtype([('N',int),('M',int),('cfl','d'),('2-norm','d')]) 
    
    
def Visualisation(order,Nv,time_method,kstages,integration_type,icase,Tfinal,alpha,beta,x_boundary,y_boundary,\
                 iplot,iconverg):
    
    dt_data = dtype([('N',int),('M',int),('cfl','d'),('2-norm','d')])
    # Domain and diffusion coefficient

    ax,bx,c = domain(icase)
    bc_type(alpha,beta)

    len_el = len(Nv)
    len_pol = len(order)
    l2e_norm = zeros((len_pol, len_el))
    max_norm = zeros((len_pol, len_el))
    
    output_file = 'output_loc.dat'
    file = open(output_file,"wb")
    file.write(array([len(Nv)]))

    for iN,N in enumerate(order):
        
        file.write(array((N,integration_type)))

        cfl = 0.2#1/(N+1)       # cfl number

        N = order[iN]
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


            qe, q,coord,intma,ntime = diffusion_solver(N,Q,Ne, Np, ax, bx, Nelx, Nely, Nx, Ny, Nbound,Nside,\
                                                 icase,Tfinal,c,cfl,kstages,time_method,alpha,beta,\
                                                 x_boundary,y_boundary)

            top = 0
            bot = 0

            for i in range(Np):
                top += (q[i] - qe[i])**2
                bot += qe[i]**2

            e2 = sqrt(top/bot)
            print(e2)
            l2e_norm[iN,e] = e2
            
            t = array((nel,ntime,cfl,e2),dtype=dt_data)
            file.write(t)
    
    file.close()
    print('Done')
    
    if(iplot):
        
        
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

        fig = figure(1)
        fx = fig.add_subplot(111, projection='3d')
        surf = fx.plot_surface(xi,yi,q_2d,rstride = 1, cstride = 1,antialiased=False)
        fig.colorbar(surf)

        title("Numerical solution")
        xlabel("x")
        ylabel("y")
        
        figure(2)
        imshow(q_2d, extent=[ax, bx, ax, bx],origin='lower',cmap=cm.coolwarm)
        colorbar()
        clim(q.min(), q.max())
        title("Numerical solution")
        xlabel("x")
        ylabel("y")
        show()

    if(iconverg):
        
        import cg_graphics           # import cg_graphics module
        figure(3)
        clf()

        for i,N in enumerate(order):

            if(N >= 3):
                p = polyfit(log(Nv[:2]), log(l2e_norm[i][:2]), 1)
            else:

                p = polyfit(log(Nv), log(l2e_norm[i]), 1)

            loglog(Nv, l2e_norm[i], '-o',markersize=5, label = 'N = {:d}: rate = {:.2f}'.format(N,p[0]))

            loglog(Nv, exp(polyval(p,log(Nv))), '--')

        cg_graphics.set_xticks(Nv)
        xlabel('# Elements')
        ylabel('Error (L2-error)')
        title('Error vs number of Elements ({:s}, {:s})'.format('cg'.upper(), time_method))
        grid(axis='both',linestyle='--')
        legend()
        show()
        
        
        
        
def simulation(file,N,Nv,time_method,kstages,integration_type,icase,Tfinal,alpha,beta,\
               x_boundary,y_boundary,ax,bx,c):
    
    dt_data = dtype([('N',int),('M',int),('cfl','d'),('2-norm','d')])
    
    cfl = 1/(N+1)       # cfl number

    #N = order[iN]
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


        qe, q,coord,intma,ntime, tf = diffusion_solver(N,Q,Ne, Np, ax, bx, Nelx, Nely, Nx, Ny, Nbound,Nside,\
                                             icase,Tfinal,c,cfl,kstages,time_method,alpha,beta,\
                                             x_boundary,y_boundary)
        
        print("\twalltime: {}".format(tf))
        
        #Compute Norm

        top = sum((q - qe)**2)
        bot = sum(qe**2)

        e2 = sqrt(top/bot)
        
        t = array((nel,ntime,cfl,e2),dtype=dt_data)
        file.write(t)


def Visualisation2(order,Nv,time_method,kstages,integration_type,icase,Tfinal,alpha,beta,x_boundary,y_boundary):
    
    # Domain and diffusion coefficient
    
    ax,bx,c = domain(icase)
    bc_type(alpha,beta)
    
    output_file = 'output.dat'
    file = open(output_file,"wb")
    file.write(array([len(Nv)]))
    for iN,N in enumerate(order):
        
        file.write(array((N,integration_type)))
        simulation(file,N,Nv,time_method,kstages,integration_type,icase,Tfinal,alpha,beta,x_boundary,y_boundary,ax,bx,c)
    
    file.close()
    
    print("\nEnd simulation")
        