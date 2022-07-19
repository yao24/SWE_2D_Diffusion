
from matplotlib.pylab import*                                         

from scipy.interpolate import griddata
from matplotlib import cm
#import pandas as pd
from time import perf_counter

from module_diffusion_2D import*
#from module_2D_gmres import*


# Domain of integration, go to module_diffusion_2D.py to add your own problem or initial condition

def domain(icase):
    
    
    if(icase == 1):
        ax = -1 ; bx = 1 ; coeff = 1.0
    elif(icase == 2):
        #ax = 0 ; bx = 2*pi ; coeff = 1.0
        ax = -3*pi/4 ; bx = 3*pi/4 ; coeff = 1
        ay = ax ; by = bx
    elif(icase == 3):
        ax = -1 ; bx = 1 ; coeff = 1.0
        ay = ax ; by = bx
    elif(icase == 4):
        ax = -1 ; bx = 1 ; coeff = 0.01
        ay = ax ; by = bx
    elif(icase == 5):
        ax = 0 ; bx = 3 ; coeff = 1/25
        ay = ax ; by = bx
    
    print("====================================")
    print("Problem: Diffusion")
    print("Domain: [{}, {}]".format(ax,bx)) 
    print("Diffusivity: {}".format(coeff)) 
    #print("====================================\n")
    
    return ax, bx,ay,by, coeff

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
       
        
def simulation(file,N,Nv,time_method,kstages,integration_type,icase,Tfinal,alpha,beta,\
               x_boundary,y_boundary,ax,bx,ay,by,c, solver_type):
    
    dt_data = dtype([('N',int),('M',int),('cfl','d'),('2-norm','d')])
    
    cfl = 0.001#1/(N+1)       # cfl number
    
    mixed = False
    mxit = 30
    stol = 1e-12

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
        
        if(solver_type == 1):

            qe, q,coord,intma,ntime, tf = diffusion_solver(N,Q,Ne, Np, ax, bx, ax,bx,Nelx, Nely, Nx, Ny, Nbound,Nside,\
                                             icase,Tfinal,c,cfl,kstages,time_method,alpha,beta,\
                                             x_boundary,y_boundary,mixed)
        if(solver_type == 2):
            qe, q,coord,intma,ntime,tf = diffusion_solver_gmres(N,Q,Ne, Np, ax, bx,ay, by, Nelx, Nely, Nx, Ny, Nbound,Nside,\
                                             icase,Tfinal,c,cfl,kstages,time_method,alpha,beta,\
                                             x_boundary,y_boundary,mixed,mxit,stol)
        
        #Compute Norm

        top = sum((q - qe)**2)
        bot = sum(qe**2)

        e2 = sqrt(top/bot)
        
        print("\tl2_norm: {}".format(e2))
        print("\twalltime: {}".format(tf))
        
        t = array((nel,ntime,cfl,e2),dtype=dt_data)
        file.write(t)


def Visualisation2(order,Nv,time_method,kstages,integration_type,icase,Tfinal,alpha,beta,x_boundary,y_boundary,solver_type):
    
    # Domain and diffusion coefficient
    
    #ax,bx,c = domain(icase)
    ax,bx,ay, by,c = domain(icase)
    #bc_type(alpha,beta)
    bc_type(x_boundary,y_boundary)
    
    output_file = 'output.dat'
    file = open(output_file,"wb")
    file.write(array([len(Nv)]))
    for iN,N in enumerate(order):
        
        file.write(array((N,integration_type)))
        simulation(file,N,Nv,time_method,kstages,integration_type,icase,Tfinal,alpha,beta,x_boundary,\
                   y_boundary,ax,bx,ay,by,c,solver_type)
    
    file.close()
    
    print("\nEnd simulation")
    
    
def simulation2(cfl,N,Nv,time_method,kstages,integration_type,icase,Tfinal,alpha,beta,\
               x_boundary,y_boundary,ax,bx,ay,by,c, solver_type):
    
    #cfl = 0.001#1/(N+1)       # cfl number
    
    mixed = False
    mxit = 30
    stol = 1e-12

    if (integration_type == 1):
        Q = N
    elif (integration_type == 2):
        Q = N+1

    wall = 0
    l2e_norm = zeros(len(Nv))

    for e, nel in enumerate(Nv):

        Nelx = nel; Nely = nel
        Nx = Nelx*N+1
        Ny = Nely*N+1
        Np = Nx*Ny
        Ne = Nelx*Nely
        Nbound = 2*Nx + 2*(Ny-2)
        Nside = 2*Ne + Nelx + Nely

        tic = perf_counter()
        
        if(solver_type == 1):

            qe, q,coord,intma,ntime, tf = diffusion_solver(N,Q,Ne, Np, ax, bx, ax,bx,Nelx, Nely, Nx, Ny, Nbound,Nside,\
                                             icase,Tfinal,c,cfl,kstages,time_method,alpha,beta,\
                                             x_boundary,y_boundary,mixed)
        if(solver_type == 2):
            qe, q,coord,intma,ntime,tf = diffusion_solver_gmres(N,Q,Ne, Np, ax, bx,ay, by, Nelx, Nely, Nx, Ny, Nbound,Nside,\
                                             icase,Tfinal,c,cfl,kstages,time_method,alpha,beta,\
                                             x_boundary,y_boundary,mixed,mxit,stol)
        
        #Compute Norm

        top = sum((q - qe)**2)
        bot = sum(qe**2)

        e2 = sqrt(top/bot)
        
        print("\tl2_norm: {}".format(e2))
        print("\twalltime: {}".format(tf))
        
        l2e_norm[e] = e2
   
    return qe, q, coord, l2e_norm 


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
    
    # Numerical solution
    
    q_2d = griddata((xe,ye),q,(xi,yi), method='cubic')
    
    fig = figure(1)
    fx = fig.add_subplot( projection='3d')
    surf = fx.plot_surface(xi,yi,q_2d,rstride = 1, cmap=cm.coolwarm,cstride = 1,antialiased=False)
    fig.colorbar(surf,anchor=(0, 0.3), shrink=0.65)

    #title("Numerical solution")
    xlabel("x")
    ylabel("y")

    show()
    
    # Exact solution
    qe_2d = griddata((xe,ye),qe,(xi,yi), method='cubic')

    fig = figure(3)
    fx = fig.add_subplot(111, projection='3d')
    surf = fx.plot_surface(xi,yi,qe_2d,rstride = 1, cmap=cm.coolwarm,cstride = 1,antialiased=False)
    fig.colorbar(surf,anchor=(0, 0.3), shrink=0.7)
    title("Exact solution")
    xlabel("x")
    ylabel("y")
    show()
    
    # Numerical solution 2D
    figure(4)
    imshow(q_2d, extent=[xmin, xmax, ymin, ymax],origin='lower',cmap=cm.coolwarm)
    colorbar()
    clim(q.min(), q.max())
    title("Numerical solution")
    xlabel("x")
    ylabel("y")
    show()
    
    print("min_q  = ",q.min())
    print("min_qe = ",qe.min())

    print("\nmax_q  = ",q.max())
    print("max_qe = ",qe.max())
    
    
    
def convergences(l2_norm, Nv, order, time_method):
    
    import cg_graphics           # import cg_graphics module

    figure(5)
    clf()

    for i,N in enumerate(order):

        if(N >= 4):
            p = polyfit(log(Nv[:2]), log(l2_norm[i][:2]), 1)
        else:

            p = polyfit(log(Nv), log(l2_norm[i]), 1)

        loglog(Nv, l2_norm[i], '-o',markersize=5, label = 'N = {:d}: rate = {:.2f}'.format(N,p[0]))

        loglog(Nv, exp(polyval(p,log(Nv))), '--')

    cg_graphics.set_xticks(Nv)
    xlabel('# Elements')
    ylabel('Error (L2-error)')
    title('Error vs number of Elements ({:s}, {:s})'.format('cg'.upper(), time_method))
    grid(axis='both',linestyle='--')
    legend()
    #savefig('Rates/gmresN6{:d}.png'.format(N))
    show()
        