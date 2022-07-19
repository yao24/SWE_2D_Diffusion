from matplotlib.pylab import*                                         

from scipy.interpolate import griddata
from matplotlib import cm
from time import perf_counter

from module_2D_ice_conditions import*

#Computes the boundary values, T_b, S_b, V(melt rate)

def TempB(Sb, a,b,c,pb): 
    '''
    Sb: interface salinity
    pb: interface pressure
    a,b,c: constants
    TB = aSb + b + cpb 
    '''
    return a*Sb + b + c*pb

def Meltrate(Sw, Sb, gammaS): 
    '''
    Sw: ambient salinity
    Sb: interface salinity
    gammaS: salinity exchange velocity
    Melt rate: V = gammaS*(Sw - Sb)/Sb 
    '''
    return gammaS*(Sw - Sb)/Sb

def SaltB(K,L,M,Sw): 
    '''
    Interface salinity 
    '''
    D = L**2 - 4*K*M*Sw
    Sb1 = (-L + sqrt(D))/(2*K) 
    Sb2 = (-L - sqrt(D))/(2*K)
    
    
    # Check for possitive salinity
    if(Sb1 > 0):
        Sb = Sb1
    elif(Sb2 > 0 and Sb2 > Sb1):
        Sb = Sb2

    return Sb


# Coefficient in the quadratic equation for salinity at the boundary
# def coefK(a, gammaS, gammaT,cw,cI):
#     return a*(1 - gammaS*cI/(gammaT*cw))

# def coefF(Tw, gammaS,gammaT, cw, Li, cI, TS, b, c, pb, a, Sw):
#     A = -Tw - (gammaS/(cw*gammaT))*(Li - cI*TS) 
#     B = (b + c*pb)*(1 - gammaS*cI/(cw*gammaT))
#     C = a*(gammaS*cI/(cw*gammaT))*Sw
#     return A + B + C

# def coefM(gammaS, gammaT, cw, Li, cI, TS, a, b, c, pb):
#     A = (gammaS/(cw*gammaT))*(Li - cI*TS) 
#     B = cI*(gammaS/(cw*gammaT))*(b + c*pb)
#     return A + B

def coefK(a, gammaS, gammaT,cw,cI):
    return a*(gammaT*cw - gammaS*cI)

def coefF(Tw, gammaS,gammaT, cw, Li, cI, TS, b, c, pb, a, Sw):
    A = -Tw*gammaT*cw - (gammaS)*(Li - cI*TS) 
    B = (b + c*pb)*(gammaT*cw - gammaS*cI)
    C = a*(gammaS*cI)*Sw
    return A + B + C

def coefM(gammaS, gammaT, cw, Li, cI, TS, a, b, c, pb):
    A = (gammaS)*(Li - cI*TS) 
    B = cI*(gammaS)*(b + c*pb)
    return A + B


# define exact, source functions
def initial_Temp(x, cst1):
    return cst1 #+ 9.5e-4*x

def initial_Salt(x, cst2):
    return cst2 #+ 4.0e-4*x
    
# Neumann boundary condition
def hT(bcst1, gammaT, TB,T,V):
    return rhoI*bcst1*V/rhoW 

def hB(bcst2,SB, S, gammaS,V):
    return rhoI*bcst2*SB*V/rhoW



# Values of the parameters in the ice-ocean simulation

a = -5.73e-2     # Salinity coefficient of freezing equation(˚C*psu^{-1})
b =  9.39e-2     # Constant coefficient of freezing equation(˚C)
c = -7.53e-8     # Pressure coefficient of freezing equation(˚C*Pa^{-1})
ci = 2009.0      # Specific heat capacity ice(J*kg^-1*K^-1)
cw = 3974.0      # Specific heat capacity water(J*kg^-1*K^-1)
Li = 3.35e+5     # Latent heat fusion(J*kg^-1)
Tw = 2.3         # Temperature of water(˚C)
Ti = -25         # Temperature of ice(˚C)
Sw = 35          # Salinity of water(psu)
Sc = 2500        # Schmidt number
Pr = 14          # Prandtl number
mu = 1.95e-6     # Kinematic viscosity of sea water(m^2*s^-1)
pb = 1.0e+7      # Pressure at ice interface(Pa)
#kT = mu/Pr       # Thermal diffusivity(m^2*s^-1)
#kS = mu/Sc       # Salinity diffusivity(m^2*s^-1)

kT = 1.3e-7      # 1.3e-7
kS = 1.8e-7      # 7.4e-10
rhoI = 920       # density of ice(kg m^-3)
rhoW = 1025      # density of sea water(kg*m^-3)

gammaT = 5.0e-5        # Thermal exchange velocity(m*s^-1)

gammaS = 0.04*gammaT   # Salinity exchange velocity(m*s^-1)

rho = rhoI/rhoW        # report between ice density and seawater density

# Coeffecients in quadratic equation for salinity at the boundary

Kf = coefK(a, gammaS, gammaT,cw,ci)
Mf = coefM(gammaS, gammaT, cw, Li, ci, Ti, a, b, c, pb)


def ice_simulation(N,Q,nel,Np, ax, bx,ay,by, integration_type,\
                                kstages, CFL,Tw,Tfinal,Nelx, Nely,\
                                Nx, Ny, Nbound,Nside,x_boundary,y_boundary,mixed,mxit,stol,solver_type):
    # For temperature
    cst1 = Tw               # Constant that initialize the initial temperature
    c1 = kT                 # Temperature diffusivity
    bcst1 = Li/(cw)#*kT)      # Constant term for the temperature gradient at the boundary
    # For salinity
    cst2 = Sw               # Constant that initialize the initial salinity
    c2 = kS                 # Salinity diffusivity
    bcst2 = 1#/kS            # Constant term for the salinity gradient at the boundary

    # Call the ice-ocean solver for the diffusion problem

    '''
    outputs:
    --------
    S          : Salinity
    T          : Temperature
    coord      : All grid points
    intma      : Intma(CG/DG)
    '''
    if(solver_type == 1):
        S, T, coord,tf,bc_ice,temp_profile = ice_ocean_Solver(N,Q,nel, Np, ax, bx,ay,by, integration_type, hT, hB, cst1,\
                            c1, bcst1,cst2, c2,bcst2,Tw, gammaS,gammaT, cw, Li, ci, Ti, b, c, pb, a,\
                            Sw, Kf, Mf, CFL, Tfinal,coefF, Meltrate, SaltB, Nelx, Nely,\
                            Nx, Ny, Nbound,Nside,kstages,x_boundary,y_boundary,mixed)
    if(solver_type == 2):
        S, T, coord,tf,bc_ice,temp_profile = ice_ocean_Solver_gmres(N,Q,nel, Np, ax, bx,ay,by, integration_type,\
                            hT, hB, cst1,c1, bcst1,cst2, c2,bcst2,Tw, gammaS,gammaT, cw, Li, ci, Ti, b, c,
                            pb, a,Sw, Kf, Mf, CFL, Tfinal,coefF, Meltrate, SaltB, Nelx, Nely,\
                            Nx, Ny, Nbound,Nside,kstages,x_boundary,y_boundary,mixed,mxit,stol)
    
    return S,T,coord,tf,bc_ice,temp_profile



def simulation2(cfl,N,Nv,kstages,integration_type,Tw,Tfinal,\
               x_boundary,y_boundary,ax,bx,ay,by, solver_type):
    
    #cfl = 0.001#1/(N+1)       # cfl number
    
    mixed = True
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

        mxit = 200
        
        # Call the ice-ocean solver

        '''
        outputs:
        --------
        S          : Salinity
        T          : Temperature
        coord      : All grid points
        intma      : Intma(CG/DG)
        '''

        S,T,coord,tf,bc_ice,temp_profile = ice_simulation(N,Q,Ne,Np, ax, bx,ay,by, integration_type,\
                                    kstages, cfl,Tw,Tfinal,Nelx, Nely,\
                                    Nx, Ny, Nbound,Nside,x_boundary,y_boundary,mixed,mxit,stol,solver_type)

        print("\twalltime = {:e}".format(tf))

        toc = perf_counter()
   
    return S,T,coord,tf,bc_ice,temp_profile


def figures(coord, S,T):

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
    
    T_2d = griddata((xe,ye),T,(xi,yi), method='cubic')

    fig = figure(1, figsize=(9,4))
    fig.suptitle("Temperature", fontsize=14)
    fx = fig.add_subplot(121, projection='3d')
    d3_surf = fx.plot_surface(xi,yi,T_2d,rstride = 1, cmap=cm.coolwarm,cstride = 1,antialiased=False)
    fig.colorbar(d3_surf, pad=0.1,shrink=0.75)

    #title("Temperature")
    xlabel("x")
    ylabel("y")
    
    fig.add_subplot(122)
    surf = imshow(T_2d, extent=[xmin, xmax, ymin, ymax],origin='lower',cmap=cm.coolwarm) # ocean, Blues, terrain, cm.coolwarm
    colorbar(surf,shrink=0.75)
    #clim(T.min(), T.max())
    #title("Temperature")
    xlabel("x")
    ylabel("y")
    
    fig.tight_layout()
    #subplot_tool()
    show()
    
    print("min temperature  = ",T.min())

    print("\nmax temperature  = ",T.max())

    
    S_2d = griddata((xe,ye),S,(xi,yi), method='cubic')

    fig = figure(2, figsize=(9,4))
    fig.suptitle("Salinity", fontsize=14)
    fx = fig.add_subplot(121, projection='3d')
    d3_surf = fx.plot_surface(xi,yi,S_2d,rstride = 1, cmap=cm.coolwarm,cstride = 1,antialiased=False)
    fig.colorbar(d3_surf, pad=0.1,shrink=0.75)

    #title("Temperature")
    xlabel("x")
    ylabel("y")
    
    fig.add_subplot(122)
    surf = imshow(S_2d, extent=[xmin, xmax, ymin, ymax],origin='lower',cmap=cm.coolwarm) # ocean, Blues, terrain, cm.coolwarm
    colorbar(surf,shrink=0.75)
    #clim(T.min(), T.max())
    #title("Temperature")
    xlabel("x")
    ylabel("y")
    
    fig.tight_layout()
    #subplot_tool()
    show()
    
    
    
def profile(data, temp_profile):
    
    figure(3)
    plot(data['Points:0'],data['theta'],label = 'NUMO')
    plot(temp_profile[0],temp_profile[1],'--',label = 'Python')
    #ylim([bc.min(), bc.max()])
    title('Temperature profile')
    xlabel('x')
    ylabel('T')
    legend()
    show()