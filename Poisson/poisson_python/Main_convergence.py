from numpy import *
from scipy.interpolate import griddata
from matplotlib import cm
import matplotlib.pyplot as plt

from module_poisson_2D import*


# Domain of integration, look into func_2D module to add your own problem or initial condition

def domain(icase):
    
    
    if(icase == 1):
        ax = 0 ; bx = 1 
        ay = 0 ; by = 1 
    elif(icase == 2):
        ax = 0 ; bx = 1 
        ay = 0 ; by = 1 
    elif(icase == 3):
        ax = -1 ; bx = 1
        ay = -1 ; by = 1
    elif(icase == 4):
        ax = -1 ; bx = 1
        ay = -1 ; by = 1
    elif(icase == 5):
        ax = -1 ; bx = 1
        ay = -1 ; by = 1
        
    print("==========================================")
    print("Problem: Poisson")
    print("Domain: [{}, {}]".format(ax,bx)) 
    return ax, bx,ay, by


# Boundary conditions 
    
def bc_type(x_boundary,y_boundary):
    
    bound = array([x_boundary,y_boundary])
    
    sbound = ["Dirichlet", "Neumann"]
    nbound = [5,4]
    
    st = ""
    
    for i in range(2):
        
        if(nbound[i] in bound):
            st += sbound[i] + " & "
            
    print("Boundary conditions: ",st)
    
    print("==========================================\n") 
    
    
    
integration_type = 1      # % = 1 is inexact and = 2 is exact
iplot = False             # plot the solution
icase = 3                 # select icase: 1,2,3

# Boundary type: 4 = Neumann, 5 = Dirichlet     
x_boundary = [4,4]    # Bottom and Top (x = -1 and x = +1)
y_boundary = [4,4]    # Left and Right (y = -1 and x = +1)

# Domain and diffusion coefficient

ax,bx,ay, by = domain(icase)
bc_type(x_boundary,y_boundary)


order = array([1, 2, 3, 4, 5])        # polynomial order
Nv = array([8,16,24,32])

# Create a data type for storing results
dt_data = dtype([('N',int),('Np','int'),('1-norm','d'),('2-norm','d'),('inf-norm','d')])  

def poisson_simulation(file,iN, N,integration_type,Nv,x_boundary,y_boundary,ax,bx,ay, by):
    
    
    Q = N

    for e, nel in enumerate(Nv):
      
        
        Np = nel*N + 1

        Nelx = nel; Nely = nel
        Nx = Nelx*N+1
        Ny = Nely*N+1
        Np = Nx*Ny
        Ne = Nelx*Nely
        Nbound = 2*Nx + 2*(Ny-2)
        Nside = 2*Ne + Nelx + Nely
        
        tic = perf_counter()
        
    
        qe, q,coord,intma = poisson_solver(N,Q,Ne, Np, ax, bx,ay,by, Nelx, Nely, Nx, Ny, \
                                    Nbound,Nside,icase,x_boundary,y_boundary)
        
        
        error = abs(q-qe)
        top = sum(error**2)
        bot = sum(qe**2)

        
        
        #print("\tl2_norm = {:.4e}".format(e2))

        e1 = sum(error)
        e3 = max(error)
        e2 = sqrt(top/bot)

        t = array((nel,Np ,e1,e2,e3),dtype=dt_data)
        file.write(t)



# Poisson Simulation

output_file = 'Neumann_py/poisson_44.dat'

integType = [1]


# Simulation. Loop over integration type, and then loop over polynomial order. 
file = open(output_file,"wb")
file.write(array([len(Nv)]))

for integration_type in integType:

    for iN,N in enumerate(order):

        # Run over range of N = 8, 16, 32,64,128,256,...
        file.write(array((N,integration_type)))

        print("order = {:d}; Integration_type = {:d}".format(N,integration_type))
        poisson_simulation(file,iN, N,integration_type,Nv,x_boundary,y_boundary,ax,bx,ay, by)
        #print("")

file.close()
print("End")
