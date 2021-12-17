from matplotlib.pylab import*

from scipy.interpolate import griddata
from matplotlib import cm
import warnings
warnings.filterwarnings('ignore')

from Inputs import*
    
order = array([2,3,4])        # polynomial order
Nv = array([8,16,24,32])
kstages = 3
cfl = 0.25
dt = 1e-2
Tfinal = 0.1
iplot = False            # plot the solution
iconverg = True

time_method = "BDF2"      # IRK, BDF2 or BDF3
integration_type = 1      # % = 1 is inexact and = 2 is exact
icase = 2                 # select icase: 1,2,3,4

alpha = 1              
beta = -1                  # Dirichlet: alpha = 0, beta = 1
                               # Neumann: alpha = 1, beta = 0
                               # Robin: alpha = 1, beta != 0

x_boundary = [4,4]    # Bottom and Top (x = -1 and x = +1)
y_boundary = [4,4]    # Left and Right (y = -1 and x = +1)


Visualisation2(order,Nv,time_method,kstages,integration_type,icase,Tfinal,alpha,beta,x_boundary,y_boundary)