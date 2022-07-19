%---------------------------------------------------------------------%
% This code solves the 2D Diffusion Equation using Unified CG methods
% with tensor product of 1D basis function with either
% Inexact Integration and Using an NPOIN based data-structure
% Written by Yao Gahounzo 29/01/2022
%            Computing PhD
%            Boise State University 
%---------------------------------------------------------------------%
clear all; 
close all;

tic

%Input Data
nel=16; %Number of Elements
nop=2;    %Interpolation Order

plot_solution = 1;
integration_type = 1; %=1 is inexact and =2 is exact
space_method = 'CG'; 
kstages = 3;
icase = 1; % 1 or 2
           
Tfinal = 0.01;           
time_method = 3;  % 1 = IRK, 2 = BDF2, 3 = BDF3

% Boundary type: 5 = dirichlet or 4 = neumann

bound_index = [4,4,4,4]; % [bottom,right,top,left] boundaries
           
ax = 0;
bx = 2*pi;

c = 1;

nelx = nel;
nely = nel;
nelem = nelx*nely; %Number of Elements
ngl = nop + 1;
npts = ngl*ngl;
npoin = (nop*nelx + 1)*(nop*nely + 1);
nboun = 2*nelx + 2*nely;
nside = 2*nelem + nelx + nely;
Nx = nop*nelx + 1;
cfl = 1/(npoin);

%Compute LGL Points
[xgl,wgl] = legendre_gauss_lobatto(ngl);

if (integration_type == 1)
    noq = nop;
    integration_text = ['Inexact'];
elseif (integration_type == 2)
    noq = nop+1;
    integration_text = ['Exact'];
end
nq = noq + 1;
main_text = [space_method ':'  integration_text];

%Compute Legendre Cardinal functions and derivatives
[psi,dpsi,xnq,wnq] = lagrange_basis(ngl,nq,xgl);

%Create CG-Storage Grid
[coord,intma,bsido] = create_grid(npoin,nelem,nboun,...
                                nelx,nely,ngl,xgl,ax,bx,bound_index);
                            
                       
%Compute Metric Terms
[ksi_x,ksi_y,eta_x,eta_y,jac] = metrics(coord,intma,psi,dpsi,wnq,nelem,ngl,nq);

%Compute Side/Edge Information
[iside,jeside] = create_side(intma,bsido,npoin,nelem,nboun,nside,ngl);
[psideh,imapl,imapr] = create_side_dg(iside,intma,nside,nelem,ngl);
[nx,ny,jac_side] = compute_normals(psideh,intma,coord,...
                   nside,ngl,nq,wnq,psi,dpsi);
       

%Create Mmatrix and Lmatrix  

Mmatrix = create_Mmatrix(jac,intma,psi,npoin,nelem,ngl,nq);

Lmatrix = create_Lmatrix(intma,jac,ksi_x,ksi_y,eta_x,...         
                   eta_y,psi,dpsi,npoin,nelem,ngl,nq);
               

% Apply boundary conditions

Mmatrix = apply_Dirichlet_BC_matrix(Mmatrix,psideh,...
               nside,ngl,imapl,intma);

DFmatrix = c*Lmatrix;

%dx = (bx-ax)/Nx;
dx = coord(2,1) - coord(1,1);
dt_est = cfl*dx^2;
ntime = floor(Tfinal/dt_est);
dt = Tfinal/ntime;

% Initial solution
time = 0;
[qe,gradq] = exact_solution(coord,npoin,icase,time);


[alpha,beta,stages] = IRK_coefficients(kstages);

Amatrix = Mmatrix - dt*alpha(stages,stages)*DFmatrix;

if(time_method == 1)
    
    time_integration = 'IRK';
    [qp,qe,time] = IRK_integration(qe,DFmatrix,Amatrix,Mmatrix,coord,npoin,icase,dt,...
              jac_side,psideh,nside,ngl,imapl,intma,nx,ny,alpha,beta,stages,time,ntime);
          
elseif(time_method == 2)
        
    time_integration = 'BDF2';
    [qp,qe] = BDF2_integration(qe,DFmatrix,Amatrix,Mmatrix,coord,npoin,icase,dt,...
              jac_side,psideh,nside,ngl,imapl,intma,nx,ny,alpha,beta,stages,time,ntime);
          
elseif(time_method == 3)
        
    time_integration = 'BDF3';
    [qp,qe] = BDF3_integration(qe,DFmatrix,Amatrix,Mmatrix,coord,npoin,icase,dt,...
              jac_side,psideh,nside,ngl,imapl,intma,nx,ny,alpha,beta,stages,time,ntime);
          
end

%Compute Norm

error = abs(qp-qe);
l1_norm = sum(error);
l2_norm = sqrt(sum(error.^2)/sum(qe.^2));
inf_norm = max(error);


%Plot Solution
if (plot_solution == 1)
    h = figure;
    figure(h);
    xmin = min(coord(:,1)); xmax=max(coord(:,1));
    ymin = min(coord(:,2)); ymax=max(coord(:,2));
    xe = coord(:,1);
    ye = coord(:,2);
    nxx=200; nyy=200;
    dx=(xmax-xmin)/nxx;
    dy=(ymax-ymin)/nyy;
    [xi,yi]=meshgrid(xmin:dx:xmax,ymin:dy:ymax);
    qi=griddata(xe,ye,qp,xi,yi,'cubic');
    %[cl,h]=contourf(xi,yi,qi);
    surf(xi,yi,qi);
    colorbar('SouthOutside');
    xlabel('X','FontSize',18);
    ylabel('Y','FontSize',18);
    axis image
    title_text=[time_integration ', Ne = ' num2str(nelem) ', N = ' num2str(nop) ', Q = ' num2str(noq) ', L2 Norm = ' num2str(l2_norm)];
    title([title_text],'FontSize',18);
    set(gca, 'FontSize', 18);

    disp(['space_method = ',space_method]);
    disp(['nop = ',num2str(nop),'  nelem = ',num2str(nelem) ]);
    q_max=max(qp);
    q_min=min(qp);
    disp(['L2_Norm = ',num2str(l2_norm),' q_max = ',num2str(q_max),'  q_min = ',num2str(q_min) ]);
    disp(['npoin = ',num2str(npoin)]);
end

toc
