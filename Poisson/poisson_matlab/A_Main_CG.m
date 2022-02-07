%---------------------------------------------------------------------%
%This code solves the 2D Poisson Equation using Unified CG/DG methods
%with tensor product of 1D basis function with either
%Exact or Inexact Integration and Using an NPOIN based data-structure
%Written by F.X. Giraldo on 9/2014
%           Department of Applied Mathematics
%           Naval Postgraduate School 
%           Monterey, CA 93943-5216
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

bc_type = 4; % 5 = dirichlet or 4 = neumann

icase = 1; % 1 = 2D with homogeneous BCs in x and y; 
           % 2 = 1D with homogeneous BCs along x=-1/+1 and non-homogeneous along y=-1/+1 .
           % 3 = 2D with non-homogeneous BCs along x and y.
           
ax = -1;
bx = 1;

nelx = nel;
nely = nel;
nelem = nelx*nely; %Number of Elements
ngl = nop + 1;
npts = ngl*ngl;
npoin = (nop*nelx + 1)*(nop*nely + 1);
nboun = 2*nelx + 2*nely;
nside = 2*nelem + nelx + nely;

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
                                nelx,nely,ngl,xgl,ax,bx);
                            
                       
%Compute Metric Terms
[ksi_x,ksi_y,eta_x,eta_y,jac] = metrics(coord,intma,psi,dpsi,wnq,nelem,ngl,nq);

%Compute Side/Edge Information
[iside,jeside] = create_side(intma,bsido,npoin,nelem,nboun,nside,ngl);
[psideh,imapl,imapr] = create_side_dg(iside,intma,nside,nelem,ngl);
[nx,ny,jac_side] = compute_normals(psideh,intma,coord,...
                   nside,ngl,nq,wnq,psi,dpsi);
       
%Compute Exact Solution
[qe,qe_x,qe_y,fe] = exact_solution(coord,npoin,icase);

%Create RHS Vector and LMatrix  

Mmatrix = create_Mmatrix(jac,intma,psi,npoin,nelem,ngl,nq);

Rvector = Mmatrix*fe;

Lmatrix = create_Lmatrix(intma,jac,ksi_x,ksi_y,eta_x,...         
                   eta_y,psi,dpsi,npoin,nelem,ngl,nq);
               

% Apply boundary conditions
    
if (bc_type == 5)  % dirichlet

   [Lmatrix,Rvector] = apply_Dirichlet_BC_Vector(Lmatrix,Rvector,psideh,...
               nside,ngl,imapl,intma,qe);

elseif(bc_type == 4) % neumann

    Rvector = apply_Neumann_BC_Vector(Rvector,jac_side,psideh,...
               nside,ngl,imapl,intma,qe_x,qe_y,nx,ny,npoin);

    Lmatrix(1,:) = 0.0;
    Lmatrix(1,1) = 1.0;
    Rvector(1) = qe(1);
end 

%Solve System 
q0 = Lmatrix\Rvector; 


%Compute Norm

error = abs(q0-qe);
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
    nxx=100; nyy=100;
    dx=(xmax-xmin)/nxx;
    dy=(ymax-ymin)/nyy;
    [xi,yi]=meshgrid(xmin:dx:xmax,ymin:dy:ymax);
    qi=griddata(xe,ye,q0,xi,yi,'cubic');
    % [cl,h]=contourf(xi,yi,qi);
    surf(xi,yi,qi);
    colorbar('SouthOutside');
    xlabel('X','FontSize',18);
    ylabel('Y','FontSize',18);
    axis image
    title_text=[space_method ', Ne = ' num2str(nelem) ', N = ' num2str(nop) ', Q = ' num2str(noq) ', L2 Norm = ' num2str(l2_norm)];
    title([title_text],'FontSize',18);
    set(gca, 'FontSize', 18);

    disp(['space_method = ',space_method]);
    disp(['nop = ',num2str(nop),'  nelem = ',num2str(nelem) ]);
    q_max=max(q0);
    q_min=min(q0);
    disp(['L2_Norm = ',num2str(l2_norm),' q_max = ',num2str(q_max),'  q_min = ',num2str(q_min) ]);
    disp(['npoin = ',num2str(npoin)]);
end

toc
