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
iplot = 1;
integration_type = 1; %=1 is inexact and =2 is exact
space_method = 'CG'; 
kstages = 3;
icase = 1; % 1 or 2 
           
           
Tfinal = 0.01;           
time_method = 2;  % 1 = IRK, 2 = BDF2, 3 = BDF3

% Boundary type: 5 = dirichlet or 4 = neumann

bound_index = [5,5,5,5]; % [bottom,right,top,left] boundaries
           
ax = 0;
bx = 2*pi;

c = 1;

nopp = [3];
Ne = [8 16 24 32];

%fileID = fopen('matlab_convergence.dat','w');

for inop = 1:size(nopp,2)
    nop = nopp(inop);
    ngl = nop + 1;

    fileID = fopen(sprintf('conv_nop%d.dat',nop),'w');
    icount = 0;
    %cfl = 1/(nop +1);
    
    for id = 1:4
        nel = Ne(id);
        icount = icount + 1;
        t0 = cputime;
        nelx = nel;
        nely = nel;
        nelem = nelx*nely; %Number of Elements
        ngl = nop + 1;
        npts = ngl*ngl;
        Nx = nop*nelx + 1;
        npoin = (nop*nelx + 1)*(nop*nely + 1);
        nboun = 2*nelx + 2*nely;
        nside = 2*nelem + nelx + nely;
        
        cfl = 0.001; %1/npoin;

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
        Lmatrix = apply_Dirichlet_BC_matrix(Lmatrix,psideh,...
                       nside,ngl,imapl,intma);


        DFmatrix = c*Lmatrix;

        dx = coord(2,1) - coord(1,1);
        %dx = (bx-ax)/Nx;
        dt_est = cfl*dx^2;
        ntime = floor(Tfinal/dt_est +1);
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
            
            Abdf2 = 3*Mmatrix - 2*dt*DFmatrix;
            
            Abdf2 = apply_Dirichlet_BC_matrix(Abdf2,psideh,...
                       nside,ngl,imapl,intma);
            
            [qp,qe] = BDF2_integration(qe,DFmatrix,Abdf2,Amatrix,Mmatrix,coord,npoin,icase,dt,...
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

        fprintf(fileID,'%d %d %12.4e %12.4e\n', nop,ntime,cfl,l2_norm);

        l2_norm_total(icount,inop)=l2_norm;
        npoin_total(icount,inop)=nel;
        t1=cputime;
        dt=t1-t0;
        
        disp([' nop  = ' num2str(nop),' nel = ' num2str(nel),' cpu = ' num2str(dt), sprintf('\tl2_norm =  %0.3e',l2_norm), ' ntime = ' num2str(ntime) ]);

    end %nel

    
end %inop

fclose(fileID);

if(iplot == 1)
    h=figure;
    figure(h);
    for inop = 1:size(nopp,2)
        switch inop
        case (1)
            p1 = polyfit(log(npoin_total(:,inop)),log(l2_norm_total(:,inop)),1);
            p(1) = p1(1)
            plot_handle=loglog(npoin_total(:,inop),l2_norm_total(:,inop),'r-*');
        case (2)
            p2 = polyfit(log(npoin_total(:,inop)),log(l2_norm_total(:,inop)),1);
            p(2) = p2(1);
            plot_handle=loglog(npoin_total(:,inop),l2_norm_total(:,inop),'b-o');
    %     case (3)
    %         p3 = polyfit(log(npoin_total(:,inop)),log(l2_norm_total(:,inop)),1);
    %         p(3) = p3(1);
    %         plot_handle=loglog(npoin_total(:,inop),l2_norm_total(:,inop),'g-x');

    end %switch
    set(plot_handle,'LineWidth',2);
    hold on
    end

    title([time_integration],'FontSize',18);
    xlabel('Number of elements','FontSize',18);
    ylabel('Normalized L^2 Error','FontSize',18);
    legend(sprintf('N = 2, rate = %0.3f',p(1)));
    set(gca, 'FontSize', 18);
end

toc
