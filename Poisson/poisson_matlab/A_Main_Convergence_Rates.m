%---------------------------------------------------------------------%
%This code solves the 2D Poisson Equation using Unified CG/DG methods
%with tensor product of 1D basis function with either
%Exact or Inexact Integration and Using an NPOIN based data-structure
%Written by F.X. Giraldo on 9/2014
%           Department of Applied Mathematics
%           Naval Postgraduate School 
%           Monterey, CA 93943-5216
%
%modified by: Yao Gahounzo
%             Boise State University
%             Computing PhD
%---------------------------------------------------------------------%
clear all; 
close all;

tic

%Input Data
plot_conv = 0;
integration_type = 1; %=1 is inexact and =2 is exact
space_method = 'CG'; 

bc_type = 4; % 5 = dirichlet or 4 = neumann

icase = 3; % 1 = 2D with homogeneous BCs in x and y; 
           % 2 = 1D with homogeneous BCs along x=-1/+1 and non-homogeneous along y=-1/+1 .
           % 3 = 2D with non-homogeneous BCs along x and y.
           
ax = -1;
bx = 1;

nopp = [1,2];
Ne = [8 16 24 32];

%fileID = fopen('matlab_convergence.dat','w');

for inop = 1:size(nopp,2)
    nop = nopp(inop);
    ngl = nop + 1;

fileID = fopen(sprintf('conv_nop%d_4.dat',nop),'w');
icount = 0;
for id = 1:size(Ne,2)
    nel = Ne(id);
    icount = icount + 1;
    t0 = cputime;
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
    
    fprintf(fileID,'%d %12.4e %12.4e %12.4e\n', npoin,l1_norm,l2_norm,inf_norm);
    
    l2_norm_total(icount,inop)=l2_norm;
    npoin_total(icount,inop)=nel;
    t1=cputime;
    dt=t1-t0;
    
    Error(:,inop) = [l1_norm l2_norm inf_norm];
    

    disp([' nop  = ' num2str(nop),' nel = ' num2str(nel),' cpu = ' num2str(dt) ]);

end %nel

    
end %inop

fclose(fileID);

if(plot_conv)
    h=figure;
    figure(h);
    for inop = 1:3
        switch inop
        case (1)
            p1 = polyfit(log(npoin_total(:,inop)),log(l2_norm_total(:,inop)),1);
            p(1) = p1(1);
            plot_handle=loglog(npoin_total(:,inop),l2_norm_total(:,inop),'r-*');
        case (2)
            p2 = polyfit(log(npoin_total(:,inop)),log(l2_norm_total(:,inop)),1);
            p(2) = p2(1);
            plot_handle=loglog(npoin_total(:,inop),l2_norm_total(:,inop),'b-o');
        case (3)
            p3 = polyfit(log(npoin_total(:,inop)),log(l2_norm_total(:,inop)),1);
            p(3) = p3(1);
            plot_handle=loglog(npoin_total(:,inop),l2_norm_total(:,inop),'g-x');
    %     case (4)
    %         p4 = polyfit(log(npoin_total(:,inop)),log(l2_norm_total(:,inop)),1);
    %         p(4) = p4(1);
    %         plot_handle=loglog(npoin_total(:,inop),l2_norm_total(:,inop),'k-+');

    end %switch
    set(plot_handle,'LineWidth',2);
    hold on
    end

    title([main_text],'FontSize',18);
    xlabel('Number of elements','FontSize',18);
    ylabel('Normalized L^2 Error','FontSize',18);
    legend(sprintf('N = 3, rate = %0.3f',p(1)),sprintf('N = 4, rate = %0.3f',p(2)),...
        sprintf('N = 5, rate = %0.3f',p(3)));
    set(gca, 'FontSize', 18);
end

toc
