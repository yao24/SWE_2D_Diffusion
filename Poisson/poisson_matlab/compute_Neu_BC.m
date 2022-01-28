%----------------------------------------------------------------------%
%This subroutine builds the FLUX term for the Weak Form LDG
%on Quadrilateral Elements.
%Written by Francis X. Giraldo on 1/2001
%           Department of Applied Mathematics
%           Naval Postgraduate School
%           Monterey, CA 93943-5216
%----------------------------------------------------------------------%
function rhs = compute_Neu_BC(psideh,nx,ny,jac_side,psi,npoin,nside,ngl,nq,imapl,intma,q_x,q_y)

%Initialize
rhs=zeros(npoin,1);

%local arrays
qrx=zeros(ngl,1);
qry=zeros(ngl,1);

%Construct FVM-type Operators
for is=1:nside

    %Store Left Side Variables
    el=psideh(is,3);
    ilocl=psideh(is,1);
      
    %Store Right Side Variables
    er=psideh(is,4);
    if(er == -4)
        for l=1:ngl
                il=imapl(ilocl,1,l);
                jl=imapl(ilocl,2,l);
                IL=intma(el,il,jl);
                qrx(l)=q_x(IL);
                qry(l)=q_y(IL);
        end %l 

        %Do Gauss-Lobatto Integration
        for l=1:nq
            wq=jac_side(is,l);

            %Store Normal Vectors
            nxl=nx(is,l);
            nyl=ny(is,l);

            qrx_k=0; qry_k = 0;
            for i=1:ngl
                qrx_k = qrx_k + psi(i,l)*qrx(i);
                qry_k = qry_k + psi(i,l)*qry(i);
            end %i 

            %Loop through Side Interpolation Points
            for i=1:ngl

              %Left States
              h_i=psi(i,l);          

              %--------------Left Side------------------%
              il=imapl(ilocl,1,i);
              jl=imapl(ilocl,2,i);
              IL=intma(el,il,jl);

              rhs(IL)=rhs(IL) + wq*h_i*(nxl*qrx_k + nyl*qry_k);

            end %i
        end %l
    end %er
end %is

