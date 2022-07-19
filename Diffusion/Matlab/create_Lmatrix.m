%---------------------------------------------------------------------%
%This function computes the 2D Laplacian Matrix on Quadrilaterals.
%Written by F.X. Giraldo on May 1, 2008
%           Department of Applied Mathematics
%           Naval Postgraduate School 
%           Monterey, CA 93943-5216
%---------------------------------------------------------------------%
function Lmatrix = create_Lmatrix(intma,jac,ksi_x,ksi_y,eta_x,...         
                   eta_y,psi,dpsi,npoin,nelem,ngl,nq)

%Initialize
%Rvector=zeros(npoin,1);
Lmatrix = zeros(npoin,npoin);
inode=zeros(ngl,ngl);

for e=1:nelem
   
   %Store Coordinates
   for j=1:ngl
   for i=1:ngl
      ip=intma(e,i,j);
      inode(i,j)=ip;
   end %i
   end %j
   
   %Do LGL Integration
   for l=1:nq
   for k=1:nq
       wq=jac(e,k,l);
       
       e_x=ksi_x(e,k,l);
       e_y=ksi_y(e,k,l);
       n_x=eta_x(e,k,l);
       n_y=eta_y(e,k,l);
           
       %Loop through I points
       for j=1:ngl
       for i=1:ngl
           jp=intma(e,i,j);
           
           h_e=dpsi(i,k)*psi(j,l);
           h_n=psi(i,k)*dpsi(j,l);
           dhdx_j=h_e*e_x + h_n*n_x;
           dhdy_j=h_e*e_y + h_n*n_y;
           
           %Interpolate Derivatives onto Quadrature Points

           for n=1:ngl
           for m=1:ngl

               ip=intma(e,m,n);
               
               h_e=dpsi(m,k)*psi(n,l);
               h_n=psi(m,k)*dpsi(n,l);
               dhdx_i=h_e*e_x + h_n*n_x;
               dhdy_i=h_e*e_y + h_n*n_y;
               
               Lmatrix(ip,jp) = Lmatrix(ip,jp) - wq*(dhdx_i*dhdx_j + dhdy_i*dhdy_j);

           end %m
           end %n
       end %i
       end%j
   end %k
   end %l
end %e

      
