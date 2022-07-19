
%----------------------------------------------------------------------%
%This subroutine builds the FLUX vector for the Neumann bc
%on Quadrilateral Elements for the 2D diffusin Equations.
%Written by Yao Gahounzo on 29/01/2022
%           Computing PhD
%           Boise State University
%----------------------------------------------------------------------%

function Rvector = Neumann_BC_Vector(Rvector,jac_side,psideh,...
               nside,ngl,imapl,intma,gradq,nx,ny,npoin,dt,Mmatrix_inv)

B = zeros(npoin,1);
%Construct FVM-type Operators
for is=1:nside

   %Store Left Side Variables
   el=psideh(is,3);
   er=psideh(is,4);
   if (er == -4) % Neumann bc
      ilocl=psideh(is,1);
      for l=1:ngl
          %Get Pointers
          il=imapl(ilocl,1,l);
          jl=imapl(ilocl,2,l);
          wq=jac_side(is,l);

          %Store Normal Vectors
          nxl=nx(is,l);
          nyl=ny(is,l);
          
          I=intma(el,il,jl);
          ndp = nxl*gradq(I,1) + nyl*gradq(I,2);
          B(I) = B(I) + wq*ndp;
      end %l  
   end %if er      
end %is

Rvector = Rvector + dt*(Mmatrix_inv*B);
end
