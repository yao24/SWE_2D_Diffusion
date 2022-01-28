

function Rvector = apply_Neumann_BC_Vector(Rvector,jac_side,psideh,...
               nside,ngl,imapl,intma,qe_x,qe_y,nx,ny,npoin)

B = zeros(npoin,1);
%Construct FVM-type Operators
for is=1:nside

   %Store Left Side Variables
   el=psideh(is,3);
   er=psideh(is,4);
   if (er == -4) %Dirichlet bc
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
          ndp = nxl*qe_x(I) + nyl*qe_y(I);
          B(I) = B(I) - wq*ndp;
      end %l  
   end %if er      
end %is

Rvector = Rvector + B;
end
