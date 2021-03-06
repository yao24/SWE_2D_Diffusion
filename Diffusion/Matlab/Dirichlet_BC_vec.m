%----------------------------------------------------------------------%
%This subroutine builds the FLUX vector for the Strong Form DGM-SEM
%on Quadrilateral Elements for the 2D Euler Equations.
%Written by Francis X. Giraldo on 1/2001
%           Naval Postgraduate School
%           Department of Applied Mathematics
%           Monterey, CA 93943-5216

% modified by Yao Gahounzo   29/01/2022
%             Boise State University
%----------------------------------------------------------------------%
function Rvector = Dirichlet_BC_vec(Rvector,psideh,nside,ngl,imapl,intma,qe)

%Construct FVM-type Operators
for is=1:nside

   %Store Left Side Variables
   el=psideh(is,3);
   er=psideh(is,4);
   if (er == -5) % Dirichlet bc
      ilocl=psideh(is,1);
      for l=1:ngl
          %Get Pointers
          il = imapl(ilocl,1,l);
          jl = imapl(ilocl,2,l);
          I = intma(el,il,jl);
          
          %Left Element
          Rvector(I) = qe(I);
          
      end %l  
   end %if er      
end %is