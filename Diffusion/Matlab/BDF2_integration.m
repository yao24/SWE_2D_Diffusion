
function [qp,qe] = BDF2_integration(qe,DFmatrix,A,Amatrix,Mmatrix,coord,npoin,icase,dt,...
              jac_side,psideh,nside,ngl,imapl,intma,nx,ny,alpha,beta,stages,time,ntime)
          
q0 = qe;


[qp,qe,time] = IRK_integration(qe,DFmatrix,Amatrix,Mmatrix,coord,npoin,icase,dt,...
              jac_side,psideh,nside,ngl,imapl,intma,nx,ny,alpha,beta,stages,time,1);
          

%A = 3*Mmatrix - 2*dt*DFmatrix;


%A_inv = inv(A);

Rmatrix = A\Mmatrix;

q = qp;

for itime = 2:ntime
    
    time = time + dt;
    
    [qe,gradq] = exact_solution(coord,npoin,icase,time);
    
    qn = 4*q - q0;
    
    qn = Dirichlet_BC_vec(qn,psideh,nside,ngl,imapl,intma,qe);
    
    qp = Rmatrix*qn;

    qp = Neumann_BC_Vector(qp,jac_side,psideh,...
               nside,ngl,imapl,intma,gradq,nx,ny,npoin,2*dt,A_inv);
           
    q0 = q;
    
    q = qp;
              
end