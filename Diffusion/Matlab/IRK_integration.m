
function [qp,qe,time] = IRK_integration(qe,DFmatrix,Amatrix,Mmatrix,coord,npoin,icase,dt,...
              jac_side,psideh,nside,ngl,imapl,intma,nx,ny,alpha,beta,stages,time,ntime)


q = qe;

Q = zeros(npoin,stages);
R = zeros(npoin,stages);

% A_inv = inv(Amatrix);
% Mmatrix_inv = inv(Mmatrix);
% Mmatrix_inv = sparse(Mmatrix_inv);

%time = 0;

for itime = 1:ntime
    

    Q(:,1) = q(:);
    R(:,1) = DFmatrix*Q(:,1);
    
    for i = 2:stages
        
       R_sum = zeros(npoin,1);
       aa = 0;
       
       for j = 1:i-1
           R_sum = R_sum + alpha(i,j)*R(:,j);
           
           aa = aa + alpha(i,j);
       end  
       
       RR = Mmatrix*Q(:,1) + dt*R_sum;
       Q(:,i) = Amatrix\RR;
       
       [qe,gradq] = exact_solution(coord,npoin,icase,time + aa*dt);
       
       Q(:,i) = Neumann_BC_Vector(Q(:,i),jac_side,psideh,...
               nside,ngl,imapl,intma,gradq,nx,ny,npoin,dt,Mmatrix_inv);
           
       %Q(:,i) = Dirichlet_BC_vec(Q(:,i),psideh,nside,ngl,imapl,intma,qe);
           
       R(:,i) = DFmatrix*Q(:,i);
       
    end
    
    R_sum = zeros(npoin,1);
    
    for i = 1:stages
        
       R_sum = R_sum + beta(i)*R(:,i);
       
    end
    
    qp = q + dt*(Mmatrix\R_sum );
    
    time = time + dt;
    
    [qe,gradq] = exact_solution(coord,npoin,icase,time);
    
    qp = Neumann_BC_Vector(qp,jac_side,psideh,nside,ngl,imapl,intma,gradq,nx,ny,npoin,dt,Mmatrix_inv);
           
    qp = Dirichlet_BC_vec(qp,psideh,nside,ngl,imapl,intma,qe);
    
    q = qp;
    
end