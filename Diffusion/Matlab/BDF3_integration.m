
function [qp,qe] = BDF3_integration(qe,DFmatrix,Amatrix,Mmatrix,coord,npoin,icase,dt,...
              jac_side,psideh,nside,ngl,imapl,intma,nx,ny,alpha,beta,stages,time,ntime)
          
    q0 = qe;


    % First step in the BDF3 method
    [q1,qe,time] = IRK_integration(qe,DFmatrix,Amatrix,Mmatrix,coord,npoin,icase,dt,...
                  jac_side,psideh,nside,ngl,imapl,intma,nx,ny,alpha,beta,stages,time,1);

    % Second step in the BDF3 method
    [qp,qe,time] = IRK_integration(qe,DFmatrix,Amatrix,Mmatrix,coord,npoin,icase,dt,...
                  jac_side,psideh,nside,ngl,imapl,intma,nx,ny,alpha,beta,stages,0,2);

    % LHS matrix in BDF3 method
    A = 11*Mmatrix - 6*dt*DFmatrix;
    
    Rmatrix = A\Mmatrix;
    
    q = qp;

    for itime = 3:ntime

        time = time + dt;

        qp = Rmatrix*(18*q - 9*q1 + 2*q0);

        [qe,gradq] = exact_solution(coord,npoin,icase,time);

        qp = Neumann_BC_Vector(qp,jac_side,psideh,...
                   nside,ngl,imapl,intma,gradq,nx,ny,npoin,6*dt,A);

        qp = Dirichlet_BC_vec(qp,psideh,nside,ngl,imapl,intma,qe);

        q0 = q1;
        q1 = q;
        q = qp;
    end
    
end