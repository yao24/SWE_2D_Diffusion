def compute_flux_SIP_qterms(q,l_basis,dl_basis,N,Q,ksi_x,ksi_y,eta_x,eta_y,iloc,e,l):

    #Interpolate Left-State onto Quadrature Points
    q_k = 0; q_e = 0; q_n = 0

    if(iloc == 0):  #Side 1: eta=-1


        for i in range(N+1):

            h_jk= l_basis[0,0]*l_basis[i,l]
            dhde_jk = dl_basis[i,l]*l_basis[0,0]
            dhdn_jk = l_basis[i,l]*dl_basis[0,0]
            q_k += h_jk*q[i]
            q_e += dhde_jk*q[i]
            q_n += dhdn_jk*q[i]

        q_x = q_e*ksi_x[e,0,l] + q_n*eta_x[e,0,l]
        q_y = q_e*ksi_y[e,0,l] + q_n*eta_y[e,0,l]
        
        

    elif(iloc == 1):  #Side 2: ksi=+1

        for i in range(N+1):

            h_jk = l_basis[N,Q]*l_basis[i,l]
            dhde_jk = dl_basis[N,Q]*l_basis[i,l]
            dhdn_jk = l_basis[N,Q]*dl_basis[i,l]
            q_k += h_jk*q[i]
            q_e += dhde_jk*q[i]
            q_n += dhdn_jk*q[i]

        q_x = q_e*ksi_x[e,l,Q] + q_n*eta_x[e,l,Q]
        q_y = q_e*ksi_y[e,l,Q] + q_n*eta_y[e,l,Q]

    elif(iloc == 2):  #Side 3: eta=+1
        
        for i in range(N+1):

            h_jk = l_basis[i,l]*l_basis[N,Q]
            dhde_jk = dl_basis[i,l]*l_basis[N,Q]
            dhdn_jk = l_basis[i,l]*dl_basis[N,Q]
            q_k += h_jk*q[i]
            q_e += dhde_jk*q[i]
            q_n += dhdn_jk*q[i]

        q_x = q_e*ksi_x[e,Q,l] + q_n*eta_x[e,Q,l]
        q_y = q_e*ksi_y[e,Q,l] + q_n*eta_y[e,Q,l]

    elif(iloc == 3):  #Side 4: ksi=-1

        for i in range(N+1):

            h_jk = l_basis[0,0]*l_basis[i,l]
            dhde_jk = dl_basis[0,0]*l_basis[i,l]
            dhdn_jk = l_basis[0,0]*dl_basis[i,l]
            q_k += h_jk*q[i]
            q_e += dhde_jk*q[i]
            q_n += dhdn_jk*q[i]

        q_x = q_e*ksi_x[e,l,0] + q_n*eta_x[e,l,0]
        q_y = q_e*ksi_y[e,l,0] + q_n*eta_y[e,l,0]


    return q_k, q_x, q_y

def compute_flux_SIP_basis(l_basis,dl_basis,N,Q,ksi_x,ksi_y,eta_x,eta_y,iloc,e,i,l):

    #Interpolate State onto Quadrature Points

    if(iloc == 0): #Side 1: eta=-1

        psi_e = dl_basis[i,l]*l_basis[0,0]
        psi_n = l_basis[i,l]*dl_basis[0,0]
        psi_x = psi_e*ksi_x[e,0,l] + psi_n*eta_x[e,0,l]
        psi_y = psi_e*ksi_y[e,0,l] + psi_n*eta_y[e,0,l]

    elif(iloc == 1):   #Side 2: ksi=+1

        psi_e = dl_basis[N,Q]*l_basis[i,l]
        psi_n = l_basis[N,Q]*dl_basis[i,l]
        psi_x = psi_e*ksi_x[e,l,Q] + psi_n*eta_x[e,l,Q]
        psi_y = psi_e*ksi_y[e,l,Q] + psi_n*eta_y[e,l,Q]

    elif(iloc == 2): #Side 3: eta=+1

        psi_e = dl_basis[i,l]*l_basis[N,Q]
        psi_n = l_basis[i,l]*dl_basis[N,Q]
        psi_x = psi_e*ksi_x[e,Q,l] + psi_n*eta_x[e,Q,l]
        psi_y = psi_e*ksi_y[e,Q,l] + psi_n*eta_y[e,Q,l]

    elif(iloc == 3): #Side 4: ksi=-1

        psi_e = dl_basis[0,0]*l_basis[i,l]
        psi_n = l_basis[0,0]*dl_basis[i,l]
        psi_x = psi_e*ksi_x[e,l,0] + psi_n*eta_x[e,l,0]
        psi_y = psi_e*ksi_y[e,l,0] + psi_n*eta_y[e,l,0]

    return psi_x, psi_y

def compute_flux_SIP_BC(rhs,q,psideh,nx,ny,jac_side,jac,l_basis,dl_basis,\
       Nside,N,Q,imapl,imapr,intma,ksi_x,ksi_y,eta_x,eta_y,mu_constant,gradq):

    #local arrays
    qr = zeros(Q+1)
    ql = zeros(Q+1)
    #Construct FVM-type Operators
    for n in range(Nside):

        #Store Left Side Variables
        el = int(psideh[n,2])
        ilocl = int(psideh[n,0])
        er = int(psideh[n,3])

        flag = 0
        
        if(er < 0):
        
            flag = er%-10
            
        for l in range(N+1):
            il = int(imapl[l,ilocl,0])
            jl = int(imapl[l,ilocl,1])
            IL = int(intma[el,il,jl])
            #print(el,IL)
            ql[l] = q[IL] 

        #Store Right Side Variables
        if(er >= 0): 
            ilocr = int(psideh[n,1]) 
            #print(ilocr)
            for l in range(N+1):
                ir = int(imapr[l,ilocr,0])
                jr = int(imapr[l,ilocr,1])
                IR = int(intma[er,ir,jr])
                #print(el,IR)
                qr[l] = q[IR]
        
        elif(er == -4):  #Neumann 
            ilocr = ilocl
            for l in range(N+1):
                il = int(imapl[l,ilocl,0])
                jl = int(imapl[l,ilocl,1])
                IL = int(intma[el,il,jl])
                nxl = nx[n,l]
                nyl = ny[n,l]

                wq = jac_side[n,l]

                qr[l] = wq*(nxl*gradq[IL,0] + nyl*gradq[IL,1])
        #print(qr)
        #Do Gauss-Lobatto Integration
        for l in range(Q+1):
            wq = jac_side[n,l]

            #Store Normal Vectors
            nxl = nx[n,l]
            nyl = ny[n,l]

            nxr = -nxl
            nyr = -nyl
            
            #Interpolate Left-State onto Quadrature Points
            ql_k, ql_x, ql_y = compute_flux_SIP_qterms(ql,l_basis,dl_basis,N,Q,ksi_x,ksi_y,eta_x,eta_y,ilocl,el,l)                 
            #print(ql_k, ql_x, ql_y)
            #Interpolate Right-State onto Quadrature Points
            if(er >= 0):
                qr_k, qr_x, qr_y = compute_flux_SIP_qterms(qr,l_basis,dl_basis,N,Q,ksi_x,ksi_y,eta_x,eta_y,ilocr,er,l)
            
            elif(er == -4):
                qr_k, qr_x, qr_y = compute_flux_SIP_qterms(qr,l_basis,dl_basis,N,Q,ksi_x,ksi_y,eta_x,eta_y,ilocl,el,l)
            #print(qr_k, qr_x, qr_y)
            #Flux Variables

            fxl = ql_x
            fyl = ql_y
            fxr = qr_x
            fyr = qr_y

            #Normal Flux Component
            #flux_q = nxl*(fxl+fxr) + nyl*(fxl+fyr)
            flux_q = nxl*(fxr) + nyl*(fyr)
            #Dissipation Term
            mu_l = mu_constant*((N+1)*(N+2)/2)*(jac_side[n,l]/jac[el,0,l])
            mu_r = mu_constant*((N+1)*(N+2)/2)*(jac_side[n,l]/jac[el,0,l])
            mu = max(mu_l,mu_r)
            diss_q = mu*(qr_k - ql_k)
            #print(mu)
            #Construct Rusanov Flux
            flux_grad_q = 0.5*(flux_q - diss_q)
            q_mean = 0.5*(ql_k + qr_k)
            #print(flux_grad_q)
            #Loop through Side Interpolation Points
            for i in range(N+1):

                #Left States
                h_i = l_basis[i,l]          
                psil_x, psil_y = compute_flux_SIP_basis(l_basis,dl_basis,N,Q,ksi_x,ksi_y,eta_x,eta_y,ilocl,el,i,l)
                flux_psil = nxl*psil_x + nyl*psil_y
                #print(flux_psil)
                #--------------Left Side------------------%
                il = int(imapl[i,ilocl,0])
                jl = int(imapl[i,ilocl,1])
                IL = int(intma[el,il,jl])
                rhs[IL] += wq*h_i*flux_grad_q + wq*flux_psil*(ql_k - q_mean)

                #--------------Right Side------------------%
#                 flux_psir = flux_psil
#                 if (er > 0):
#                     psir_x, psir_y = compute_flux_SIP_basis(l_basis,dl_basis,N,Q,ksi_x,ksi_y,eta_x,eta_y,ilocr,er,i,l)
#                     flux_psir = nxr*psir_x + nyr*psir_y
#                     ir = int(imapr[i,ilocl,0])
#                     jr = int(imapr[i,ilocl,1])
#                     IR = int(intma[er,ir,jr])
#                     rhs[IR] -= wq*h_i*flux_grad_q + wq*flux_psir*(qr_k - q_mean)

    return rhs