function [alpha,beta,stages] = IRK_coefficients(ti_method)
    %UNTITLED Summary of this function goes here
    %   Detailed explanation goes here
    %IRK2
    if (ti_method == 1)
        stages=2;
        alpha=zeros(stages,stages);
        beta=zeros(stages,1);
        alpha(2,1)=0;
        alpha(2,2)=1;
        beta(:)=alpha(stages,:);
    elseif (ti_method == 2)
        stages=3;
        alpha=zeros(stages,stages);
        beta=zeros(stages,1);
        alpha(2,1)=1 - 1/sqrt(2);
        alpha(2,2)=alpha(2,1);
        alpha(3,1)=1/(2*sqrt(2));
        alpha(3,2)=alpha(3,1);
        alpha(3,3)=alpha(2,2);
        beta(:)=alpha(stages,:);
    elseif (ti_method == 3)
        stages=4;
        alpha=zeros(stages,stages);
        beta=zeros(stages,1);
        alpha(2,1)=1767732205903.0/4055673282236.0;
        alpha(2,2)=1767732205903.0/4055673282236.0;
        alpha(3,1)=2746238789719.0/10658868560708.0;
        alpha(3,2)=-640167445237.0/6845629431997.0;
        alpha(3,3)=alpha(2,2);
        alpha(4,1)=1471266399579.0/7840856788654.0;
        alpha(4,2)=-4482444167858.0/7529755066697.0;
        alpha(4,3)=11266239266428.0/11593286722821.0;
        alpha(4,4)=alpha(2,2);
        beta(:)=alpha(stages,:);
    elseif (ti_method == 4)
        stages=6;
        alpha=zeros(stages,stages);
        beta=zeros(stages,1);
        alpha(2,1)=1.0/4.0;
        alpha(2,2)=1.0/4.0;
        alpha(3,1)=8611.0/62500.0;
        alpha(3,2)=-1743.0/31250.0;
        alpha(3,3)=alpha(2,2);
        alpha(4,1)=5012029.0/34652500.0;
        alpha(4,2)=-654441.0/2922500.0;
        alpha(4,3)=174375.0/388108.0;
        alpha(4,4)=alpha(2,2);
        alpha(5,1)=15267082809.0/155376265600.0;
        alpha(5,2)=-71443401.0/120774400.0;
        alpha(5,3)=730878875.0/902184768.0;
        alpha(5,4)=2285395.0/8070912.0;
        alpha(5,5)=alpha(2,2);
        alpha(6,1)=82889.0/524892.0;
        alpha(6,2)=0.0;
        alpha(6,3)=15625.0/83664.0;
        alpha(6,4)=69875.0/102672.0;
        alpha(6,5)=-2260.0/8211.0;
        alpha(6,6)=alpha(2,2);
        beta(:)=alpha(stages,:);

    end

end

