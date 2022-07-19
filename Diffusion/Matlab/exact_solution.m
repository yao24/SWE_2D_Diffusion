%---------------------------------------------------------------------%
%This function computes the Initial and Analytic Solutions.
%Written by Yao Gahounzo   29/01/2022
%           Computing PhD
%           Boise State University
%---------------------------------------------------------------------%
function [qe,gradq] = exact_solution(coord,npoin,icase,time)

%Initialize
qe = zeros(npoin,1);
gradq = zeros(npoin,2);
    
%Generate Grid Points
for i = 1:npoin
    
    x = coord(i,1);
    y = coord(i,2);
    
    if(icase == 1) 
        
        qe(i) = sin(x)*exp(-time);
        gradq(i,1) = cos(x)*exp(-time);
        gradq(i,2) = 0;
        
    elseif(icase == 2) 
        
        qe(i) = cos(pi*x)*cos(pi*y)*exp(-2*pi^2*time);
        gradq(i,1) = -pi*sin(pi*x)*cos(pi*y)*exp(-2*pi^2*time);
        gradq(i,2) = -pi*cos(pi*x)*sin(pi*y)*exp(-2*pi^2*time);
   
    end
end %ip      



      
