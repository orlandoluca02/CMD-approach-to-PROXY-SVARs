function [logLik]=Likelihood_SVAR(teta)

global Sigma_Eta

global TAll

G=[teta(1)  teta(2)  teta(3)  0        0;
   teta(4)  teta(5)  0        0        0;    
   teta(6)  teta(7)  teta(8)  0        0;
   teta(9)  0        0        teta(10) 0;
   teta(11) 0        0        0        teta(12)];

    K = G^(-1);
    
    T=TAll;
    M=size(G,1);

    logLik=-(-0.5*T*M*(log(2*pi))...
        +0.5*T*log((det(K))^2)-0.5*T*trace(K'*K*Sigma_Eta));     

end