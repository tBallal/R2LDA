function [gama, k] = bpr(U,s, y)
% gama is the regularisation parameter for the problem y = Ax + w
% A = U*S*V'; s = diag(S);
%  $Author: Tarig Ballal                                                                                                                                            
%  $Rev: 0                                                                                                                 
%
%  Version History
%  1.0         13/May/2015     Tarig Ballal    First version

EPS   =  1e-10; 

s2    =  s.^2; 
n     =  length(s);
b     =  U'*y; 
dbbT  =  b.*b;

gama     =  0; 
k        =  0;
d        =  1./(s2); 
sd       =  sum(d);
d2       =  d.^2;
sd2      =  sum(d2);
G0       =  sd*( d'*dbbT ) - n*(d2'*dbbT);
if G0<0
d3       =  d2.*d;    
dG0      =  -sd*( d2'*dbbT ) - sd2*( d'*dbbT ) + 2*n*( d3'*dbbT );
gama     =  gama - G0/dG0;
for k = 1:1000
    d        =  1./(s2 + gama); 
    sd       =  sum(d);
    d2       =  d.^2;
    sd2      =  sum(d2);
    d3       =  d2.*d;   
    G        =  sd*( d'*dbbT ) - n*(d2'*dbbT);
    dG       =  -sd*( d2'*dbbT ) - sd2*( d'*dbbT ) + 2*n*( d3'*dbbT );
    gama     =  gama - G/dG;
    if abs(G) < EPS
        break;
    end
end

% end 



end

