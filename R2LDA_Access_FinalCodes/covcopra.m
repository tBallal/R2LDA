
function [gama] = covcopra(U,dd,y,ns)

% covcopra returns the regularization parameter for a linear model with a
% matrix equal to the square root of a covariance matrix S.
%
%  gama is the regularization parameter
%  gama is the regularization prameter for the RLS estimator c_hat =
%  (S +gama*I)^-1*(S^1/2)'*y based on the model y = A*x + noise
%
%  S   = U*D^2*U'                    S is (n by n)  
%  dd  = diag(D^2)                   dd is (n by 1) 
%  y   = observation vector          y is (n by 1)
%  ns  = number of significant eigenvalues, a good choice is ns = rank(S)

 n=length(dd);
 n2=n-ns;
 
%  nit=0; % meaning that BPR failed
 if n2==0
     [gama, nit]=bpr(U,sqrt(dd),y);
%      if nit==0
%          ns=ns-1; % remove the smallest eigenvalue
%          [gama] = copra(U,dd,y,ns);
%      end
 else
     [gama] = copra(U,dd,y,ns);
 end
 
% [gama] = copra(U,dd,y,ns);

 function [gama] = copra(U,dd,y,ns)
%  n=length(dd);
%  n2=n-ns;
 
 Stop_limit =1e-25;
 gama0  = 1e-10;
    
    dd1= dd(1:ns);
   
    beta = n/ns;
    
    
    b= U'*y ; 
    bb= b.^2;
    
    F = dd+gama0;
   
    
    F0 = dd1+gama0; 
    
    F1 = (((F).^(-2)).*(bb));
   
    F2 = (beta*F0);
    
    F3 = (F0).^(-2);
    
    F4 = F3.*F2 ;

    F5 = dd.*F1;
    
        
    T1 = (F).^(-3); 
    
    T2 = T1.*bb;
    
    T3 = sum(dd.*T2); 
    
    T4 = (F0).^(-3);
    
    T5 = sum(F5);
       
    T6 = sum(dd1.*F4); 
    
    T7 = sum(F4);
    
    T8 = sum(F1);
      
    G0 = (T5*T7+ (n2/gama0)*T5  -T8*T6);
    
    term1_0 = 2*sum(T2)*T6 - sum((dd1.*F3)-(2*(dd1.*T4).*F2))*T8; 
  
    term2_0 = (n2/gama0)*(2*T3+(1/gama0)*T5); 
  
    term3_0 = sum(F3-(2*(T4).*F2))*T5- 2*T3*T7; 
    
    dG0 =term1_0-term2_0+term3_0 ; 
  
    gama   =  gama0 -(G0/dG0);
    
    for Iteration_number =2:500
        
        Temp0  = dd1+gama ; 
        Temp_0 = dd+gama; 
        Temp1  = (beta*dd1+gama);
        Temp2  = ((Temp0).^(-2));
        Temp_1 = Temp2.*Temp1;
        Temp3  = (Temp_0).^(-2).*(bb);
        Temp4  = sum(dd.*Temp3); 
        Temp5  = sum(dd1.*Temp_1);
        Temp7  = ((Temp_0).^(-3).*bb);
        Temp8  = sum(dd.*Temp7); 
        Temp9  = sum(Temp_1);
        Temp10 = sum(Temp3); 
        Temp13 = ((Temp0).^(-3));
        Temp14 = Temp13.*Temp1 ; 

        G      =   Temp4*(Temp9+(n2/gama)) -Temp10*Temp5;   %Temp 9 was Temp 12
 
        term1  =   2*sum(Temp7)*Temp5- Temp10*sum(dd1.*Temp2-(2*dd1.*Temp14)); 
        
        term2  =  (n2/gama)*(2*Temp8+(1/gama)*Temp4);     %same 
        
        term3  =  sum(Temp2-(2*Temp14))* Temp4- 2*Temp8*Temp9; 
        
        dG     =  term1-term2+term3 ;
    
        
        gama   =  gama -(G/dG);

          if abs(G)< Stop_limit
            break
          end
    end
          
 end

end
    

