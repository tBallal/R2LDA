clc,clear all;

% addpath('.\R2LDA_Access_codes');
% addpath('.\R2LDA_Access_codes\DataSets') 
% addpath('.\R2LDA_Access_codes\Impr[21]') 
% addpath('.\R2LDA_Access_codes\regu') 
% addpath('.\R2LDA_Access_codes\BPR') 

%% Gaussian
p=100; %dimensionality
S0=toeplitz([1 0.1*ones(1,p-1)]);  %actual cov matrix of class 0
S1=S0+eye(p);
Del_sq=9;
m0=sqrt(Del_sq/(4*sum(sum(inv(S0)))))*ones(p,1);  %mean vectors of two classes  (Type-I)
m1=-m0;
C=[0, 1];    
%class labels
prob=[1/2, 1/2];      %class prior probabilities
ntr=10:10:110;        %training data size (use even)  20:20:200;
nts=500;              %testing data size 500
max_iter=500;          %no of trials
for stdn = [0 0.1 0.2 0.3 0.4 0.5];             %noise standard deviation
randn('state', 10);                
rand('state', 10);     
error_bpr=zeros(1,length(ntr));
error_bpri=zeros(1,length(ntr));
error_quasi=zeros(1,length(ntr));
error_gcv=zeros(1,length(ntr));
error_curv=zeros(1,length(ntr));
error_bay=zeros(1,length(ntr));
error_asy=zeros(1,length(ntr));
error_imp=zeros(1,length(ntr));
for ii=1:length(ntr)             %can use parallel for loop "parfor"
    N1=ntr(ii);
    disp(['Gaussian: ' num2str([stdn N1])])
    %[stdn N1];
    ntr0=0.5*N1;          %training samples for class 0
    ntr1=N1-ntr0;       %training samples for class 1
    c0=ntr0/N1; c1=ntr1/N1;       
    cc=log(c1/c0);        %decision threshold
    gg=-10:1:10;ropt_range=1000.^(gg./10);
    err_bay_iter=zeros(max_iter,1); err_asy_iter=zeros(max_iter,1);
    err_bpr_iter=zeros(max_iter,1); 
    err_bpri_iter=zeros(max_iter,1); 
    err_quasi_iter=zeros(max_iter,1);
    err_gcv_iter=zeros(max_iter,1); err_curv_iter=zeros(max_iter,1);
    err_imp_iter=zeros(max_iter,1);
    parfor iter = 1:max_iter
        warning('off');
        %%Training & Testing data
        %generate X1 (training data) and class labels
        X1 = [mvnrnd(m0,S0,ntr0); mvnrnd(m1,S1,ntr1)];    %N1xp data matrix
        %generate X2 (test data) and class labels
        nts0 = 0.5*nts;          %test sample of class 0
        nts1 = nts - nts0;       %test sample of class 1
        N2 = nts0 + nts1;
        X2 = [mvnrnd(m0,S0,nts0); mvnrnd(m1,S1,nts1)];    %N2xp data matrix
        nrm =max(abs([X1(:); X2(:)]));
        X1=X1/nrm;              %data matrix of class 0
        X2=X2/nrm;              %data matrix of class 1
        X2 = X2 + stdn*randn(size(X2));
        y1 = [C(1)*ones(ntr0,1); C(2)*ones(ntr1,1)];      %class labels             
        y2 = [C(1)*ones(nts0,1); C(2)*ones(nts1,1)];      %class labels
        %random permutation of test data
        perm = randperm(N2);
        X2 = X2(perm,:); 
        y2 = y2(perm);
        %%estimate class statistics
        m0_hat = [mean(X1(1:ntr0,:))]';
        m1_hat = [mean(X1(ntr0+1:end,:))]';
        m_plus = m0_hat + m1_hat;
        m_minus = m0_hat - m1_hat;
        S0_hat = (X1(1:ntr0,:)-repmat(m0_hat',ntr0,1))'*(X1(1:ntr0,:)-repmat(m0_hat',ntr0,1))/(ntr0-1);
        S1_hat = (X1(ntr0+1:end,:)-repmat(m1_hat',ntr1,1))'*(X1(ntr0+1:end,:)-repmat(m1_hat',ntr1,1))/(ntr1-1);
        %S0_hat = S0_hat + 0.5*randn(size(S0_hat));
        %S1_hat = S1_hat + 0.5*randn(size(S1_hat));
        S_hat = 1/(ntr0+ntr1-2)*((ntr0-1)*S0_hat + (ntr1-1)*S1_hat);
        %%%% Bayes classifier %%%%
%         W_Bay=prob(1)*mvnpdf(X2,m0',S0)-prob(2)*mvnpdf(X2,m1',S1);

        X2 = X2 + stdn*randn(size(X2));
        
        %%%% Asymptotic classifier  %%%%%%
        epsl=[];
        for tt = 1:length(ropt_range)
            H0 = inv(eye(p) + ropt_range(tt)*S_hat);
            G0 = (m0_hat - 1/2*m_plus)'*H0*m_minus;
            G1 = (m1_hat - 1/2*m_plus)'*H0*m_minus;
            D0 = m_minus'*H0*S_hat*H0*m_minus;
            del_hat = (p - trace(H0))/(ropt_range(tt)*(N1-2-p + trace(H0)));
            eps_0 = normcdf((-G0+(N1-2)*del_hat/ntr0+cc)/sqrt((1+ropt_range(tt)*del_hat)^2*D0),0,1);  %e0
            eps_1 = normcdf((G1+(N1-2)*del_hat/ntr1-cc)/sqrt((1+ropt_range(tt)*del_hat)^2*D0),0,1);  %e1
            epsl = [epsl, c0*eps_0 + c1*eps_1];   %total error
        end
        [e_min,indx_opt]=min(epsl);
        r_asy=ropt_range(indx_opt);
        H=inv(r_asy*S_hat + eye(p));
        W_Asy=(X2-1/2*repmat(m_plus',N2,1))*H*m_minus;
        
        %%%%% Improved Classifier (Sifaou)   
%         [Sig_imp_OII,thet_opt] = train_OII(X1,y1);
%         [y2_hat]=test_OIILDA(Sig_imp_OII,thet_opt,m0_hat,m1_hat,X2);
%         err_imp_iter(iter)=sum(y2~=y2_hat)/N2; 
        
        %%%%% RRLDA classifier  $$$$$$$$$$$$
        [U,DD,~]=svd(S_hat);
        dd=diag(DD);
        d=sqrt(dd);
%         thresh=0.001*mean(dd);      %eigenvalues less than thresh are discarded
%         len1=sum(dd>thresh);
        len1 = min(N1, p);
        [rbi]=covcopra(U,dd,m_minus,len1);
        [rb]=bpr(U,d,m_minus);
        [rb_gcv]=gcv(U,d,m_minus,'tikh')^2;
        [rb_quasi]=0; % quasiopt(U,d,m_minus,'tikh')^2;
        [rb_curv]=0; % l_curve(U,d,m_minus,'tikh')^2;
        %%% test statistics
        Um_minus=U'*m_minus;
        W_BPR=zeros(N2,1);  W_BPRI=zeros(N2,1);W_Quasi=zeros(N2,1);W_GCV=zeros(N2,1);W_CURV=zeros(N2,1);
        for jj=1:N2
            x_temp=X2(jj,:)' - 1/2*m_plus;
            x_tempU=x_temp'*U;
            [rz]=bpr(U,d,x_temp);
            [rzi]=covcopra(U,dd,x_temp,len1);
            [rz_gcv]=gcv(U,d,x_temp,'tikh')^2;
            [rz_quasi]=0; %quasiopt(U,d,x_temp,'tikh')^2;
            [rz_curv]=0;%l_curve(U,d,x_temp,'tikh')^2;
            W_BPR(jj,1)=x_tempU*diag(dd./( (dd+rz).*(dd+rb) ) )*Um_minus;
            W_BPRI(jj,1)=x_tempU*diag(dd./( (dd+rzi).*(dd+rbi) ) )*Um_minus;
            W_Quasi(jj,1)=x_tempU*diag(dd./( (dd+rz_quasi).*(dd+rb_quasi) ) )*Um_minus;
            W_GCV(jj,1)=x_tempU*diag(dd./( (dd+rz_gcv).*(dd+rb_gcv) ) )*Um_minus;
            W_CURV(jj,1)=x_tempU*diag(dd./( (dd+rz_curv).*(dd+rb_curv) ) )*Um_minus;
%             W_BPR(jj,1)=x_temp'*U*Sig*inv(Sig+((rz))*eye(p))*inv(Sig+real(abs(rb))*eye(p))*U'*m_minus;
%             W_BPRI(jj,1)=x_temp'*U*Sig*inv(Sig+((rzi))*eye(p))*inv(Sig+real(abs(rbi))*eye(p))*U'*m_minus;
%             W_Quasi(jj,1)=x_temp'*U*Sig*inv(Sig+rz_quasi*eye(p))*inv(Sig+rb_quasi*eye(p))*U'*m_minus;
%             W_GCV(jj,1)=x_temp'*U*Sig*inv(Sig+rz_gcv*eye(p))*inv(Sig+rb_gcv*eye(p))*U'*m_minus;
%             W_CURV(jj,1)=x_temp'*U*Sig*inv(Sig+rz_curv*eye(p))*inv(Sig+rb_curv*eye(p))*U'*m_minus;  
        end       
        
%         y2_hat = zeros(N2,1);           %Bayes
%         y2_hat(W_Bay>cc)=C(1);
%         y2_hat(W_Bay<=cc)=C(2);
%         err_bay_iter(iter)=sum(y2~=y2_hat)/N2;

        y2_hat = zeros(N2,1);           %Asym
        y2_hat(W_Asy>0)=C(1);
        y2_hat(W_Asy<=0)=C(2);
        err_asy_iter(iter)=sum(y2~=y2_hat)/N2;

        y2_hat=zeros(N2,1);             %BPR
        y2_hat(W_BPR>cc)=C(1);
        y2_hat(W_BPR<=cc)=C(2);
        err_bpr_iter(iter)=sum(y2~=y2_hat)/N2;
        
        y2_hat=zeros(N2,1);             %BPRI
        y2_hat(W_BPRI>cc)=C(1);
        y2_hat(W_BPRI<=cc)=C(2);
        err_bpri_iter(iter)=sum(y2~=y2_hat)/N2;

        y2_hat=zeros(N2,1);             %Quasi
        y2_hat(W_Quasi>cc)=C(1);
        y2_hat(W_Quasi<=cc)=C(2);
        err_quasi_iter(iter)=sum(y2~=y2_hat)/N2;

        y2_hat=zeros(N2,1);             %GCV
        y2_hat(W_GCV>cc)=C(1);
        y2_hat(W_GCV<=cc)=C(2);
        err_gcv_iter(iter)=sum(y2~=y2_hat)/N2;

        y2_hat=zeros(N2,1);             %CURV
        y2_hat(W_CURV>cc)=C(1);
        y2_hat(W_CURV<=cc)=C(2);
        err_curv_iter(iter)=sum(y2~=y2_hat)/N2;        
    end
%     error_bay(ii)=mean(err_bay_iter);
    error_imp(ii)=mean(err_imp_iter);
    error_asy(ii)=mean(err_asy_iter);
    error_bpr(ii)=mean(err_bpr_iter);
    error_bpri(ii)=mean(err_bpri_iter);
    error_quasi(ii)=mean(err_quasi_iter);
    error_gcv(ii)=mean(err_gcv_iter);
    error_curv(ii)=mean(err_curv_iter);
    %%save complete data
%     err_asy_stat(:,ii)=err_asy_iter;
%     err_bay_stat(:,ii)=err_bay_iter;
%     err_bpr_stat(:,ii)=err_bpr_iter;
%     err_quasi_stat(:,ii)=err_quasi_iter;
%     err_gcv_stat(:,ii)=err_gcv_iter;
%     err_curv_stat(:,ii)=err_curv_iter;
end
%%plot results
figure,
plot(ntr, 100*error_asy,'k*-'),
hold on,
plot(ntr, NaN*error_imp,'gH-'),
plot(ntr, 100*error_gcv,'mv-'),
plot(ntr, 100*error_bpr,'bo-'),
plot(ntr, 100*error_bpri,'rs-'),
grid on
hold off
xlabel('# of samples'), ylabel('Avg. percentage error'), 
legend('Asym-RLDA-','OII-LDA','GCV-R2LDA','BPR-R2LDA','COPRA-R2LDA');


% figure,
% plot(ntr, 100*error_asy,'k*-'),
% hold on,
% plot(ntr, 100*error_imp,'gH-'),
% plot(ntr, 100*error_gcv,'mv-'),
% plot(ntr, 100*error_bpr,'bo-'),
% plot(ntr, 100*error_bpri,'rs-'),
% plot(ntr, 100*error_curv,'k--'),
% plot(ntr, 100*error_quasi,'k-.'),
% grid on
% hold off
% xlabel('# of samples'), ylabel('Avg. percentage error'), 
% legend('RLDA-Asym','RLDA-Impr','R2LDA-GCV','R2LDA-BPR','R2LDA-COPRA','Lcurve-R2LDA','Quasi-R2LDA' );

% pth = "full path";
% spth = strcat(pth, filesep, "Gauss", num2str(stdn), '.fig')
% savefig(spth)
end


% pth = "D:\Tarig KAUST Work\Publications\SPL_Alam_Aug2018\RRLDA_Codes\CodeOct2019\MNIST_Results19Feb2020";
% spth = strcat(pth, filesep, "MNIST79_", '2', '.epsc')
% saveas(gcf, spth)
% close all

'Stope here!';

% for k=0:3
%     figure(k+19)
% pth='C:\Users\ahmedt\OneDrive - KAUST\R2LDA_Results_04Oct2020';
% stdn=k/10;
% stdnstr=num2str(10*stdn);
% figpth = strcat(pth, filesep, "Gauss_", stdnstr, '.fig');
% epspth = strcat(pth, filesep, "Gauss_", stdnstr, '.epsc')
% savefig(figpth);
% saveas(gcf, epspth)
% end