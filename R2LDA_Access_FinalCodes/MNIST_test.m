clc,clear all;


% addpath('.\R2LDA_Access_codes');
% addpath('.\R2LDA_Access_codes\DataSets') 
% addpath('.\R2LDA_Access_codes\Impr[21]') 
% addpath('.\R2LDA_Access_codes\regu')
% addpath('.\R2LDA_Access_codes\BPR') 


%%  generate training/testing data from MNIST dataset
%training data
for digitNo=1:3
if digitNo==1    
    C=[1,7];     %class labels/digits [1,7],[5,8],[7,9]
elseif digitNo==2
    C=[5,8];
else
    C=[7,9];
end
load('MNIST_train.mat');
load('MNIST_test.mat');
labels=MNIST_train(:,1);
data_train=MNIST_train(:,2:401);
indx0=find(labels==C(1));
indx1=find(labels==C(2));
ntr0_full=length(indx0);          %no of samples in class 0
ntr1_full=length(indx1);          %no of samples in class 1
ntr_full=ntr0_full+ntr1_full;
XXtr0=data_train(indx0,:);      %training data matrix of class 0
XXtr1=data_train(indx1,:);      %training data matrix of class 1
clear MNIST_train data_train labels indx0 indx1
%testing data
labels=MNIST_test(:,1);
data_test=MNIST_test(:,2:401);
indx0=find(labels==C(1));
indx1=find(labels==C(2));
nts0_full=length(indx0);          %no of samples in class 0
nts1_full=length(indx1);          %no of samples in class 1
nts_full=nts0_full+nts1_full;
XXts0=data_test(indx0,:);       %training data matrix of class 0
XXts1=data_test(indx1,:);       %training data matrix of class 1
clear MNIST_test data_test labels indx0 indx1

Co = C;
C=[0, 1]; 

%% simulation parameters
p=400;              % dimensions/features
ntr=[40:40:360];    % no of training samples 50:50:300; 
nts=500;            % testing data size 
max_iter= 500;      % no of trials 
for stdn = [0 1 2];   
% randn('state', 10);                
% rand('state', 10);   
error_asy=zeros(1,length(ntr));
error_bpr=zeros(1,length(ntr));
error_bpri=zeros(1,length(ntr));
error_quasi=zeros(1,length(ntr));
error_gcv=zeros(1,length(ntr));
error_curv=zeros(1,length(ntr));
error_imp=zeros(1,length(ntr));
for ii=1:length(ntr)
    N1=ntr(ii);
    disp(['MNIST (' num2str(Co(1)) ', '  num2str(Co(2)) '): ' num2str([stdn N1])])
%     [stdn N1]
    ntr0=0.5*N1;            %training samples for class 0
    ntr1=N1-ntr0;           %training samples for class 1
%     ntr0=round( N1*ntr0_full/(ntr0_full+ntr1_full) );            %training samples for class 0
%     ntr1=N1-ntr0;           %training samples for class 1
    c0=ntr0/N1; c1=ntr1/N1;       
    cc=log(c1/c0);          %decision threshold
    gg=-10:1:10;ropt_range=1000.^(gg./10);
    err_asy_iter=zeros(max_iter,1);
    err_bpr_iter=zeros(max_iter,1); 
    err_bpri_iter=zeros(max_iter,1); 
    err_quasi_iter=zeros(max_iter,1);
    err_gcv_iter=zeros(max_iter,1); err_curv_iter=zeros(max_iter,1);
    err_imp_iter=zeros(max_iter,1); 
    parfor iter=1:max_iter
        warning('off');
        %%Training & Testing data
%         %generate X1 (training data) and class labels randomly from dataset
%         X1=[XXtr0(randi(ntr0_full,1,ntr0),:); XXtr1(randi(ntr1_full,1,ntr1),:)];
        X1=[XXtr0(randperm(ntr0_full,ntr0),:); XXtr1(randperm(ntr1_full,ntr1),:)]; 
        y1=[C(1)*ones(ntr0,1); C(2)*ones(ntr1,1)];    %class labels vector
        %generate X2 (test data) and class labels randomly
        nts0=0.5*nts;          %test sample of class 0
        nts1=nts-nts0;       %test sample of class 1
%         nts0=round( nts*nts0_full/(nts0_full+nts1_full) );       %test samples of class 0
%         nts1=nts-nts0;       %test samples of class 1
        N2=nts0+nts1;
        
%         X2=[XXts0(randi(nts0_full,1,nts0),:); XXts1(randi(nts1_full,1,nts1),:)];  
        X2=[XXts0(randperm(nts0_full,nts0),:); XXts1(randperm(nts1_full,nts1),:)];
        X2 = X2 + stdn*randn(size(X2));
        y2=[C(1)*ones(nts0,1); C(2)*ones(nts1,1)];      %class labels
        
%         nts0=0.5*nts;          %test sample of class 0
%         nts1=nts-nts0;       %test sample of class 1
%         N2=nts0+nts1;
%         indrnd0=randperm(ntr0_full);
%         indrnd1=randperm(ntr1_full);
%         ind0=indrnd0(1:ntr0); 
%         ind1=indrnd1(1:ntr1); 
%         X1=[XXtr0(ind0,:); XXtr1(ind1,:)]; 
%         indrnd0=randperm(nts0_full);
%         indrnd1=randperm(nts1_full);
%         ind0=indrnd0(1:nts0); 
%         ind1=indrnd1(1:nts1);
%         X2=[XXts0(ind0,:); XXts1(ind1,:)]; 
%         y1=[C(1)*ones(ntr0,1); C(2)*ones(ntr1,1)];  % class labels for training data
%         y2=[C(1)*ones(nts0,1); C(2)*ones(nts1,1)];  % class labels for test data
        
        X2 = X2 + stdn*randn(size(X2));

        %random permutation of test data
        perm=randperm(N2);
        X2=X2(perm,:); 
        y2=y2(perm);

        %%estimate class statistics
        m0_hat=[mean(X1(1:ntr0,:))]';
        m1_hat=[mean(X1(ntr0+1:end,:))]';
        m_plus=m0_hat+m1_hat;
        m_minus=m0_hat-m1_hat;
        S0_hat=(X1(1:ntr0,:)-repmat(m0_hat',ntr0,1))'*(X1(1:ntr0,:)-repmat(m0_hat',ntr0,1))/(ntr0-1);
        S1_hat=(X1(ntr0+1:end,:)-repmat(m1_hat',ntr1,1))'*(X1(ntr0+1:end,:)-repmat(m1_hat',ntr1,1))/(ntr1-1);
        S_hat=1/(ntr0+ntr1-2)*((ntr0-1)*S0_hat + (ntr1-1)*S1_hat);
        
        %%%% Asymptotic classifier  %%%%%%
        epsl=[];
        for tt=1:length(ropt_range)
            H0=inv(eye(p)+ropt_range(tt)*S_hat);
            G0=(m0_hat-1/2*m_plus)'*H0*m_minus;
            G1=(m1_hat-1/2*m_plus)'*H0*m_minus;
            D0=m_minus'*H0*S_hat*H0*m_minus;
            del_hat=(p-trace(H0))/(ropt_range(tt)*(N1-2-p+trace(H0)));
            eps_0=normcdf((-G0+(N1-2)*del_hat/ntr0+cc)/sqrt((1+ropt_range(tt)*del_hat)^2*D0),0,1);  %e0
            eps_1=normcdf((G0+(N1-2)*del_hat/ntr1-cc)/sqrt((1+ropt_range(tt)*del_hat)^2*D0),0,1);  %e1
            epsl=[epsl,c0*eps_0+c1*eps_1];   %total error
        end
        [e_min,indx_opt]=min(epsl);
        r_asy=ropt_range(indx_opt);
        H=inv(r_asy*S_hat+eye(p));
        W_Asy=(X2-1/2*repmat(m_plus',N2,1))*H*m_minus;
        
        
        %%%%% Improved Classifier (Sifaou)   
%         [Sig_imp_OII,thet_opt] = train_OII(X1,y1);
%         [y2_hat]=test_OIILDA(Sig_imp_OII,thet_opt,m0_hat,m1_hat,X2);
%         err_imp_iter(iter)=sum(y2~=y2_hat)/N2; 
        
        %%%%% RRLDA classifier  $$$$$$$$$$$$
        [U,DD,~]=svd(S_hat);
        dd=diag(DD);
        d=sqrt(dd);
%         thresh=0.01*mean(dd);      %eigenvalues less than thresh are discarded
%         len1=sum(dd>thresh);
        len1 = min(N1, p);
        [rbi]=covcopra(U,dd,m_minus,len1);
        [rb]=bpr(U,d,m_minus);
        [rb_gcv]=gcv(U,d,m_minus,'tikh')^2;
        [rb_quasi]=0;%quasiopt(U,d,m_minus,'tikh')^2;
        [rb_curv]=0;%l_curve(U,d,m_minus,'tikh')^2;
        %%% test statistics
        Um_minus=U'*m_minus;
        W_BPR=zeros(N2,1);  W_BPRI=zeros(N2,1);W_Quasi=zeros(N2,1);W_GCV=zeros(N2,1);W_CURV=zeros(N2,1);
        for jj=1:N2
            x_temp=X2(jj,:)' - 1/2*m_plus;
            x_tempU=x_temp'*U;
            [rz]=bpr(U,d,x_temp);
            [rzi]=covcopra(U,dd,x_temp,len1);
            [rz_gcv]=gcv(U,d,x_temp,'tikh')^2;
            [rz_quasi]=0;%quasiopt(U,d,x_temp,'tikh')^2;
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
        
        y2_hat=zeros(N2,1);           %Asym
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

    error_imp(ii)=mean(err_imp_iter);
    error_asy(ii)=mean(err_asy_iter);
    error_bpr(ii)=mean(err_bpr_iter);
    error_bpri(ii)=mean(err_bpri_iter);
    error_quasi(ii)=mean(err_quasi_iter);
    error_gcv(ii)=mean(err_gcv_iter);
    error_curv(ii)=mean(err_curv_iter);
    %%save complete data
%     err_asy_stat(:,ii)=err_asy_iter;
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
% plot(ntr, 100*error_curv,'k--'),
% plot(ntr, 100*error_quasi,'k-.'),
grid on
hold off
xlabel('# of samples'), ylabel('Avg. percentage error'), 
legend('Asym-RLDA','OII-LDA','GCV-R2LDA','BPR-R2LDA','COPRA-R2LDA');
% pth = "full path";
% spth = strcat(pth, filesep, "MNIST179_", num2str(stdn), '.fig')
% savefig(spth)

end

end

'Stop here!';

% for k=1:3
%     figure(k+6)
% pth='C:\Users\ahmedt\OneDrive - KAUST\R2LDA_Results_04Oct2020';
% stdn=k/10;
% stdnstr=num2str(10*stdn);
% figpth = strcat(pth, filesep, "MNIST79_", stdnstr, '.fig');
% epspth = strcat(pth, filesep, "MNIST79_", stdnstr, '.epsc')
% savefig(figpth);
% saveas(gcf, epspth)
% end
