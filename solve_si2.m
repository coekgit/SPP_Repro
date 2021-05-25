function [s_i] = solve_si2(i,X)
[Eigen_NUM ,Train_NUM] = size(X);
% Eigen_NUM = Train_NUM;
MatrixS=zeros(Train_NUM,Train_NUM);
k=i;
A=zeros(Eigen_NUM,Train_NUM-1);
if k==1
   y=X(:,1);
   A=X(:,2:Train_NUM);
   x0=inv(A'*A)*A'*y;
   %solve the L1_minimization through y=Ax
   xp=l1eq_pd(x0,A,[],y,1e-3);
   xp=xp/norm(xp,1);
   s_i=[0 xp'];
else
   y=X(:,k);
   A(:,1:k-1)=X(:,1:k-1);
   A(:,k:Train_NUM-1)=X(:,k+1:Train_NUM);
   x0=inv(A'*A)*A'*y;
   xp=l1eq_pd(x0,A,[],y,1e-3);
   xp=xp/norm(xp,1);
   %xp1=xp(1:k-1);
   %xp2=xp(k:Train_NUM-1);
   %MatrixS(:,k)=[xp1' 0 xp2'];
   s_i=[xp(1:k-1)' 0 xp(k:Train_NUM-1)'];
end
clear A;
end