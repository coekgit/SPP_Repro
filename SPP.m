function [W] = SPP(X,Eigen_NUM)
[MatrixS] = solve_Mat_S(X);

S_b = MatrixS+MatrixS'-MatrixS'*MatrixS;
Mat1 = X*S_b*X';
Mat2 = X*X';
[W,Gen_Value]=Find_K_Max_Gen_Eigen(Mat1,Mat2,Eigen_NUM);
end
