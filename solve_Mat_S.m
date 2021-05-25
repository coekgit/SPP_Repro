function [MatrixS] = solve_Mat_S(Train_SET)
[n_rows,n_columns] = size(Train_SET);
MatrixS = zeros(n_columns,n_columns);
delete(gcp('nocreate'))
parpool(4);
parfor idx = 1:n_columns
    idx
    [s_i] = solve_si(idx,Train_SET);
    MatrixS(:,idx) = s_i;
end
