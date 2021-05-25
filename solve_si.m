 function [s_i] = solve_si(i,X)
    eps = 0.05;
    x_i=X(:,i);
    f = ones(1,size(X,2));
    f = [f,f];
    A = [X;-X];
    A = [A,-A];
    b = [x_i+eps,eps-x_i];
    Aeq = [ones(1,size(X,2))];
    s_ii = zeros(1,size(X,2));
    s_ii(i) = 1;
    Aeq = [Aeq;s_ii];
    Aeq = [Aeq,-Aeq];
    beq = [1,0];
    [y,z]=linprog(f,A,b,Aeq,beq,zeros(1,size(f,2)));
    s_i = y(1:size(X,2))-y(size(X,2)+1:end);
 end