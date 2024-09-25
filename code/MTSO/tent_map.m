function x = tent_map(n, r, x0)
% n: number of random numbers
% r: parameter
% x0: initial value
x(1) = x0;
for i = 2:n
    if x(i-1) < r
        x(i) = x(i-1)/r;
    else
        x(i) = (1-x(i-1))/(1-r);
    end
end
end
