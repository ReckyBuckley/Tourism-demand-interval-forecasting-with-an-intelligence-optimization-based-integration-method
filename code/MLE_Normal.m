%% MLE for Normal
clc
clear

data = table2array(readtable("main\data\forecast results\Forecasting sequence of ARIMAX for Jiuzhaigou.xlsx",sheet=1));

arimax = data(:,8);
actual = data(:,1);
u=size(arimax,1);

[f,X]=ecdf(arimax);
k=size(X,1);
f2=f';

phat=mle('Normal',X);
mu=phat(1);
sigma=phat(2);
f1=@(t)normpdf(t,mu,sigma);

for i=1:k
    F1(i)=normcdf(X(i),mu,sigma);
    ess1(i)=(F1(i)-f2(i))^2;
    tss1(i)=(f2(i)-mean(f2))^2;
end
% R square
RFmle=1-(sum(ess1)/sum(tss1));