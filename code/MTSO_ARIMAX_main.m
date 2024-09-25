%% MTSO-ARIMAX
clc
clear

data = table2array(readtable("main\data\forecast results\Forecasting sequence of ARIMAX for Jiuzhaigou.xlsx",sheet=1));

arimax = data(:,8);
actual = data(:,1);
u=size(arimax,1);

ALPHA = [0.05,0.10,0.15];
for k = 1:3
alpha = ALPHA(k);
phat=mle('Lognormal',arimax);
mu1=phat(1);
sigma1=phat(2);
[~,pCov] = lognlike([mu1,sigma1],arimax);
p = logncdf(arimax,mu1,sigma1);
[~,LB_int(:,1),UB_int(:,1)] = logninv(p,mu1,sigma1,pCov,alpha);


%Parameter setting
popsize=100;      %Population size (Too large value will slow down the convergence speed)
MaxFEs=200;       %Maximum evaluation number (MaxFEs=dimension*10000)
Func1=@Funtsoi;   %Objective function test suite
dimension1 = 2;
xmax1(1,dimension1-1:dimension1)=50000*ones(1,2);   %Upper boundary
xmin1(1,dimension1-1:dimension1)=-50000*ones(1,2);  %Lower boundary

h1=size(actual,1);
[LB_fin1(:,1),UB_fin1(:,1),metric2(:,:)]=MTSO(popsize,dimension1,xmax1,xmin1,MaxFEs,Func1,LB_int,UB_int,actual,h1,alpha);


Yt1 = max(LB_fin1(1:end,1),0);
Yt2 = UB_fin1(1:end,1);
u=size(Yt1,1);

% Calculate metric
alpha1 = 1-alpha;
outside=0;
for kk=1:u
    if actual(kk)<Yt1(kk) || actual(kk)>Yt2(kk)
    outside=outside+1;
end
outside;
end
inside=u-outside; %Calculate the number of true values that fall within the interval

% PICP
PICP=inside/u;

% NMPIW
width=Yt2-Yt1;
MPIW=mean(width);
R=max(actual)-min(actual);
NMPIW=MPIW/R;

% CWC
eta=10; % Penalty coefficient
if PICP<alpha1
    gamma1=1;
else
    gamma1=0;
end

ACE = PICP - alpha1;

CWC=NMPIW*(1+gamma1*exp(-eta*(PICP-alpha1)));

matrix2(k,:) = [PICP,NMPIW,CWC]; % Metrics

predict1(:,2*k-1:2*k) = [Yt1,Yt2]; % Forecasting interval
end
