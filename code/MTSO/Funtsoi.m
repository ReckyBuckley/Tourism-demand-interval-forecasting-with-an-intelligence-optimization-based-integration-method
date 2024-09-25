function o = Funtsoi(x,lb_int1,ub_int1,actual,alpha1)
alpha=1-alpha1;

Yt1 = lb_int1+x(1);
Yt2 = ub_int1+x(2);

uu=size(Yt1,1);

outside=0;
for kk=1:uu
    if actual(kk)<Yt1(kk) || actual(kk)>Yt2(kk)
    outside=outside+1;
end
outside;
end
inside=uu-outside; %Calculate the number of true values that fall within the interval

% PICP
PICP=inside/uu;

for kk=1:uu
    if actual(kk)<Yt1(kk)
        winkler(kk,1) = (Yt2(kk)-Yt1(kk))+2*(Yt1(kk)-actual(kk))/alpha;
    elseif actual(kk)<Yt2(kk) && actual(kk)>Yt1(kk)
        winkler(kk,1) = Yt2(kk)-Yt1(kk);
    elseif actual(kk)>Yt2(kk)
        winkler(kk,1) = (Yt2(kk)-Yt1(kk))+2*(actual(kk)-Yt2(kk))/alpha;
    end
end

Winkler = mean(winkler);

% NMPIW
width=Yt2-Yt1; 
MPIW=mean(width);
R=max(actual)-min(actual); 
NMPIW=MPIW/R;


% CWC
eta=10; % Penalty coefficient

if PICP<alpha
    gamma1=1;
else
    gamma1=0;
end

CWC=NMPIW*(1+gamma1*exp(-eta*(PICP-alpha)));

o1=(alpha-PICP);
o2=abs(CWC);
o=abs(o1)+o2;