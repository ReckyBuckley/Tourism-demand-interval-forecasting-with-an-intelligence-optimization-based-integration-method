%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The following code are extracted from the reference below:
% https://authors.elsevier.com/sd/article/S2666-7207(22)00018-2
% Please cite this article as:
%  M. Mirrashid and H. Naderpour, Transit search: An optimization algorithm
%  based on exoplanet exploration; Results in Control and Optimization
%  (2022), doi: https://doi.org/10.1016/j.rico.2022.100127.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [LB_fin,UB_fin,Best_Solution]=MTSO(ns,dim,ub,lb,maxcycle,CostFunction,LB_int,UB_int,actual,h1,alpha)


%% Definition of the Algorithm Parameters
% ns = 10;                 % Number of Stars
SN = 20;                % Signal to Noise Ratio
% Note: (ns*SN)=Number of population for the TS algorithm

%% Transit Search Optimization Algorithm

disp('Transit Search is runing...')
%% Initialization
Empty.Location = [];
Empty.Cost = inf;
Galaxy_Center = repmat (Empty, 1, 1);
region = repmat (Empty, ns*SN, 1);
selested_regions = repmat (Empty, ns, 1);
Stars = repmat (Empty, ns, 1);
Stars_sorted = zeros(ns,1);
Ranks = 1:1:ns;
Stars_Ranks = zeros(ns,1);
Luminosity = zeros(ns,1);
Star_RanksNormal = zeros(ns,1);
Distance = zeros(ns,1);
Transit0 = zeros(ns,1);
SN_P = repmat (Empty, SN, 1);
Bests=region;


%% Galaxy Phase

% Initial Location of The Center of the Galaxy
Galaxy_Center.Location = unifrnd(lb,ub,1,dim);

Galaxy_Center.Cost = CostFunction(Galaxy_Center.Location,LB_int(1:h1),UB_int(1:h1),actual(1:h1),alpha);

% Galactic Habitate Zone of the Galaxy
% Generate Tent chaos map random vector
chaos = tent_map(dim, 2, 0.5);

for l = 1:(ns*SN)
    zone = randi(2);
    if zone ==1

         difference = chaos.*Galaxy_Center.Location - chaos.*unifrnd(lb,ub,1,dim);
    else

        difference = chaos.*Galaxy_Center.Location + chaos.*unifrnd(lb,ub,1,dim);
    end

    Noise = (chaos.^3).*unifrnd(lb,ub,1,dim);

    region(l).Location = Galaxy_Center.Location + difference - Noise;
    region(l).Location = max(region(l).Location, lb);
    region(l).Location = min(region(l).Location, ub);
    region(l).Cost = CostFunction(region(l).Location,LB_int(1:h1),UB_int(1:h1),actual(1:h1),alpha);
end

% Selection of Stars from the Galactic Habitate Zone of the Galaxy
[Sort,index]=sort([region.Cost]);
for i = 1:ns
    selested_regions(i) = region(index(1,i));
    for k = 1:SN
        zone = randi(2);
        if zone ==1
            difference = rand().*(selested_regions(i).Location)-rand().*(unifrnd(lb,ub,1,dim));
        else
            difference = rand().*(selested_regions(i).Location)+rand().*(unifrnd(lb,ub,1,dim));
        end
        Noise = ((rand(1,dim)).^3).*(unifrnd(lb,ub,1,dim));
        new.Location = selested_regions(i).Location + difference - Noise;
        new.Location = max(new.Location, lb);
        new.Location = min(new.Location, ub);
        new.Cost = CostFunction(new.Location,LB_int(1:h1),UB_int(1:h1),actual(1:h1),alpha);
        if new.Cost < Stars(i).Cost
            Stars(i) = new;
        end
    end
end

% Initial Location of the Best Planets (Start Point: Its Star)
Best_Planets = Stars;

% Specification of the Best Planet
[Sort,index]=sort([Best_Planets.Cost]);
Best_Planet = Best_Planets(index(1,1));

% Telescope Location
Telescope.Location = unifrnd(lb,ub,1,dim);

% Determination of the Luminosity of the Stars
for i = 1:ns
    Stars_sorted(i,1) = Stars(i).Cost;
end
Stars_sorted = sort (Stars_sorted);
for i = 1:ns
    for ii = 1:ns
        if Stars(i).Cost == Stars_sorted(ii,1)
            Stars_Ranks(i,1) = Ranks(1,ii);
            Star_RanksNormal(i,1) = (Stars_Ranks(i,1))./ns;
        end
    end
    Distance(i,1) = sum((Stars(i).Location-Telescope.Location).^2).^0.5;
    Luminosity(i,1) = Star_RanksNormal(i,1)/((Distance(i,1))^2);
end
Luminosity_new = Luminosity;
Stars2 = Stars;

%% Loops of the TS Algorithm
for it = 1:maxcycle
    
    %% Transit Phase
    Transit = Transit0;
    Luminosity = Luminosity_new;
    
    for i = 1:ns
        difference = (2*rand()-1).*(Stars(i).Location);
        Noise = ((rand(1,dim)).^3).*(unifrnd(lb,ub,1,dim));
        Stars2(i).Location = Stars(i).Location + difference - Noise;
        Stars2(i).Location = max(Stars2(i).Location, lb);
        Stars2(i).Location = min(Stars2(i).Location, ub);
        Stars2(i).Cost = CostFunction(Stars2(i).Location,LB_int(1:h1),UB_int(1:h1),actual(1:h1),alpha);
    end
    
    for i = 1:ns
        Stars_sorted(i,1) = Stars2(i).Cost;
    end
    Stars_sorted = sort (Stars_sorted);
    for i = 1:ns
        for ii = 1:ns
            if Stars2(i).Cost == Stars_sorted(ii,1)
                Stars_Ranks(i,1) = Ranks(1,ii);
                Star_RanksNormal(i,1) = (Stars_Ranks(i,1))./ns;
            end
        end
        Distance(i,1) = sum((Stars2(i).Location-Telescope.Location).^2).^0.5;
        Luminosity_new(i,1) = Star_RanksNormal(i,1)/((Distance(i,1))^2);
        if Luminosity_new(i,1) < Luminosity(i,1)
            Transit (i,1) = 1;      % Has transit been observed?  0 = No; 1 = Yes
        end
    end
    Stars = Stars2;
    
    %% Location Phase (Exploration)
    for i = 1:ns
        if Transit (i,1) == 1
            
            % Determination of the Location of the Planet
            Luminosity_Ratio = Luminosity_new(i,1)/Luminosity(i,1);
            Planet.Location = (rand().*Telescope.Location + Luminosity_Ratio.*Stars(i).Location)./2;
            
            for k = 1:SN
                zone = randi(3);
                if zone ==1
                    new.Location = Planet.Location - (2*rand()-1).*(unifrnd(lb,ub,1,dim));
                elseif zone ==2
                    new.Location = Planet.Location + (2*rand()-1).*(unifrnd(lb,ub,1,dim));
                else
                    new.Location = Planet.Location + (2.*rand(1,dim)-1).*(unifrnd(lb,ub,1,dim));
                end
                new.Location = max(new.Location, lb);
                new.Location = min(new.Location, ub);
                %                             new.Cost = CostFunction(new.Location);
                SN_P(k) = new;
            end
            SUM = 0;
            for k = 1:SN
                SUM = SUM+SN_P(k).Location;
            end
            new.Location = SUM./SN;
            new.Cost = CostFunction(new.Location,LB_int(1:h1),UB_int(1:h1),actual(1:h1),alpha);
            
            if new.Cost < Best_Planets(i).Cost
                Best_Planets(i) = new;
            end
            
        else  % No Transit observed: Neighbouring planets
            
            Neighbor.Location = (rand().*Stars(i).Location + rand().*(unifrnd(lb,ub,1,dim)))./2;
            
            for k = 1:SN
                zone = randi(3);
                if zone ==1
                    Neighbor.Location = Neighbor.Location - (2*rand()-1).*(unifrnd(lb,ub,1,dim));
                elseif zone ==2
                    Neighbor.Location = Neighbor.Location + (2*rand()-1).*(unifrnd(lb,ub,1,dim));
                else
                    Neighbor.Location = Neighbor.Location + (2.*rand(1,dim)-1).*(unifrnd(lb,ub,1,dim));
                end
                Neighbor.Location = max(Neighbor.Location, lb);
                Neighbor.Location = min(Neighbor.Location, ub);
                Neighbor.Cost = CostFunction (Neighbor.Location,LB_int(1:h1),UB_int(1:h1),actual(1:h1),alpha);
                SN_P(k) = Neighbor;
            end
            SUM = 0;
            for k = 1:SN
                SUM = SUM+SN_P(k).Location;
            end
            Neighbor.Location = SUM./SN;
            Neighbor.Cost = CostFunction (Neighbor.Location,LB_int(1:h1),UB_int(1:h1),actual(1:h1),alpha);
            
            if Neighbor.Cost < Best_Planets(i).Cost
                Best_Planets(i) = Neighbor;
            end
        end
    end
    
    %% Signal Amplification of the Best Planets (Exploitation)
    for i = 1:ns
        for k = 1:SN
            RAND = randi(2 );
            if RAND ==1
                Power = randi(SN*ns);
                Coefficient = 2*rand();
                Noise = ((rand(1,dim)).^Power).*(unifrnd(lb,ub,1,dim));
            else
                Power = randi(SN*ns);
                Coefficient = 2*rand();
                Noise = -((rand(1,dim)).^Power).*(unifrnd(lb,ub,1,dim));
            end
            
            chance = randi(2);
            if chance ==1
                new.Location = Best_Planets(i).Location - Coefficient.*Noise;
            else
                new.Location = (rand().*Best_Planets(i).Location) - Coefficient.*Noise;
            end
            new.Location = max(new.Location, lb);
            new.Location = min(new.Location, ub);
            new.Cost = CostFunction(new.Location,LB_int(1:h1),UB_int(1:h1),actual(1:h1),alpha);
            
            if new.Cost < Best_Planets(i).Cost
                Best_Planets(i) = new;
            end
        end
        if Best_Planets(i).Cost < Best_Planet.Cost
            Best_Planet = Best_Planets(i);
        end
    end
    
    % Results
    Bests(it)=Best_Planet;
end

Best_Cost = Bests(maxcycle).Cost;
Best_Solution = Bests(maxcycle).Location;

%% Figure
x=zeros(maxcycle,1);
y=zeros(maxcycle,1);
for i = 1:maxcycle
    y(i,1) = Bests(i).Cost;
    x(i,1) = i;
end
figure

LB_fin = LB_int(1:end,1)+Best_Solution(1);
UB_fin = UB_int(1:end,1)+Best_Solution(2);

end