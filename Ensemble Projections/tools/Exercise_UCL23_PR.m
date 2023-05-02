clear all
close all

Nmod = 14;

% Import files
direc = '/Users/ioulianikolskaia/Boulot/my_JOB/my_ENSEIGNEMENT/UCL/2023_UCL_LPHYS2265/EXERCISE_p3/PR/'
file_list = ls (direc);
Filename = strsplit(file_list);
Filename = sort(Filename(1:3*Nmod));

% Get model names
for i = 1:3*Nmod
   z  = strsplit(Filename{i},'.txt');
   zz = strsplit(z{1},'_');
   Pert(i) = zz(1);
   zzz = strsplit(z{1},[Pert{i},'_']);
   Modelname(i) = zzz(2);
   Modelname(i) = strrep(Modelname(i), '_', '\_');
end

% Maximum snow depth from control run
% hs_ctl = [ 0.0818 0.0 0.3907 0. 0.3943 ...
%            0.3454 0.4007 0.3984 0.0776 ...
%            0.3896 0.3907 0.3907 0.3958 0.3991 ...
%            0.0788 0.3920 ];
       
%hs_ctl = [  0    0.3889    0.3983    0.3915    0.3950         0    0.3907    0.0778    0.3920 ]
  
% Extract model output
yr      = zeros(100, Nmod);
himin   = zeros(100, Nmod,3);
himean  = zeros(100, Nmod,3);
himax   = zeros(100, Nmod,3);
hsmax   = zeros(100, Nmod,3);
Tsumin  = zeros(100, Nmod,3);

i_mod = 0;
for i = 1:Nmod*3
    
    i
   
    i_mod = i_mod + 1;
    if ( i_mod > Nmod ); i_mod = 1; end
    Modelname{i_mod}
    Pert{i}
    zfile = [direc,'/',char(Filename(i))]
    A = importdata(zfile);
    
%    i_test = isequal(Modelname{i_mod},'Vemund');
    
     if ( isequal(Pert{i},'PR03' ) ); k = 1; end
     if ( isequal(Pert{i},'PR06' ) ); k = 2; end
     if ( isequal(Pert{i},'PR12' ) ); k = 3; end
     
%      if ( i_test == 0);   
%         yr(:,i_mod) = A(:,1);
%         himin(:,i_mod,k) = A(:,2);
%         himean(:,i_mod,k) = A(:,3);
%         himax(:,i_mod,k) = A(:,4);
%         Tsumin(:,i_mod,k) = A(:,5);
%      else

    if (  (~isequal(Modelname(i), {'COLDGATE'})) ...
        & (~isequal(Modelname(i), {'DOUGLACE'})) ...  
        & (~isequal(Modelname(i), {'MisterFreeze'})) ...  
        & (~isequal(Modelname(i), {'ICENBERG'})) );
        yr(:,i_mod)       = A(:,1);
        himin(:,i_mod,k)  = A(:,2);
        himean(:,i_mod,k) = A(:,3);
        himax(:,i_mod,k)  = A(:,4);
        hsmax(:,i_mod,k)  = A(:,5);
        Tsumin(:,i_mod,k) = A(:,6);
    end
    
    if (  (isequal(Modelname(i), {'COLDGATE'})) ...
       || (isequal(Modelname(i), {'DOUGLACE'})) ... 
       || (isequal(Modelname(i), {'MisterFreeze'})) ... 
       || (isequal(Modelname(i), {'ICENBERG'})) );      
        yr(:,i_mod)       = A(:,1);
        himin(:,i_mod,k)  = A(:,2);
        himean(:,i_mod,k) = A(:,3);
        himax(:,i_mod,k)  = A(:,4);
        hsmax(:,i_mod,k)  = 0.;
        Tsumin(:,i_mod,k) = A(:,5);
    end
    
        
        if ( isequal(Modelname(i), {'MYSIM'} ) );

           Tsumin(:,i) = Tsumin(:,i) + 273.15

        end
%     end
       
end


% Rate of ice thickness change (m/yr)
dh_dy = zeros(Nmod,3);
for i_mod = 1:Nmod
    for i_lw = 1:3
       zz = polyfit(yr(:,i_mod),himin(:,i_mod,i_lw),1);
       dh_dy(i_mod,i_lw) = zz(1);
    end
end

%-------------------------------------------------------------------------
% PLOTS
%-------------------------------------------------------------------------

i_plot  = ones(Nmod,1)'
%i_plot = [ 1, 1, 1 , 1, 1, 1, 1, 1, 1, 1, 1] %,1,1,1,1,1,1]%, 1,  1, 1, 1, 1, 1,   1, 1, 1, 1, 1,  1, 1, 1, 1]; % 
ls     = ['-','-','-', '-' ,'-', '-', '-', '-', '-', '-', '-','-','-','-']%,'-','-','-' ]%,'-', '-','-','-','-','-', '-','-', '-', '-', '-',  '-', '-', '-', '-' ];
lw     = [ 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3 ]% ,3, 3, 3,3,3,3]%, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3 ]%
%ls = ['-','-','-','-']
%lw = [3,3,3,3]

%------------------------------------
% Rate of ice retreat versus longwave
%------------------------------------

figure
dlw = zeros(Nmod,3);
dlw(:,1) = 3;
dlw(:,2) = 6;
dlw(:,3) = 12;

plot(dlw,dh_dy*100,'+','MarkerSize', 10, 'LineWidth', 3);
xlim([0 14]); ylim([-3 0]);
xlabel('\Delta F_{LW}','FontSize', 16); ylabel('dh_{min}/dt [cm/yr]','FontSize', 16)
legend('+3W/m^2', '+6W/m^2', '+12W/m^2')


set(gca,'fontsize', 16)
set(gca, 'FontName', 'Myriad Pro')

%-----------------------------------------------
%Rate of ice retreat versus control snow depth
%-----------------------------------------------
%--- currently commented !!! maybe uncomment if we can
% figure; hold on
% plot(hs_ctl,dh_dy(:,1)*100,'+','MarkerSize', 10, 'LineWidth', 3)
% plot(hs_ctl,dh_dy(:,2)*100,'+','MarkerSize', 10, 'LineWidth', 3)
% plot(hs_ctl,dh_dy(:,3)*100,'+','MarkerSize', 10, 'LineWidth', 3)
% xlim([-0.1 0.5]); ylim([-3 0]);
% xlabel('h_s','FontSize', 16); ylabel('dh_{min}/dt [cm/yr]','FontSize', 16')
% legend('+3W/m^2', '+6W/m^2', '+12W/m^2')
% 
% set(gca,'fontsize', 16)
% set(gca, 'FontName', 'Myriad Pro')

%----------------------------------------------------------------
% Ice thickness at year one versus snow depth in the control run
%-----------------------------------------------------------------
% figure; hold on
% wplot(hs_ctl,himax(50,:,1),'+','MarkerSize', 10, 'LineWidth', 3)
% plot(hs_ctl,himax(50,:,2),'+','MarkerSize', 10, 'LineWidth', 3)
% plot(hs_ctl,himax(50,:,3),'+','MarkerSize', 10, 'LineWidth', 3)
% xlim([-0.1 0.5]); ylim([0.5 3.5]);
% xlabel('h_s (m)','FontSize', 16); ylabel('himax, year 50 (m)','FontSize', 16')

%----------------------------------------------------------------
% Mean ice thickness for all models 
%----------------------------------------------------------------
figure
%subplot(2,2,1);
hold on
%set(gca,'ColorOrder',jet(Nmod))

% Different models, 1 scenario
for i_mod = 1:Nmod
    if ( i_plot(i_mod) == 1 );
       plot(himean(:,i_mod,3), ls(i_mod), 'LineWidth', lw(i_mod))
       %plot(himax(:,i_mod,3), 'LineWidth', 1)
       %plot(himin(:,i_mod,3), 'LineWidth', 1)
       %plot(himax(:,i_mod,3), colors{i_mod}, 'LineWidth', 1)
       %plot(himean(:,i_mod,2), colors(i_mod), 'LineWidth', 2)
       %plot(himean(:,i_mod,3), colors(i_mod), 'LineWidth', 2)
    end
end
xlim([0 100]); ylim([0 4]);
xlabel('years','FontSize', 16); ylabel('himean','FontSize', 16)
title('+12 W/m^2')

legend(Modelname(1:Nmod));

%plot(mean(himean(:,find(i_plot == 1),3),2), 'k-', 'Linewidth', 10) % mean model

xlim([0 100]); ylim([0 4]);
xlabel('Years', 'FontSize', 16);  
ylabel('h_i [m]', 'FontSize', 16);

set(gca,'fontsize', 16)
set(gca, 'FontName', 'Myriad Pro')


%----------------------------------------------------------------
% Minimum temperature time series
%----------------------------------------------------------------
figure;
hold on;
for i_mod = 1:Nmod;
    if ( i_plot(i_mod) == 1 );
       plot(Tsumin(:,i_mod,3)-273.15, ls(i_mod), 'LineWidth', lw(i_mod))
    end
end
xlim([0 100]); ylim([-34 -20]);
xlabel('years','FontSize', 16); ylabel('Tsu_{min} (°C)','FontSize', 16)

set(gca,'fontsize', 16)
set(gca, 'FontName', 'Myriad Pro')


%----------------------------------------------------------------
% Minimum and max time series
%----------------------------------------------------------------

%i_plot = [ 1, 1, 1, 1, 1, 1, 1, 1, 1]; % 
%i_plot = [ 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1 ]; % remove outlier
figure
%subplot(2,2,1);
hold on
%set(gca,'ColorOrder',jet(Nmod))
%i_plot = [ 1, 1, 1, 1, 1, 1, 1, 1, 1]; % 

% Different models, 1 scenario
plot(mean(himin(:, find(i_plot == 1) ,3),2), 'r-', 'Linewidth', 10)
plot(mean(himax(:, find(i_plot == 1),3),2), 'b-', 'Linewidth', 10)

legend('h_{min}', 'h_{max}')
for i_mod = 1:Nmod
    if ( i_plot(i_mod) == 1 );
    plot(himin(:,i_mod,3), '--', 'LineWidth', 0.1, 'Color',[1 .5 0])
    plot(himax(:,i_mod,3), '--', 'LineWidth', 0.1, 'Color',[1 .5 1])
    end
end

xlim([0 100]); ylim([0 3.5]);
xlabel('Years', 'FontSize', 16);  
ylabel('h_i [m]', 'FontSize', 16);  


%----------------------------------------------------------------
% Minimum for all scenarios -----%
%----------------------------------------------------------------

figure
%subplot(2,2,1);
hold on
%set(gca,'ColorOrder',jet(Nmod))

% Different models, 1 scenario
plot(mean(himin(:,find(i_plot == 1),1),2), 'b-', 'Linewidth', 10)
plot(mean(himin(:,find(i_plot == 1),2),2), 'g-', 'Linewidth', 10)
plot(mean(himin(:,find(i_plot == 1),3),2), 'r-', 'Linewidth', 10)
legend('+3 W/m^2', '+6 W/m^2', '+12 W/m^2')

for i_mod = 1:Nmod
    if ( i_plot(i_mod) == 1 );
    plot(himin(:,i_mod,1), 'b', 'LineWidth', 0.5)
    plot(himin(:,i_mod,2), 'g', 'LineWidth', 0.5)
    plot(himin(:,i_mod,3), 'r', 'LineWidth', 0.5)
    end
end

 
xlim([0 100]); ylim([0 3.5]);
xlabel('Years', 'FontSize', 16);  
ylabel('h_i (minimum) [m]', 'FontSize', 16);  

set(gca,'fontsize', 16)
set(gca, 'FontName', 'Myriad Pro')

%-------------------------


model  = Modelname;

year = 1:100;
 
savefile = [ direc, '../PR.mat' ]
 
save(savefile, 'Nmod', 'model', 'year', 'himin', 'himax', 'himean', 'hsmax', 'Tsumin')

