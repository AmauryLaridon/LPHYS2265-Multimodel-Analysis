clear all
close all

Nmod = 15;

%Import files
% adjust your directory
direc = '/Users/ioulianikolskaia/Boulot/my_JOB/my_ENSEIGNEMENT/UCL/2023_UCL_LPHYS2265/EXERCISE_p3/CTL/'
file_list = ls(direc)
Filename = strsplit(file_list)
Filename = sort(Filename(1:Nmod))

%Get model names
for i = 1:Nmod
   Filename{i};
   z  = strsplit(Filename{i},'CTL_');
   zz = strsplit(z{2},'.txt');
   Modelname(i) = zz(1);
   %Modelname(i) = strrep(Modelname(i), '_', '\_')
end

doy_CTL  = zeros(365, Nmod);
hi_CTL   = zeros(365, Nmod);
Tsu_CTL  = zeros(365, Nmod);
hs_CTL   = zeros(365, Nmod);
Tw_CTL   = zeros(365, Nmod);

for i = 1:Nmod
    Modelname{i}
    zfile = [direc,'/',char(Filename(i))];
    A = importdata(zfile);
    
    if (  (~isequal(Modelname(i), {'COLDGATE'})) ...
        & (~isequal(Modelname(i), {'DOUGLACE'})) ...  
        & (~isequal(Modelname(i), {'MisterFreeze'})) ...  
        & (~isequal(Modelname(i), {'ICENBERG'})) );
   
       doy_CTL(:,i) = A(:,1);
       Tsu_CTL(:,i) = A(:,2);
       hi_CTL(:,i) = A(:,3);
       hs_CTL(:,i) = A(:,4);
       Tw_CTL(:,i) = A(:,5);
    end
    
    % Two model miss snow
    if (  (isequal(Modelname(i), {'COLDGATE'})) ...
       || (isequal(Modelname(i), {'ICENBERG'})) );       
       doy_CTL(:,i) = A(:,1);
       Tsu_CTL(:,i) = A(:,2);
       hi_CTL(:,i) = A(:,3);
       %hs_CTL(:,i) = A(:,4);
       Tw_CTL(:,i) = A(:,4);        
    end
    
    % This model has fucked up order of columns
    if (  (isequal(Modelname(i), {'DOUGLACE'}) ) )
               doy_CTL(:,i) = A(:,1);
       hi_CTL(:,i) = A(:,2);
       Tsu_CTL(:,i) = A(:,3);
       %hs_CTL(:,i) = A(:,4);
       Tw_CTL(:,i) = A(:,4);        
    end
    
    % This model has fucked up order of columns
    if (  (isequal(Modelname(i), {'MisterFreeze'}) ) )
       Tsu_CTL(:,i) = A(:,1);
       hi_CTL(:,i) = A(:,2);
       Tw_CTL(:,i) = A(:,3);        
    end
    
    % some screw up temperature units
    if ( isequal(Modelname(i), {'ARCTICTACOS'} ) || ...
         isequal(Modelname(i), {'MYSIM'} )       || ...
         isequal(Modelname(i), {'stolen_from_dirk_by_dominik'} ) );
         disp('beurps')
        Tsu_CTL(:,i) = Tsu_CTL(:,i) + 273.15;
    end
    
end

%----- Ice Thickness ----------%
figure
subplot(2,2,1);
hold on

set(gca,'ColorOrder',jet(Nmod))

%plot(hi_CTL(:,1), '--', 'LineWidth', 2)
for i = 1:Nmod
   plot(hi_CTL(:,i), '-', 'LineWidth', 2)
end

% plot(hi_CTL(:,6), '--', 'LineWidth', 2)
% for i = 7:9
%    plot(hi_CTL(:,i), '-', 'LineWidth', 2)
% end

ylim([0. 3.5]); xlim([0 365]);
xlabel('Days', 'FontSize', 16);  
ylabel('h_i [m]', 'FontSize', 16);  
legend(Modelname,'Interpreter', 'none');

hi_MU = [ 2.82 2.89 2.97 3.04 3.10 3.14 2.96 2.78 2.73 2.71 2.71 2.75 ]
plot(15:30:365,hi_MU,'k+', 'LineWidth', 2)

set(gca,'fontsize', 16)
set(gca, 'FontName', 'Myriad Pro')

%----- Snow Depth ----------%

subplot(2,2,2);
%figure
set(gca,'ColorOrder',jet(Nmod))
hold on

for i = 1:Nmod
   plot(hs_CTL(:,i), '-', 'LineWidth', 2)
end
ylim([0 0.4]); xlim([0 365]);
xlabel('Days', 'FontSize', 16);  
ylabel('h_s [m]', 'FontSize', 16);

set(gca,'fontsize', 16)
set(gca, 'FontName', 'Myriad Pro')


%legend(Modelname);

%------------- Temperature -------------
%figure
subplot(2,2,3);
set(gca,'ColorOrder',jet(Nmod))
hold on

for i = 1:Nmod
   plot(Tsu_CTL(:,i)-273.15, '-', 'LineWidth', 1.5)
end

ylim([-32. 0.]); xlim([0 365]);
xlabel('Days', 'FontSize', 16);  
ylabel('T_{su}(degC)', 'FontSize', 16);

set(gca,'fontsize', 16)
set(gca, 'FontName', 'Myriad Pro')


%legend(Modelname);

%----- Scatter Plot ----------%
subplot(2,2,4);
%figure
set(gca,'ColorOrder',jet(Nmod))
hold on
for i = 1:Nmod
   c_sym='*'
   %if ( mean(hs_CTL(:,i)) == 0 ); c_sym = '+'; end
   if ( isequal(Modelname(i), {'YBLAIRE'}) ); c_sym = '+'; end
   if ( isequal(Modelname(i), {'BRAILLE_ANE'}) ); c_sym = 'o'; disp('ICHI'); end
   plot(max(hs_CTL(:,i)), min(Tsu_CTL(:,i)-273.15), c_sym, 'LineWidth', 2, 'MarkerSize', 20)%, 'Color', jet(1))
end
%ylim([-32. -23.]); xlim([2.8 3.6]);
xlabel('h_s^{max} (m)', 'FontSize', 16);  
ylabel('T_{su}^{min}(degC)', 'FontSize', 16);

set(gca,'fontsize', 16)
set(gca, 'FontName', 'Myriad Pro')

%--- PROCESSES

% 1) Growth rate versus thickness

figure; 

subplot(2,2,1); hold on

set(gca,'ColorOrder',jet(Nmod))

for i = 1:Nmod
   c_sym='*';
   if ( mean(hs_CTL(:,i)) == 0 ); c_sym = '+'; end
   
   hi_max = max(hi_CTL(:,i));
   hi_min = min(hi_CTL(:,i));
   hi_min = min(hi_CTL(:,i));
   amplitude = hi_max - hi_min;
   
   plot(hi_min, amplitude, c_sym, 'LineWidth', 2, 'MarkerSize', 20)%, 'Color', jet(1))
end
%ylim([-32. -23.]); xlim([2.8 3.6]);
xlabel('h_i^{min} (m)', 'FontSize', 16);  
ylabel('Seasonal cycle amplitude of ice thickness (m)', 'FontSize', 16);

set(gca,'fontsize', 16)
set(gca, 'FontName', 'Myriad Pro')

%--- Min temperature VS snow depth

subplot(2,2,2);
%figure
set(gca,'ColorOrder',jet(Nmod))
hold on
for i = 1:Nmod
   c_sym='*'
   if ( mean(hs_CTL(:,i)) == 0 ); c_sym = '+'; end
   plot(max(hs_CTL(:,i)), min(Tsu_CTL(:,i)-273.15), c_sym, 'LineWidth', 2, 'MarkerSize', 20)%, 'Color', jet(1))
end
%ylim([-32. -23.]); xlim([2.8 3.6]);
xlabel('h_s^{max} (m)', 'FontSize', 16);  
ylabel('T_{su}^{min}(degC)', 'FontSize', 16);

set(gca,'fontsize', 16)
set(gca, 'FontName', 'Myriad Pro')


%--- Max thickness VS max snow depth

subplot(2,2,3);
%figure
set(gca,'ColorOrder',jet(Nmod))
hold on
for i = 1:Nmod
   c_sym='*'
   if ( mean(hs_CTL(:,i)) == 0 ); c_sym = '+'; end
   plot(max(hs_CTL(:,i)), max(hi_CTL(:,i)), c_sym, 'LineWidth', 2, 'MarkerSize', 20)%, 'Color', jet(1))
end
%ylim([-32. -23.]); xlim([2.8 3.6]);
xlabel('h_s^{max} (m)', 'FontSize', 16);  
ylabel('h_i^{max}(m)', 'FontSize', 16);

set(gca,'fontsize', 16)
set(gca, 'FontName', 'Myriad Pro')

subplot(2,2,4);
%figure
set(gca,'ColorOrder',jet(Nmod))
hold on
for i = 1:Nmod
   c_sym='*'
   if ( mean(hs_CTL(:,i)) == 0 ); c_sym = '+'; end
   plot(max(hi_CTL(:,i)), min(Tsu_CTL(:,i)-273.15), c_sym, 'LineWidth', 2, 'MarkerSize', 20)%, 'Color', jet(1))
end
%ylim([-32. -23.]); xlim([2.8 3.6]);
xlabel('h_i^{max} (m)', 'FontSize', 16);  
ylabel('T_{su}^{min}(degC)', 'FontSize', 16);

set(gca,'fontsize', 16)
set(gca, 'FontName', 'Myriad Pro')

% {'BRAILLE\_ANE'}    {'Florina'}    {'HS'}    {'KvdH'}    {'LF'}    {'ML'}    {'SUPERN\_ICE'}    {'SurprIce'}    {'YBLAIRE'} 
% for PR script
% hsmax(1) = max(hs_CTL(:,2)); %bryan
% hsmax(2) = max(hs_CTL(:,3)); %florina
% hsmax(3) = max(hs_CTL(:,4)); %hs
% hsmax(4) = max(hs_CTL(:,6)); % KvdH
% hsmax(5) = max(hs_CTL(:,7)); % LF
% hsmax(6) = max(hs_CTL(:,8)); % ML
% hsmax(7) = max(hs_CTL(:,11)); % SUPERNICE
% hsmax(8) = max(hs_CTL(:,12)); % SurprIce
% hsmax(9) = max(hs_CTL(:,13)); % YBLAIRE
% 
% hsmax

% SAVE MATFILE
model = Modelname;
hi = hi_CTL;
hs = hs_CTL;
Tsu = Tsu_CTL;
Tw = Tw_CTL;
doy = 1:365;

savefile = [ direc, '../CTL.mat' ]

save(savefile,'Nmod', 'model', 'hi', 'hs', 'Tsu', 'Tw', 'doy')

