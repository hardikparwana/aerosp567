rng('default')
clc
close all
clear

%% Plot initial data
X = [0.1 -0.9 0.2 0.8 -0.6 0.3 0.5 -0.5 -0.01 -0.9];
Y = [0.05 0.3 0.4 -0.3 0.3 -0.2 -0.84 0.85 -0.7 -0.9];

Sensor_0 = [3.39382006 3.2073034 3.39965035 3.68810201 2.96941623...
    2.99495501 3.94274928 2.7968011 3.34929734...
    3.9129616];

figure
scatter(X,Y,10,Sensor_0)
colormap(gca,'default')
colorbar
grid on
xlabel('X Position, (m)','Interpreter','latex')
ylabel('Y Position, (m)','Interpreter','latex')



%% Fit GP 0

input_data0 = [X' Y'];
output_data0 = Sensor_0';

kparams0 = [3.5, 6.2];
sigma0 = 0.2;

gpr0 = fitrgp(input_data0,output_data0,  'KernelFunction','squaredexponential','KernelFunction','squaredexponential','KernelParameters',kparams0,'Sigma',sigma0, 'FitMethod','sr','PredictMethod','fic','ActiveSetMethod','entropy',...
      'OptimizeHyperparameters','all','HyperparameterOptimizationOptions');

x_test = 2*rand(1000,1)-1; 
y_test = 2*rand(1000,1)-1;
[sensor_pred,~,intervals] = predict(gpr0,[x_test y_test]);

[X_grid,Y_grid] = meshgrid(linspace(-1,1,1000),linspace(-1,1,1000)) ;
lower_interval_grid = griddata(x_test,y_test,intervals(:,1),X_grid,Y_grid) ;
upper_interval_grid = griddata(x_test,y_test,intervals(:,2),X_grid,Y_grid) ;
prediction_grid = griddata(x_test,y_test,sensor_pred,X_grid,Y_grid) ;

figure
scatter(x_test,y_test,10,sensor_pred)
colormap(gca,'default')
colorbar
grid on
xlabel('X Position, (m)','Interpreter','latex')
ylabel('Y Position, (m)','Interpreter','latex')


figure
scatter3(input_data0(:,1),input_data0(:,2),output_data0,'r') % Observed data points
hold on
scatter3(x_test, y_test, sensor_pred,'g') % GPR predictions
mesh(X_grid,Y_grid,prediction_grid,'FaceAlpha','0.5')
mesh(X_grid,Y_grid,lower_interval_grid,'FaceAlpha','0.5')
mesh(X_grid,Y_grid,upper_interval_grid,'FaceAlpha','0.5')
scatter3(x_test, y_test, intervals(:,1), 'b')
scatter3(x_test, y_test, intervals(:,2), 'b')
hold off
title('GPR Fit of Noise-Free Observations')
legend({'Noise-free observations','GPR prediction points', 'GPR prediction mesh','95% Prediction Lower Bound Mesh','95% Prediction Upper Bound Mesh','95% Prediction Lower Bound Point','95% Prediction Upper Bound Point'},'Location','best')
xlabel('X Position, (m)','Interpreter','latex')
ylabel('Y Position, (m)','Interpreter','latex')
zlabel('Sensor Reading','Interpreter','latex')

keyboard 
%% Fit GP 1

 
  
ix_new = [4.500000000000000111e-01 7.199999999999999734e-01 -8.000000000000000444e-01 -8.000000000000000444e-01];
iy_new = [6.500000000000000222e-01 -6.400000000000000133e-01 -6.500000000000000222e-01 5.999999999999999778e-01 ];
sensor_new = [3.368086692765749124e+00 3.752857874639865532e+00 3.420476373601408770e+00 3.040183158829708354e+00];

input_data1 = [input_data0; ix_new' iy_new'];
output_data1 = [output_data0; sensor_new'];

figure
scatter(input_data1(:,1),input_data1(:,2),10,output_data1)
colormap(gca,'default')
colorbar
grid on
xlabel('X Position, (m)','Interpreter','latex')
ylabel('Y Position, (m)','Interpreter','latex')

kparams0 = [3.5, 6.2];
sigma0 = 0.2;

gpr1 = fitrgp(input_data1,output_data1, 'KernelFunction','squaredexponential','KernelFunction','squaredexponential','KernelParameters',kparams0,'Sigma',sigma0, 'FitMethod','sr','PredictMethod','fic','ActiveSetMethod','entropy',...
      'OptimizeHyperparameters','all','HyperparameterOptimizationOptions',...
      struct('MaxObjectiveEvaluations',60));

x_test = 2*rand(1000,1)-1; 
y_test = 2*rand(1000,1)-1;
[sensor_pred,~,intervals] = predict(gpr1,[x_test y_test]);

[X_grid,Y_grid] = meshgrid(linspace(-1,1,1000),linspace(-1,1,1000)) ;
lower_interval_grid = griddata(x_test,y_test,intervals(:,1),X_grid,Y_grid) ;
upper_interval_grid = griddata(x_test,y_test,intervals(:,2),X_grid,Y_grid) ;
prediction_grid = griddata(x_test,y_test,sensor_pred,X_grid,Y_grid) ;

figure
scatter(x_test,y_test,10,sensor_pred)
colormap(gca,'default')
colorbar
grid on
xlabel('X Position, (m)','Interpreter','latex')
ylabel('Y Position, (m)','Interpreter','latex')


figure
scatter3(input_data1(:,1),input_data1(:,2),output_data1,'r') % Observed data points
hold on
scatter3(x_test, y_test, sensor_pred,'g') % GPR predictions
mesh(X_grid,Y_grid,prediction_grid,'FaceAlpha','0.5')
mesh(X_grid,Y_grid,lower_interval_grid,'FaceAlpha','0.5')
mesh(X_grid,Y_grid,upper_interval_grid,'FaceAlpha','0.5')
scatter3(x_test, y_test, intervals(:,1), 'b')
scatter3(x_test, y_test, intervals(:,2), 'b')
hold off
title('GPR Fit of Noise-Free Observations')
legend({'Noise-free observations','GPR prediction points', 'GPR prediction mesh','95% Prediction Lower Bound Mesh','95% Prediction Upper Bound Mesh','95% Prediction Lower Bound Point','95% Prediction Upper Bound Point'},'Location','best')
xlabel('X Position, (m)','Interpreter','latex')
ylabel('Y Position, (m)','Interpreter','latex')
zlabel('Sensor Reading','Interpreter','latex')
