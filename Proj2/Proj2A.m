%% Drew Hanover Proj 2A
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

%% Implement naive GPR manually


% squared exponential kernel function
kfcn = @(XN,XM,theta) (exp(theta(2))^2)*exp(-(pdist2(XN,XM).^2)/(2*exp(theta(1))^2));

% Guess values for theta and noise
theta0 = [0.1;0.5];
NOISE = 0.01;

covar = kfcn([X;Y]',[X;Y]',theta0);
invcovar = (covar+NOISE)^-1;
pred_point = [X;Y]';
vec_pred = kfcn(pred_point, [X;Y]',theta0);
pred_mean = vec_pred*(invcovar*Sensor_0');

cov_predict_pre = kfcn(pred_point,pred_point, theta0);
cov_predict_up = vec_pred*invcovar*vec_pred';
pred_cov = cov_predict_pre - cov_predict_up;


figure
plot(Sensor_0,'r.','MarkerSize',10);
hold on
plot(pred_mean,'k','LineWidth',1);
xlabel('Input Pair','Interpreter', 'latex');
ylabel('Output Reading','Interpreter', 'latex');
legend({'data','Initial Fit'},'Location','Best','Interpreter', 'latex');
hold off


%% Train Model and Plot Hyperparam Optimization
tbl = readtable('init_data.txt','FileType','text','ReadVariableNames',true);
gprMdl1 = fitrgp(tbl,'Sensor','KernelFunction','squaredexponential');
gprMdl2 = fitrgp(tbl,'Sensor',...
      'FitMethod','sr','PredictMethod','fic','ActiveSetMethod','entropy',...
      'OptimizeHyperparameters','all','HyperparameterOptimizationOptions',...
      struct('MaxObjectiveEvaluations',30,'UseParallel',true));
ypred = resubPredict(gprMdl2);
train_Loss = resubLoss(gprMdl2)

ypred1 = resubPredict(gprMdl1);
ypred2 = resubPredict(gprMdl2);

figure
plot(tbl.Sensor,'r.','MarkerSize',10);
hold on
plot(ypred1,'b');
plot(ypred2,'k','LineWidth',1);
xlabel('Input Pair','Interpreter', 'latex');
ylabel('Output Reading','Interpreter', 'latex');
legend({'data','Initial Fit','Optimized Fit'},'Location','Best','Interpreter', 'latex');
title('Impact of Optimization','Interpreter', 'latex');
hold off


%% Test Model and Plot
X_test = 2*rand(1000,1)-1;
Y_test = 2*rand(1000,1)-1;
[sensor_pred, ~, intervals] = predict(gprMdl2,[X_test Y_test]);


[X_grid,Y_grid] = meshgrid(linspace(-1,1,1000),linspace(-1,1,1000)) ;
lower_interval_grid = griddata(X_test,Y_test,intervals(:,1),X_grid,Y_grid) ;
upper_interval_grid = griddata(X_test,Y_test,intervals(:,2),X_grid,Y_grid) ;
prediction_grid = griddata(X_test,Y_test,sensor_pred,X_grid,Y_grid) ;


figure
scatter(X_test,Y_test,10,sensor_pred)
colormap(gca,'default')
colorbar
grid on
xlabel('X Position, (m)','Interpreter','latex')
ylabel('Y Position, (m)','Interpreter','latex')


figure
scatter3(tbl.X,tbl.Y,tbl.Sensor,'r') % Observed data points
hold on
scatter3(X_test, Y_test, sensor_pred,'g') % GPR predictions
mesh(X_grid,Y_grid,prediction_grid,'FaceAlpha','0.5')
mesh(X_grid,Y_grid,lower_interval_grid,'FaceAlpha','0.5')
mesh(X_grid,Y_grid,upper_interval_grid,'FaceAlpha','0.5')
scatter3(X_test, Y_test, intervals(:,1), 'b')
scatter3(X_test, Y_test, intervals(:,2), 'b')
hold off
title('GPR Fit of Noise-Free Observations')
legend({'Noise-free observations','GPR prediction points', 'GPR prediction mesh','95% Prediction Lower Bound Mesh','95% Prediction Upper Bound Mesh','95% Prediction Lower Bound Point','95% Prediction Upper Bound Point'},'Location','best')
xlabel('X Position, (m)','Interpreter','latex')
ylabel('Y Position, (m)','Interpreter','latex')
zlabel('Sensor Reading','Interpreter','latex')
%% ReTrain Model with Request 1 and Plot Hyperparam Optimization
tbl = readtable('init_data_with_request_1.txt','FileType','text','ReadVariableNames',true);
gprMdl1 = fitrgp(tbl,'Sensor','KernelFunction','squaredexponential');
gprMdl2 = fitrgp(tbl,'Sensor',...
      'FitMethod','sr','PredictMethod','fic','ActiveSetMethod','entropy',...
      'OptimizeHyperparameters','all','HyperparameterOptimizationOptions',...
      struct('MaxObjectiveEvaluations',60,'UseParallel',true));
ypred = resubPredict(gprMdl2);
train_Loss = resubLoss(gprMdl2)

ypred1 = resubPredict(gprMdl1);
ypred2 = resubPredict(gprMdl2);

figure
plot(tbl.Sensor,'r.','MarkerSize',10);
hold on
plot(ypred1,'b');
plot(ypred2,'k','LineWidth',1);
xlabel('Input Pair','Interpreter', 'latex');
ylabel('Output Reading','Interpreter', 'latex');
legend({'data','Initial Fit','Optimized Fit'},'Location','Best','Interpreter', 'latex');
title('Impact of Optimization','Interpreter', 'latex');
hold off

figure
scatter(tbl.X,tbl.Y,10,tbl.Sensor)
colormap(gca,'default')
colorbar
grid on
xlabel('X Position, (m)','Interpreter','latex')
ylabel('Y Position, (m)','Interpreter','latex')


%% Test Model and Plot with data from Request 1
X_test = 2*rand(1000,1)-1;
Y_test = 2*rand(1000,1)-1;
[sensor_pred, ~, intervals] = predict(gprMdl2,[X_test Y_test]);


[X_grid,Y_grid] = meshgrid(linspace(-1,1,1000),linspace(-1,1,1000)) ;
lower_interval_grid = griddata(X_test,Y_test,intervals(:,1),X_grid,Y_grid) ;
upper_interval_grid = griddata(X_test,Y_test,intervals(:,2),X_grid,Y_grid) ;
prediction_grid = griddata(X_test,Y_test,sensor_pred,X_grid,Y_grid) ;



figure
scatter(X_test,Y_test,10,sensor_pred)
colormap(gca,'default')
colorbar
grid on
xlabel('X Position, (m)','Interpreter','latex')
ylabel('Y Position, (m)','Interpreter','latex')


figure
scatter3(tbl.X,tbl.Y,tbl.Sensor,'r') % Observed data points
hold on
scatter3(X_test, Y_test, sensor_pred,'g') % GPR predictions
mesh(X_grid,Y_grid,prediction_grid,'FaceAlpha','0.5')
mesh(X_grid,Y_grid,lower_interval_grid,'FaceAlpha','0.5')
mesh(X_grid,Y_grid,upper_interval_grid,'FaceAlpha','0.5')
scatter3(X_test, Y_test, intervals(:,1), 'b')
scatter3(X_test, Y_test, intervals(:,2), 'b')
hold off
title('GPR Fit of Noise-Free Observations')
legend({'Noise-free observations','GPR prediction points', 'GPR prediction mesh','95% Prediction Lower Bound Mesh','95% Prediction Upper Bound Mesh','95% Prediction Lower Bound Point','95% Prediction Upper Bound Point'},'Location','best')
xlabel('X Position, (m)','Interpreter','latex')
ylabel('Y Position, (m)','Interpreter','latex')
zlabel('Sensor Reading','Interpreter','latex')

%% ReTrain Model with Request 1 and 2, and Plot Hyperparam Optimization
tbl = readtable('init_data_with_request_1and2.txt','FileType','text','ReadVariableNames',true);
gprMdl1 = fitrgp(tbl,'Sensor','KernelFunction','squaredexponential');
gprMdl2 = fitrgp(tbl,'Sensor',...
      'FitMethod','sr','PredictMethod','fic','ActiveSetMethod','entropy',...
      'OptimizeHyperparameters','all','HyperparameterOptimizationOptions',...
      struct('MaxObjectiveEvaluations',180,'UseParallel',true));
ypred = resubPredict(gprMdl2);
train_Loss = resubLoss(gprMdl2)

ypred1 = resubPredict(gprMdl1);
ypred2 = resubPredict(gprMdl2);

figure
plot(tbl.Sensor,'r.','MarkerSize',10);
hold on
plot(ypred1,'b');
plot(ypred2,'k','LineWidth',1);
xlabel('Input Pair','Interpreter', 'latex');
ylabel('Output Reading','Interpreter', 'latex');
legend({'data','Initial Fit','Optimized Fit'},'Location','Best','Interpreter', 'latex');
title('Impact of Optimization','Interpreter', 'latex');
hold off

figure
scatter(tbl.X,tbl.Y,10,tbl.Sensor)
colormap(gca,'default')
colorbar
grid on
xlabel('X Position, (m)','Interpreter','latex')
ylabel('Y Position, (m)','Interpreter','latex')


%% Test Model and Plot with data from Request 1 and 2
X_test = 2*rand(1000,1)-1;
Y_test = 2*rand(1000,1)-1;
[sensor_pred, ~, intervals] = predict(gprMdl2,[X_test Y_test]);


[X_grid,Y_grid] = meshgrid(linspace(-1,1,1000),linspace(-1,1,1000)) ;
lower_interval_grid = griddata(X_test,Y_test,intervals(:,1),X_grid,Y_grid) ;
upper_interval_grid = griddata(X_test,Y_test,intervals(:,2),X_grid,Y_grid) ;
prediction_grid = griddata(X_test,Y_test,sensor_pred,X_grid,Y_grid) ;



figure
scatter(X_test,Y_test,10,sensor_pred)
colormap(gca,'default')
colorbar
grid on
xlabel('X Position, (m)','Interpreter','latex')
ylabel('Y Position, (m)','Interpreter','latex')


figure
scatter3(tbl.X,tbl.Y,tbl.Sensor,'r') % Observed data points
hold on
scatter3(X_test, Y_test, sensor_pred,'g') % GPR predictions
mesh(X_grid,Y_grid,prediction_grid,'FaceAlpha','0.5')
mesh(X_grid,Y_grid,lower_interval_grid,'FaceAlpha','0.5')
mesh(X_grid,Y_grid,upper_interval_grid,'FaceAlpha','0.5')
scatter3(X_test, Y_test, intervals(:,1), 'b')
scatter3(X_test, Y_test, intervals(:,2), 'b')
hold off
title('GPR Fit of Noise-Free Observations')
legend({'Noise-free observations','GPR prediction points', 'GPR prediction mesh','95% Prediction Lower Bound Mesh','95% Prediction Upper Bound Mesh','95% Prediction Lower Bound Point','95% Prediction Upper Bound Point'},'Location','best')
xlabel('X Position, (m)','Interpreter','latex')
ylabel('Y Position, (m)','Interpreter','latex')
zlabel('Sensor Reading','Interpreter','latex')

%% ReTrain Model with Request 1 and Plot Hyperparam Optimization
tbl = readtable('init_data_with_request_1and2and3.txt','FileType','text','ReadVariableNames',true);
gprMdl1 = fitrgp(tbl,'Sensor','KernelFunction','squaredexponential');
gprMdl2 = fitrgp(tbl,'Sensor',...
      'FitMethod','sr','PredictMethod','fic','ActiveSetMethod','entropy',...
      'OptimizeHyperparameters','all','HyperparameterOptimizationOptions',...
      struct('MaxObjectiveEvaluations',60,'UseParallel',true));
ypred = resubPredict(gprMdl2);
train_Loss = resubLoss(gprMdl2)

ypred1 = resubPredict(gprMdl1);
ypred2 = resubPredict(gprMdl2);

figure
plot(tbl.Sensor,'r.','MarkerSize',10);
hold on
plot(ypred1,'b');
plot(ypred2,'k','LineWidth',1);
xlabel('Input Pair','Interpreter', 'latex');
ylabel('Output Reading','Interpreter', 'latex');
legend({'data','Initial Fit','Optimized Fit'},'Location','Best','Interpreter', 'latex');
title('Impact of Optimization','Interpreter', 'latex');
hold off

figure
scatter(tbl.X,tbl.Y,10,tbl.Sensor)
colormap(gca,'default')
colorbar
grid on
xlabel('X Position, (m)','Interpreter','latex')
ylabel('Y Position, (m)','Interpreter','latex')


%% Test Model and Plot with data from Request 1,2 and 3
X_test = 2*rand(1000,1)-1;
Y_test = 2*rand(1000,1)-1;
[sensor_pred, ~, intervals] = predict(gprMdl2,[X_test Y_test]);


[X_grid,Y_grid] = meshgrid(linspace(-1,1,1000),linspace(-1,1,1000)) ;
lower_interval_grid = griddata(X_test,Y_test,intervals(:,1),X_grid,Y_grid) ;
upper_interval_grid = griddata(X_test,Y_test,intervals(:,2),X_grid,Y_grid) ;
prediction_grid = griddata(X_test,Y_test,sensor_pred,X_grid,Y_grid) ;



figure
scatter(X_test,Y_test,10,sensor_pred)
colormap(gca,'default')
colorbar
grid on
xlabel('X Position, (m)','Interpreter','latex')
ylabel('Y Position, (m)','Interpreter','latex')


figure
scatter3(tbl.X,tbl.Y,tbl.Sensor,'r') % Observed data points
hold on
scatter3(X_test, Y_test, sensor_pred,'g') % GPR predictions
mesh(X_grid,Y_grid,prediction_grid,'FaceAlpha','0.5')
mesh(X_grid,Y_grid,lower_interval_grid,'FaceAlpha','0.5')
mesh(X_grid,Y_grid,upper_interval_grid,'FaceAlpha','0.5')
scatter3(X_test, Y_test, intervals(:,1), 'b')
scatter3(X_test, Y_test, intervals(:,2), 'b')
hold off
title('GPR Fit of Noise-Free Observations')
legend({'Noise-free observations','GPR prediction points', 'GPR prediction mesh','95% Prediction Lower Bound Mesh','95% Prediction Upper Bound Mesh','95% Prediction Lower Bound Point','95% Prediction Upper Bound Point'},'Location','best')
xlabel('X Position, (m)','Interpreter','latex')
ylabel('Y Position, (m)','Interpreter','latex')
zlabel('Sensor Reading','Interpreter','latex')