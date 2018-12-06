clear all;
load('MS_Harm_h3_N1568_RMS70_P2P350.mat');
load('data.mat');

U = u_m';
Y = y_m';

%numero de estados
state_dimension = size(A,1);
%numero de observacoes
obsv_dimension = size(C,1);

%numero de dados
N = size(U,2);

%Ruido de processo
var_pnoise = .1;                   % variancia
mu_pnoise = 0;                     % media
std_pnoise = sqrt(var_pnoise)';    % desvio padrao
pnoise = std_pnoise * randn(state_dimension,N) + mu_pnoise*ones(state_dimension,N);    % Gaussian Process Noise
Q = cov(pnoise');

%Ruido de observacao
var_onoise = 2;                    % variancia
mu_onoise = 0;                     % media
std_onoise = sqrt(var_onoise)';    % desvio padrao
onoise = std_onoise * randn(obsv_dimension, N) + mu_onoise*ones(obsv_dimension, N);    % Gaussian Observation Noise
R = cov(onoise');    

%Estado inicial
x_estimated(:,1) = 0.001*randn(state_dimension,1);    
px(:,:,1) = eye(state_dimension)*10^2; 

for i = 1 : size(Y,2) - 1
    
    %E[x(k+1)|Yk-1] = Ax(k|k-1) + BU(k)
    x_estimated_priori(:,i+1) = A * x_estimated(:,i) + B*U(:,i);
   
    %P(k+1|k) = A(k)P(k|k-1)A^T + Q(k)
    px_priori = A*px(:,:,i)*A' + Q;
    
    %Kf(k) = P(k|k-1)C(k)^T(C(k)P(k|k-1)C(k)^T + R(k))^-1
    K = px_priori * C' * inv(C*px_priori*C' + R);
    
    % y(k+1|k) = C(k)x(k|k-1) + DU(k)
    y_estimated(:,i+1) = C * x_estimated_priori(:,i+1) + D*U(:,i+1);
    
    %e(k) = y(k) - (C(k)x(k|k-1) + DU(k))
    inov(:,i+1) = Y(:,i+1) - y_estimated(:,i+1);
    
    % x(k+1|k) = E[x(k+1)|Yk-1] + E[x(k+1)|e(k)]  
    x_estimated(:,i+1) = x_estimated_priori(:,i+1) + K * inov(:,i+1);
    
    %P(k|k) = P(k|k-1) - Kf(k)C(k)P(k|k-1)
    px(:,:,i+1) = px_priori - K*C*px_priori;
    
end

figure, plot(Y(1,:),'b')
hold on, plot(y_estimated(1,:),'r')
grid on
legend('Real','Estimado') 

