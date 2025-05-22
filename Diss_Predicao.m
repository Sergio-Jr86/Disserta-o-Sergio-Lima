clear; clc; close all;

% --- Seleção do arquivo CSV ---
[file, path] = uigetfile('*.csv', 'Selecione um arquivo CSV para predição');
if file == 0
    error('Nenhum arquivo selecionado. O script será encerrado.');
end
arquivo = fullfile(path, file);

% --- Modelos e nomes ---
models = {'modelo_vGRF_RL.mat', 'modelo_TCN_vGRF_RL.mat', 'modelo_TCN_BiLSTM_vGRF_RL.mat'};
model_names = {'Bi-LSTM', 'TCN', 'Híbrido'};

% --- Carregar parâmetros de normalização comuns ---
load('normalization_params.mat', 'mean_grf', 'std_grf');

% --- Configuração do filtro ---
cutoff_freq = 15; fs = 100;
[b, a] = butter(4, cutoff_freq / (fs / 2), 'low');
time_window = 30;

% --- Colunas necessárias ---
cols_r = {'P6_RF_acc_x', 'P6_RF_acc_y', 'P6_RF_acc_z', 'rightTotalForce_N_'};
cols_l = {'P6_LF_acc_x', 'P6_LF_acc_y', 'P6_LF_acc_z', 'leftTotalForce_N_'};

% --- Leitura e validação ---
dados = readtable(arquivo);
if ~all(ismember([cols_r, cols_l], dados.Properties.VariableNames))
    error('Colunas necessárias ausentes no arquivo.');
end

% --- Função de pré-processamento (filtro + z-score) ---
preproc = @(x) (filtfilt(b, a, x) - mean(x)) / std(x);

% --- Processar sinais de entrada ---
acc_r = [preproc(dados.P6_RF_acc_x), preproc(dados.P6_RF_acc_y), preproc(dados.P6_RF_acc_z)];
acc_l = [preproc(dados.P6_LF_acc_x), preproc(dados.P6_LF_acc_y), preproc(dados.P6_LF_acc_z)];

% --- GRF reais ---
grf_r = filtfilt(b, a, dados.rightTotalForce_N_);
grf_l = filtfilt(b, a, dados.leftTotalForce_N_);
grf_r = grf_r(time_window:end);
grf_l = grf_l(time_window:end);

% --- Janelas temporais ---
X_r = {}; X_l = {};
for i = 1:(length(acc_r) - time_window)
    X_r{end+1} = acc_r(i:i+time_window-1,:)';
    X_l{end+1} = acc_l(i:i+time_window-1,:)';
end

% --- Inicialização ---
pred_r = cell(1,3); pred_l = cell(1,3);
metricas_r = zeros(3,3); metricas_l = zeros(3,3);
correl_r = zeros(3,1); correl_l = zeros(3,1);

%% FIGURA PÉ DIREITO
fig_r = figure('Name','Pé Direito','Units','normalized','Position',[0.1 0.1 0.6 0.85]);
for i = 1:3
    load(models{i}, 'net');
    Y_pred_r = predict(net, X_r);
    Y_pred_r = filtfilt(b, a, Y_pred_r .* std_grf + mean_grf);
    pred_r{i} = Y_pred_r;

    N = min(length(Y_pred_r), length(grf_r));
    r_rmse = sqrt(mean((grf_r(1:N) - Y_pred_r(1:N)).^2));
    r_rRMSE = r_rmse / (max(grf_r(1:N)) - min(grf_r(1:N))) * 100;
    r_r2 = 1 - sum((grf_r(1:N) - Y_pred_r(1:N)).^2) / sum((grf_r(1:N) - mean(grf_r(1:N))).^2);
    r_corr = max(xcorr(grf_r(1:N), Y_pred_r(1:N), 'coeff'));
    metricas_r(i,:) = [r_rmse, r_rRMSE, r_r2];
    correl_r(i) = r_corr;

    subplot(3,1,i);
    plot(grf_r(1:N), 'k', 'LineWidth', 1.5); hold on;
    plot(Y_pred_r(1:N), 'r', 'LineWidth', 1.5);
    title(['Pé Direito - ', model_names{i}], 'FontWeight','bold');
    ylabel('GRF (N)'); xlabel('Amostras'); grid on;
end
exportgraphics(fig_r, fullfile(path, 'GRF_direito_subplots.png'), 'Resolution', 600);

%% FIGURA PÉ ESQUERDO
fig_l = figure('Name','Pé Esquerdo','Units','normalized','Position',[0.1 0.1 0.6 0.85]);
for i = 1:3
    load(models{i}, 'net');
    Y_pred_l = predict(net, X_l);
    Y_pred_l = filtfilt(b, a, Y_pred_l .* std_grf + mean_grf);
    pred_l{i} = Y_pred_l;

    N = min(length(Y_pred_l), length(grf_l));
    l_rmse = sqrt(mean((grf_l(1:N) - Y_pred_l(1:N)).^2));
    l_rRMSE = l_rmse / (max(grf_l(1:N)) - min(grf_l(1:N))) * 100;
    l_r2 = 1 - sum((grf_l(1:N) - Y_pred_l(1:N)).^2) / sum((grf_l(1:N) - mean(grf_l(1:N))).^2);
    l_corr = max(xcorr(grf_l(1:N), Y_pred_l(1:N), 'coeff'));
    metricas_l(i,:) = [l_rmse, l_rRMSE, l_r2];
    correl_l(i) = l_corr;

    subplot(3,1,i);
    plot(grf_l(1:N), 'k', 'LineWidth', 1.5); hold on;
    plot(Y_pred_l(1:N), 'b', 'LineWidth', 1.5);
    title(['Pé Esquerdo - ', model_names{i}], 'FontWeight','bold');
    ylabel('GRF (N)'); xlabel('Amostras'); grid on;
end
exportgraphics(fig_l, fullfile(path, 'GRF_esquerdo_subplots.png'), 'Resolution', 600);

% --- Salvar métricas e correlação cruzada ---
tabela_r = table(model_names', metricas_r(:,1), metricas_r(:,2), metricas_r(:,3), correl_r, ...
    'VariableNames', {'Modelo','RMSE','rRMSE','R2','CorrelacaoCruzada'});
tabela_l = table(model_names', metricas_l(:,1), metricas_l(:,2), metricas_l(:,3), correl_l, ...
    'VariableNames', {'Modelo','RMSE','rRMSE','R2','CorrelacaoCruzada'});

writetable(tabela_r, fullfile(path, 'metricas_direito.csv'));
writetable(tabela_l, fullfile(path, 'metricas_esquerdo.csv'));
disp('Métricas e correlação cruzada salvas com sucesso.');
%% --- Gráficos de Dispersão: GRF Real vs Predita ---

% --- Pé Direito ---
fig_disp_r = figure('Name','Dispersão - Pé Direito','Units','normalized','Position',[0.1 0.3 0.75 0.4]);
for i = 1:3
    N = min(length(pred_r{i}), length(grf_r));
    subplot(1,3,i);
    scatter(grf_r(1:N), pred_r{i}(1:N), 10, 'filled', 'MarkerFaceAlpha', 0.3);
    xlabel('GRF Real (N)'); ylabel('GRF Predita (N)');
    title(['Pé Direito - ', model_names{i}]);
    axis equal;
    grid on;
end
sgtitle('Dispersão - GRF Real vs Predita (Pé Direito)', 'FontWeight','bold');
exportgraphics(fig_disp_r, fullfile(path, 'Dispersao_direito_modelos.png'), 'Resolution', 600);

% --- Pé Esquerdo ---
fig_disp_l = figure('Name','Dispersão - Pé Esquerdo','Units','normalized','Position',[0.1 0.3 0.75 0.4]);
for i = 1:3
    N = min(length(pred_l{i}), length(grf_l));
    subplot(1,3,i);
    scatter(grf_l(1:N), pred_l{i}(1:N), 10, 'filled', 'MarkerFaceAlpha', 0.3);
    xlabel('GRF Real (N)'); ylabel('GRF Predita (N)');
    title(['Pé Esquerdo - ', model_names{i}]);
    axis equal;
    grid on;
end
sgtitle('Dispersão - GRF Real vs Predita (Pé Esquerdo)', 'FontWeight','bold');
exportgraphics(fig_disp_l, fullfile(path, 'Dispersao_esquerdo_modelos.png'), 'Resolution', 600);

disp('Gráficos de dispersão salvos com sucesso.');
