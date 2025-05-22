clear; clc; close all;

% --- Seleção da pasta com arquivos CSV ---
folder_path = uigetdir('', 'Selecione a pasta com os arquivos CSV');
if folder_path == 0
    error('Nenhuma pasta selecionada. O script será encerrado.');
end

% --- Listar arquivos CSV na pasta ---
file_list = dir(fullfile(folder_path, '*.csv'));
arquivos = {file_list.name};

% --- Configurações iniciais ---
cutoff_freq = 10; % Frequência de corte do filtro passa-baixa em Hz
inputs = [];
outputs = [];

% Ajustar filtro passa-baixa para 100 Hz
[b, a] = butter(4, cutoff_freq / (100 / 2), 'low');

% Colunas necessárias
required_columns = {'P6_RF_acc_x', 'P6_RF_acc_y', 'P6_RF_acc_z', ...
                    'P6_LF_acc_x', 'P6_LF_acc_y', 'P6_LF_acc_z', ...
                    'rightTotalForce_N_', 'leftTotalForce_N_'};

valid_files = {};

for i = 1:length(arquivos)
    arquivo = fullfile(folder_path, arquivos{i});
    dados = readtable(arquivo);

    if all(ismember(required_columns, dados.Properties.VariableNames))
        valid_files{end+1} = arquivo;

        % --- Pé direito ---
        acc_r = [dados.P6_RF_acc_x, dados.P6_RF_acc_y, dados.P6_RF_acc_z];
        grf_r = dados.rightTotalForce_N_;

        for j = 1:3
            acc_r(:, j) = filtfilt(b, a, acc_r(:, j));
            acc_r(:, j) = (acc_r(:, j) - mean(acc_r(:, j))) / std(acc_r(:, j));
        end

        % --- Pé esquerdo ---
        acc_l = [dados.P6_LF_acc_x, dados.P6_LF_acc_y, dados.P6_LF_acc_z];
        grf_l = dados.leftTotalForce_N_;

        for j = 1:3
            acc_l(:, j) = filtfilt(b, a, acc_l(:, j));
            acc_l(:, j) = (acc_l(:, j) - mean(acc_l(:, j))) / std(acc_l(:, j));
        end

        inputs = [inputs; acc_r; acc_l];
        outputs = [outputs; grf_r; grf_l];
    end
end

if isempty(valid_files)
    error('Nenhum arquivo válido foi encontrado na pasta selecionada.');
end

disp(['Arquivos utilizados no treinamento: ', num2str(length(valid_files))]);

% --- Normalização da saída (vGRF) ---
mean_grf = mean(outputs, 1);
std_grf = std(outputs, 0, 1);
outputs = (outputs - mean_grf) ./ std_grf;
save('normalization_params.mat', 'mean_grf', 'std_grf');

% --- Criação de janelas temporais ---
time_window = 30;
step_sizes = [1, 3, 5];

X = {};
Y = [];

for step_size = step_sizes
    for i = 1:step_size:(size(inputs, 1) - time_window)
        janela = inputs(i:i + time_window - 1, :);
        X{end+1} = janela';  % (num_features, time_window)
        Y = [Y; outputs(i + time_window - 1, :)];
    end
end

disp(['Dimensão final de X: ', mat2str(size(X))]);
disp(['Dimensão final de Y: ', mat2str(size(Y))]);

% --- Validação cruzada ---
k = 5;
indices = crossvalind('Kfold', size(Y, 1), k);

rmse_folds = zeros(k, 1);
rRMSE_folds = zeros(k, 1);
r2_folds = zeros(k, 1);

for fold = 1:k
    disp(['Iniciando fold ', num2str(fold), ' de ', num2str(k)]);
    test_idx = (indices == fold);
    train_idx = ~test_idx;

    X_train = X(train_idx);
    Y_train = Y(train_idx, :);
    X_test = X(test_idx);
    Y_test = Y(test_idx, :);

    if isempty(X_train) || isempty(Y_train)
        error('Os dados de treinamento estão vazios.');
    end

    % --- Rede Híbrida TCN + Bi-LSTM ---
    layers = [
        sequenceInputLayer(size(X{1},1), 'Name', 'input')

        % Bloco TCN com convoluções causais dilatadas
        convolution1dLayer(3, 64, 'Padding', 'causal', 'DilationFactor', 1, 'Name', 'conv1')
        reluLayer('Name', 'relu1')
        dropoutLayer(0.2, 'Name', 'drop1')

        convolution1dLayer(3, 64, 'Padding', 'causal', 'DilationFactor', 2, 'Name', 'conv2')
        reluLayer('Name', 'relu2')
        dropoutLayer(0.2, 'Name', 'drop2')

        % Camada Bi-LSTM após convoluções dilatadas
        bilstmLayer(64, 'OutputMode','last', 'Name', 'bilstm')
        dropoutLayer(0.3, 'Name', 'drop_bilstm')

        % Camadas densas e saída
        fullyConnectedLayer(64, 'Name', 'fc1')
        reluLayer('Name', 'relu_fc1')
        fullyConnectedLayer(1, 'Name', 'fc_out')

        regressionLayer('Name', 'regressionoutput')
    ];

    options = trainingOptions('adam', ...
        'MaxEpochs', 30, ...
        'InitialLearnRate', 5e-3, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.8, ...
        'LearnRateDropPeriod', 10, ...
        'MiniBatchSize', 128, ...
        'Shuffle', 'every-epoch', ...
        'ValidationData', {X_test, Y_test}, ...
        'ValidationFrequency', 50, ...
        'ValidationPatience', 10, ...
        'Verbose', false, ...
        'Plots', 'training-progress', ...
        'L2Regularization', 1e-4, ...
        'GradientThreshold', 1);

    % Treinamento
    net = trainNetwork(X_train, Y_train, layers, options);

    % Previsão
    predictions = predict(net, X_test);

    % Desnormalização
    predictions = (predictions .* std_grf) + mean_grf;
    Y_test = (Y_test .* std_grf) + mean_grf;

    % Avaliação
    rmse_folds(fold) = sqrt(mean((Y_test - predictions).^2));
    range_val = max(Y_test) - min(Y_test);
    rRMSE_folds(fold) = (rmse_folds(fold) / range_val) * 100;

    ss_total = sum((Y_test - mean(Y_test)).^2);
    ss_residual = sum((Y_test - predictions).^2);
    r2_folds(fold) = 1 - (ss_residual / ss_total);

    % Exportar resíduos
    residuos = Y_test - predictions;
    residuos_tabela = table((1:length(Y_test))', Y_test, predictions, residuos, ...
        'VariableNames', {'Sample', 'GRF_real', 'GRF_predito', 'Resíduo'});
    nome_arquivo_residuos = sprintf('residuos_TCN_BiLSTM_fold_%d.csv', fold);
    writetable(residuos_tabela, nome_arquivo_residuos);
    disp(['Resíduos do fold ', num2str(fold), ' salvos em "', nome_arquivo_residuos, '".']);
end

% --- Resultados Finais ---
mean_rmse = mean(rmse_folds);
mean_rRMSE = mean(rRMSE_folds);
mean_r2 = mean(r2_folds);

disp('--- Resultados Finais ---');
disp(['Média RMSE: ', sprintf('%.2f', mean_rmse)]);
disp(['Média rRMSE (%): ', sprintf('%.2f', mean_rRMSE), '%']);
disp(['Média R²: ', sprintf('%.2f', mean_r2)]);

% --- Salvar métricas e modelo ---
metricas_tabela = table((1:k)', rmse_folds, rRMSE_folds, r2_folds, ...
    'VariableNames', {'Fold', 'RMSE', 'rRMSE', 'R2'});
writetable(metricas_tabela, 'metricas_TCN_BiLSTM_vGRF_RL.csv');
disp('Métricas salvas no arquivo "metricas_TCN_BiLSTM_vGRF_RL.csv".');

save('modelo_TCN_BiLSTM_vGRF_RL.mat', 'net');
disp('Modelo treinado salvo como "modelo_TCN_BiLSTM_vGRF_RL.mat".');
