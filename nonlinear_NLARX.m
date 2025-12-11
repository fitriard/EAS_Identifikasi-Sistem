%% ===============================
fprintf('\n=== IDENTIFIKASI NONLINEAR ARX (NLARX) ===\n');

% basic checks
if ~exist('best_model','var') || isempty(best_model)
    error('best_model not found. Create linear ARX and store as best_model.');
end
if ~exist('data_train','var') || ~exist('data_test','var')
    error('data_train/data_test not found.');
end

% get orders
try
    na = best_model.na; nb = best_model.nb; nk = best_model.nk;
catch
    try
        na = get(best_model,'na'); nb = get(best_model,'nb'); nk = get(best_model,'nk');
    catch
        error('Cannot read na/nb/nk from best_model. Ensure best_model is idpoly.');
    end
end
Orders = [na nb nk];
fprintf('Using orders [na nb nk] = [%d %d %d]\n', Orders);

% build candidates (robust)
cand_models = {}; cand_names = {};
fprintf('\nBuilding NLARX candidates...\n');

% Wavenet
try
    mdl_wav = nlarx(data_train, Orders, wavenet('NumberOfUnits',10));
    cand_models{end+1} = mdl_wav; cand_names{end+1}='NLARX Wavenet';
catch ME
    warning('Wavenet build failed: %s', ME.message);
end

% Sigmoid
try
    mdl_sig = nlarx(data_train, Orders, sigmoidnet('NumberOfUnits',10));
    cand_models{end+1} = mdl_sig; cand_names{end+1}='NLARX Sigmoid';
catch ME
    warning('SigmoidNet build failed: %s', ME.message);
end

% Tree
try
    mdl_tree = nlarx(data_train, Orders, treepartition);
    cand_models{end+1} = mdl_tree; cand_names{end+1}='NLARX Tree';
catch ME
    warning('TreePartition build failed: %s', ME.message);
end

if isempty(cand_models)
    error('No NLARX candidate built. Check toolbox availability.');
end

% helper metrics
MSE_func = @(y,yh) mean((y(:)-yh(:)).^2);
RMSE_func = @(y,yh) sqrt(MSE_func(y,yh));
MAE_func = @(y,yh) mean(abs(y(:)-yh(:)));
MAPE_func = @(y,yh) mean(abs((y(:)-yh(:))./ (y(:) + (y(:)==0)) ))*100;

% evaluate each candidate
nCand = length(cand_models);
fit_train = nan(1,nCand); fit_test = nan(1,nCand);
MSE_train = nan(1,nCand); RMSE_train = nan(1,nCand); MAE_train = nan(1,nCand); MAPE_train = nan(1,nCand);
MSE_test  = nan(1,nCand); RMSE_test  = nan(1,nCand); MAE_test  = nan(1,nCand); MAPE_test  = nan(1,nCand);
yhat_train_all = cell(1,nCand); yhat_test_all = cell(1,nCand);
resid_train_all = cell(1,nCand); resid_test_all = cell(1,nCand);

for k=1:nCand
    mdl = cand_models{k};
    name = cand_names{k};
    fprintf('\nEvaluating %s ...\n', name);
    
    % TRAIN compare
    try
        [cmp_tr, fit_tr] = compare(data_train, mdl);
        y_train = data_train.y(:);
        if isobject(cmp_tr) && isprop(cmp_tr,'OutputData')
            yhat_train = cmp_tr.OutputData(:);
        elseif isstruct(cmp_tr) && isfield(cmp_tr,'y')
            yhat_train = cmp_tr.y(:);
        else
            yhat_train = cmp_tr(:);
        end
        fit_train(k) = fit_tr;
        yhat_train_all{k} = yhat_train;
        resid_train_all{k} = y_train - yhat_train;
    catch ME
        warning('Compare train failed for %s: %s', name, ME.message);
        fit_train(k) = NaN; 
        yhat_train_all{k} = nan(size(data_train.y(:)));
        resid_train_all{k} = nan(size(data_train.y(:)));
    end
    
    % TEST compare
    try
        [cmp_te, fit_te] = compare(data_test, mdl);
        y_test = data_test.y(:);
        if isobject(cmp_te) && isprop(cmp_te,'OutputData')
            yhat_test = cmp_te.OutputData(:);
        elseif isstruct(cmp_te) && isfield(cmp_te,'y')
            yhat_test = cmp_te.y(:);
        else
            yhat_test = cmp_te(:);
        end
        fit_test(k) = fit_te;
        yhat_test_all{k} = yhat_test;
        resid_test_all{k} = y_test - yhat_test;
    catch ME
        warning('Compare test failed for %s: %s', name, ME.message);
        fit_test(k) = -Inf; 
        yhat_test_all{k} = nan(size(data_test.y(:)));
        resid_test_all{k} = nan(size(data_test.y(:)));
    end

    % compute metrics train
    try
        MSE_train(k) = MSE_func(y_train, yhat_train_all{k});
        RMSE_train(k)= RMSE_func(y_train, yhat_train_all{k});
        MAE_train(k) = MAE_func(y_train, yhat_train_all{k});
        MAPE_train(k)= MAPE_func(y_train, yhat_train_all{k});
    catch
        MSE_train(k)=NaN; RMSE_train(k)=NaN; MAE_train(k)=NaN; MAPE_train(k)=NaN;
    end
    
    % metrics test
    try
        MSE_test(k) = MSE_func(y_test, yhat_test_all{k});
        RMSE_test(k)= RMSE_func(y_test, yhat_test_all{k});
        MAE_test(k) = MAE_func(y_test, yhat_test_all{k});
        MAPE_test(k)= MAPE_func(y_test, yhat_test_all{k});
    catch
        MSE_test(k)=NaN; RMSE_test(k)=NaN; MAE_test(k)=NaN; MAPE_test(k)=NaN;
    end
end

% Print ringkasan TRAIN / TEST
fprintf('\n=== RINGKASAN PERFORMA NLARX (TRAIN) ===\n');
fprintf('%-15s\t%-8s\t%-12s\t%-10s\t%-10s\t%-10s\n', 'Model','FIT(%)','MSE','RMSE','MAE','MAPE(%)');
for k=1:nCand
    fprintf('%-15s\t%-8.2f\t%-12.6f\t%-10.6f\t%-10.6f\t%-10.4f\n', ...
        cand_names{k}, fit_train(k), MSE_train(k), RMSE_train(k), MAE_train(k), MAPE_train(k));
end

fprintf('\n=== RINGKASAN PERFORMA NLARX (TEST) ===\n');
fprintf('%-15s\t%-8s\t%-12s\t%-10s\t%-10s\t%-10s\n', 'Model','FIT(%)','MSE','RMSE','MAE','MAPE(%)');
for k=1:nCand
    fprintf('%-15s\t%-8.2f\t%-12.6f\t%-10.6f\t%-10.6f\t%-10.4f\n', ...
        cand_names{k}, fit_test(k), MSE_test(k), RMSE_test(k), MAE_test(k), MAPE_test(k));
end

% select best by FIT test
[fit_best, idx_best] = max(fit_test);
best_model_nl = cand_models{idx_best};
best_name = cand_names{idx_best};
fprintf('\n>>> Model NLARX terbaik (berdasarkan FIT TEST) = %s, FIT = %.2f %%\n', best_name, fit_best);

%% ========================================
%% 1. PLOT RESIDU (ANALISIS RESIDUAL)
%% ========================================
fprintf('\n=== ANALISIS RESIDU ===\n');

% Ljung-Box test untuk semua model
fprintf('\nLjung-Box Test (Lags=20):\n');
fprintf('%-15s\t%-10s\t%-10s\t%-20s\n', 'Model','H','p-value','Kesimpulan');
h_lb_all = nan(1,nCand); p_lb_all = nan(1,nCand);

for k=1:nCand
    try
        e_test = resid_test_all{k};
        [h_lb, p_lb] = lbqtest(e_test, 'Lags', 20);
        h_lb_all(k) = h_lb;
        p_lb_all(k) = p_lb;
        if h_lb == 0
            kesimpulan = 'White noise';
        else
            kesimpulan = 'Bukan white noise';
        end
        fprintf('%-15s\t%-10d\t%-10.4f\t%-20s\n', cand_names{k}, h_lb, p_lb, kesimpulan);
    catch ME
        warning('Ljung-Box test failed for %s: %s', cand_names{k}, ME.message);
        fprintf('%-15s\t%-10s\t%-10s\t%-20s\n', cand_names{k}, 'N/A', 'N/A', 'Error');
    end
end

% Plot residu untuk semua model
figure('Name','Analisis Residu - Semua Model','Units','normalized','Position',[0.05 0.05 0.9 0.85]);

for k=1:nCand
    % Time series plot
    subplot(nCand, 3, (k-1)*3 + 1);
    try
        plot(resid_test_all{k}, 'b-', 'LineWidth', 0.8);
        grid on;
        xlabel('Sampel'); ylabel('Residu');
        title(sprintf('%s\nFIT=%.2f%%, p-val=%.4f', cand_names{k}, fit_test(k), p_lb_all(k)));
        yline(0, 'r--', 'LineWidth', 1);
    catch
        text(0.5, 0.5, 'Data tidak tersedia', 'HorizontalAlignment', 'center');
    end
    
    % Histogram
    subplot(nCand, 3, (k-1)*3 + 2);
    try
        histogram(resid_test_all{k}, 30, 'Normalization', 'pdf', 'FaceColor', [0.3 0.6 0.9]);
        hold on;
        % Overlay normal distribution
        res = resid_test_all{k};
        mu_res = mean(res);
        sigma_res = std(res);
        x_range = linspace(min(res), max(res), 100);
        y_norm = normpdf(x_range, mu_res, sigma_res);
        plot(x_range, y_norm, 'r-', 'LineWidth', 2);
        hold off;
        grid on;
        xlabel('Nilai Residu'); ylabel('Densitas');
        title(sprintf('Histogram\nμ=%.4f, σ=%.4f', mu_res, sigma_res));
        legend('Data', 'Normal', 'Location', 'best');
    catch
        text(0.5, 0.5, 'Data tidak tersedia', 'HorizontalAlignment', 'center');
    end
    
    % ACF plot
    subplot(nCand, 3, (k-1)*3 + 3);
    try
        [acf_vals, lags, bounds] = autocorr(resid_test_all{k}, 'NumLags', 20);
        stem(lags, acf_vals, 'b', 'filled', 'MarkerSize', 4);
        hold on;
        plot([0 20], [bounds(1) bounds(1)], 'r--', 'LineWidth', 1);
        plot([0 20], [bounds(2) bounds(2)], 'r--', 'LineWidth', 1);
        hold off;
        grid on;
        xlabel('Lag'); ylabel('ACF');
        title('Autocorrelation');
        ylim([-1 1]);
    catch
        text(0.5, 0.5, 'Data tidak tersedia', 'HorizontalAlignment', 'center');
    end
end

sgtitle('Analisis Residu Model NLARX (Data Test)', 'FontSize', 14, 'FontWeight', 'bold');

%% ========================================
%% 2. MODEL MATEMATIKA
%% ========================================
fprintf('\n========================================\n');
fprintf('MODEL MATEMATIKA NLARX\n');
fprintf('========================================\n');

% Regressors
regressors = cell(1, na + nb);
idxr = 1;
for i=1:na
    regressors{idxr} = sprintf('y(t-%d)', i); idxr = idxr+1;
end
for j=0:nb-1
    regressors{idxr} = sprintf('u(t-%d)', nk + j); idxr = idxr+1;
end

fprintf('\n--- REPRESENTASI UMUM NLARX ---\n');
fprintf('Model NLARX mengikuti struktur:\n');
fprintf('  y(t) = F[Φ(t)] + e(t)\n\n');
fprintf('dimana:\n');
fprintf('  Φ(t) = [%s] adalah vektor regressor\n', strjoin(regressors, ', '));
fprintf('  F[·] adalah fungsi nonlinear\n');
fprintf('  e(t) adalah error/noise\n\n');

fprintf('--- DETAIL UNTUK SETIAP MODEL ---\n\n');

for k=1:nCand
    fprintf('MODEL: %s\n', cand_names{k});
    fprintf(repmat('-', 1, 60)); fprintf('\n');
    
    if contains(cand_names{k}, 'Wavenet', 'IgnoreCase', true)
        fprintf('Tipe: Wavelet Network\n');
        fprintf('Struktur matematis:\n');
        fprintf('  y(t) = Σ[i=1 to N] w_i * ψ((Φ(t) - b_i) / a_i) + e(t)\n\n');
        fprintf('dimana:\n');
        fprintf('  ψ(·) = fungsi wavelet (biasanya Mexican hat atau Morlet)\n');
        fprintf('  N = jumlah unit wavelet (10 unit)\n');
        fprintf('  w_i = bobot output untuk unit ke-i\n');
        fprintf('  a_i = parameter skala (dilation) untuk unit ke-i\n');
        fprintf('  b_i = parameter translasi untuk unit ke-i\n');
        fprintf('  Φ(t) = vektor regressor\n\n');
        fprintf('Karakteristik:\n');
        fprintf('  - Mampu menangkap pola lokal dan transient\n');
        fprintf('  - Efektif untuk sistem dengan dinamika yang berubah cepat\n');
        fprintf('  - Fungsi basis wavelet memberikan lokalisasi waktu-frekuensi\n\n');
        
    elseif contains(cand_names{k}, 'Sigmoid', 'IgnoreCase', true)
        fprintf('Tipe: Sigmoid Network (Neural Network)\n');
        fprintf('Struktur matematis:\n');
        fprintf('  y(t) = Σ[i=1 to N] w_i * σ(Φ(t)ᵀ·v_i + b_i) + w_0 + e(t)\n\n');
        fprintf('dimana:\n');
        fprintf('  σ(z) = 1/(1 + exp(-z)) adalah fungsi sigmoid\n');
        fprintf('  N = jumlah neuron hidden layer (10 unit)\n');
        fprintf('  w_i = bobot dari hidden unit ke-i ke output\n');
        fprintf('  v_i = vektor bobot input ke hidden unit ke-i\n');
        fprintf('  b_i = bias untuk hidden unit ke-i\n');
        fprintf('  w_0 = bias output\n');
        fprintf('  Φ(t) = vektor regressor\n\n');
        fprintf('Karakteristik:\n');
        fprintf('  - Universal function approximator\n');
        fprintf('  - Smooth nonlinear mapping\n');
        fprintf('  - Efektif untuk berbagai jenis nonlinearitas\n\n');
        
    elseif contains(cand_names{k}, 'Tree', 'IgnoreCase', true)
        fprintf('Tipe: Tree Partition (Piecewise Linear)\n');
        fprintf('Struktur matematis:\n');
        fprintf('  y(t) = Σ[k=1 to R] I_k(Φ(t)) · [a_k0 + a_kᵀ·Φ(t)] + e(t)\n\n');
        fprintf('dimana:\n');
        fprintf('  R = jumlah region/partisi\n');
        fprintf('  I_k(Φ) = fungsi indikator (1 jika Φ dalam region k, 0 jika tidak)\n');
        fprintf('  a_k0 = intercept untuk region ke-k\n');
        fprintf('  a_k = vektor koefisien linear untuk region ke-k\n');
        fprintf('  Φ(t) = vektor regressor\n\n');
        fprintf('Setiap region k memiliki model linear lokal:\n');
        fprintf('  f_k(Φ) = a_k0');
        for r=1:length(regressors)
            fprintf(' + a_k%d·%s', r, regressors{r});
        end
        fprintf(' + e(t)\n\n');
        fprintf('Karakteristik:\n');
        fprintf('  - Piecewise linear approximation\n');
        fprintf('  - Interpretable: setiap region = model linear sederhana\n');
        fprintf('  - Efisien secara komputasi\n');
        fprintf('  - Cocok untuk sistem dengan mode operasi berbeda\n\n');
        
        % Coba ekstrak info tree jika memungkinkan
        try
            mdl = cand_models{k};
            if isprop(mdl, 'Tree') && ~isempty(mdl.Tree)
                fprintf('Informasi Tree Structure:\n');
                fprintf('  (Detail struktur tree bisa diekstrak dari mdl.Tree)\n\n');
            end
        catch
            % Skip jika tidak bisa
        end
    end
    
    fprintf('\n');
end

%% ========================================
%% 3. KESIMPULAN
%% ========================================
fprintf('\n========================================\n');
fprintf('KESIMPULAN IDENTIFIKASI NLARX\n');
fprintf('========================================\n\n');

fprintf('1. PERBANDINGAN PERFORMA MODEL:\n');
fprintf('   %-15s : FIT Train = %.2f%%, FIT Test = %.2f%%\n', ...
    cand_names{1}, fit_train(1), fit_test(1));
for k=2:nCand
    fprintf('   %-15s : FIT Train = %.2f%%, FIT Test = %.2f%%\n', ...
        cand_names{k}, fit_train(k), fit_test(k));
end

fprintf('\n2. MODEL TERBAIK:\n');
fprintf('   Berdasarkan FIT Test, model terbaik adalah: %s\n', best_name);
fprintf('   - FIT Training  : %.2f%%\n', fit_train(idx_best));
fprintf('   - FIT Testing   : %.2f%%\n', fit_test(idx_best));
fprintf('   - MSE Testing   : %.6f\n', MSE_test(idx_best));
fprintf('   - RMSE Testing  : %.6f\n', RMSE_test(idx_best));
fprintf('   - MAE Testing   : %.6f\n', MAE_test(idx_best));
fprintf('   - MAPE Testing  : %.4f%%\n', MAPE_test(idx_best));

fprintf('\n3. ANALISIS RESIDU MODEL TERBAIK:\n');
if ~isnan(h_lb_all(idx_best))
    if h_lb_all(idx_best) == 0
        fprintf('   ✓ Residu adalah WHITE NOISE (p-value = %.4f)\n', p_lb_all(idx_best));
        fprintf('   ✓ Model telah menangkap dinamika sistem dengan baik\n');
        fprintf('   ✓ Tidak ada autokorelasi signifikan dalam residu\n');
    else
        fprintf('   ✗ Residu BUKAN white noise (p-value = %.4f)\n', p_lb_all(idx_best));
        fprintf('   ✗ Masih terdapat struktur dalam residu yang belum dimodelkan\n');
        fprintf('   → Rekomendasi: tuning parameter atau coba arsitektur lain\n');
    end
else
    fprintf('   ! Ljung-Box test tidak dapat dihitung\n');
end

fprintf('\n4. KEUNGGULAN MODEL NONLINEAR (NLARX):\n');
fprintf('   + Mampu menangkap nonlinearitas dalam sistem\n');
fprintf('   + Lebih fleksibel dibanding model linear ARX\n');
if contains(best_name, 'Wavenet', 'IgnoreCase', true)
    fprintf('   + Wavenet: baik untuk dinamika lokal dan transient\n');
elseif contains(best_name, 'Sigmoid', 'IgnoreCase', true)
    fprintf('   + Sigmoid: universal approximator dengan smooth mapping\n');
elseif contains(best_name, 'Tree', 'IgnoreCase', true)
    fprintf('   + Tree: interpretable dengan piecewise linear regions\n');
end

fprintf('\n5. REKOMENDASI:\n');
if fit_test(idx_best) > 80
    fprintf('   ✓ Model memiliki akurasi SANGAT BAIK (FIT > 80%%)\n');
    fprintf('   ✓ Model siap digunakan untuk prediksi/kontrol\n');
elseif fit_test(idx_best) > 70
    fprintf('   ✓ Model memiliki akurasi BAIK (FIT > 70%%)\n');
    fprintf('   → Bisa digunakan, pertimbangkan tuning untuk hasil lebih baik\n');
elseif fit_test(idx_best) > 60
    fprintf('   ~ Model memiliki akurasi CUKUP (FIT > 60%%)\n');
    fprintf('   → Disarankan tuning parameter atau coba struktur lain\n');
else
    fprintf('   ✗ Model memiliki akurasi RENDAH (FIT < 60%%)\n');
    fprintf('   → Perlu perbaikan: tuning ekstensif atau model order lebih tinggi\n');
end

fprintf('\n========================================\n\n');

%% ========================================
%% PLOT TAMBAHAN
%% ========================================

% Plot comparison zoomed
y_data = data_test.y(:);
Ntest = length(y_data);
center = round(Ntest/2);
win = max(1, center-50):min(Ntest, center+50);
tvec = (1:Ntest)';

figure('Name','Simulated Response Comparison','Units','normalized','Position',[0.05 0.6 0.9 0.3]);
hold on;
plot(tvec(win), y_data(win), 'Color',[0.2 0.2 0.2], 'LineWidth',1.5, 'DisplayName','Data Aktual');
colors = lines(nCand);
for k=1:nCand
    try
        yhatk = yhat_test_all{k};
        plot(tvec(win), yhatk(win), 'Color', colors(k,:), 'LineWidth',1.2, ...
            'DisplayName', sprintf('%s: %.2f%%', cand_names{k}, fit_test(k)));
    catch
    end
end
xlabel('Sampel'); ylabel('Output'); 
title('Perbandingan Response Model NLARX (Zoomed Window)');
legend('Location','best'); grid on; hold off;

% Plot best model: train vs test
yhat_train_best = yhat_train_all{idx_best};
yhat_test_best  = yhat_test_all{idx_best};

figure('Name','Model Terbaik: Train vs Test','Units','normalized','Position',[0.05 0.05 0.9 0.45]);
subplot(2,1,1);
plot(data_train.y(:), 'b-', 'LineWidth',1.2); hold on;
plot(yhat_train_best, 'r--', 'LineWidth',1.0); hold off;
grid on; xlabel('Sampel'); ylabel('Output');
title(sprintf('Model Terbaik: %s | Data Training vs Prediksi | FIT = %.2f%%', best_name, fit_train(idx_best)));
legend('Aktual','Prediksi','Location','best');

subplot(2,1,2);
plot(data_test.y(:), 'b-', 'LineWidth',1.2); hold on;
plot(yhat_test_best, 'r--', 'LineWidth',1.0); hold off;
grid on; xlabel('Sampel'); ylabel('Output');
title(sprintf('Data Testing vs Prediksi | FIT = %.2f%%', fit_test(idx_best)));
legend('Aktual','Prediksi','Location','best');

% save to workspace
assignin('base','model_nl_best', best_model_nl);
assignin('base','model_nl_name', best_name);
assignin('base','fit_nl_train', fit_train(idx_best));
assignin('base','fit_nl_test', fit_test(idx_best));
assignin('base','metrics_nl_test', struct('MSE', MSE_test(idx_best), 'RMSE', RMSE_test(idx_best), ...
    'MAE', MAE_test(idx_best), 'MAPE', MAPE_test(idx_best)));

fprintf('\n✓ NLARX analysis completed.\n');
fprintf('✓ Model ''%s'' tersimpan di workspace sebagai ''model_nl_best''.\n', best_name);
fprintf('✓ Metrik performa tersimpan di ''metrics_nl_test''.\n\n');