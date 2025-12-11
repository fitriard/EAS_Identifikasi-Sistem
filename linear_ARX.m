%% ===========================================
%   IDENTIFIKASI SISTEM ARX (LINIER) BERBASIS IO
%   Metodologi (BENAR):
%   0. Panggil data IO
%   1. Plot IO
%   2. Cek korelasi
%   3. Cek stasioneritas (jika tidak → stasionerkan)
%   4. Tentukan pemilihan model (ARX-IO)
%   5. Split Data Train - Test (70% / 30%)
%   6. Pemilihan orde dengan AIC (Grid Search)
%   7. Evaluasi MSE, RMSE, MAE, MAPE
%   8. Cek residu dengan Ljung-Box (white noise test)
%   9. Buat model matematika
% ===========================================
clc; % clear; close all;

%% ===========================================
% 0. PANGGIL DATA INPUT-OUTPUT
% ============================================
% Asumsi data sudah berupa iddata z2 (y: output, u: input)
% Load dari workspace atau file MAT

if ~exist('z2', 'var')
    error(['Data z2 belum ada di workspace.\n', ...
           'Ketik: load(''namafile.mat'') sebelum menjalankan script ini.\n', ...
           'File .mat harus mengandung variabel z2 (iddata).']);
end

y  = z2.y;
u  = z2.u;
Ts = z2.Ts;     % sampling time
N  = length(y);

fprintf('========================================\n');
fprintf('0. DATA INPUT-OUTPUT DIMUAT\n');
fprintf('========================================\n');
fprintf('Jumlah sampel: %d\n', N);
fprintf('Sampling time: %.4f\n', Ts);
fprintf('========================================\n\n');

%% ===========================================
% 1. PLOT DATA INPUT-OUTPUT
% ============================================
fprintf('========================================\n');
fprintf('1. PLOT DATA INPUT-OUTPUT\n');
fprintf('========================================\n');

figure('Name', '1. Plot Data IO', 'Position', [100 100 900 500]); 
subplot(2,1,1); 
plot(y, 'b-', 'LineWidth', 1.5); grid on;
title('Output y(k)', 'FontWeight', 'bold');
xlabel('Sample (k)'); ylabel('y(k)');

subplot(2,1,2); 
plot(u, 'r-', 'LineWidth', 1.5); grid on;
title('Input u(k)', 'FontWeight', 'bold');
xlabel('Sample (k)'); ylabel('u(k)');

fprintf('✓ Plot data selesai.\n');
fprintf('========================================\n\n');

%% ===========================================
% 2. CEK KORELASI ANTARA INPUT DAN OUTPUT
% ============================================
fprintf('========================================\n');
fprintf('2. CEK KORELASI INPUT-OUTPUT\n');
fprintf('========================================\n');

% 2A. Korelasi global (lag 0)
R_yu = corrcoef(y, u);
rho_yu = R_yu(1,2);
fprintf('Matriks korelasi (y,u):\n');
disp(R_yu);
fprintf('Korelasi linier (lag-0): %.4f\n\n', rho_yu);

% 2B. Scatter plot
figure('Name', '2. Scatter Plot IO');
plot(u, y, 'b.', 'MarkerSize', 8);
grid on;
xlabel('Input u(k)'); ylabel('Output y(k)');
title('Scatter Plot: Input vs Output', 'FontWeight', 'bold');

% 2C. Auto & Cross Correlation
maxLag = min(30, N-1);
conf   = 2/sqrt(N);

[Ru, lags]  = xcorr(u - mean(u), maxLag, 'coeff');
[Ry, ~]     = xcorr(y - mean(y), maxLag, 'coeff');
[Ryu, ~]    = xcorr(y - mean(y), u - mean(u), maxLag, 'coeff');

figure('Name', '2. Correlation Analysis', 'Position', [100 100 1000 700]);
tiledlayout(3,1);

nexttile;
stem(lags, Ru, 'filled'); hold on;
yline(0,'k-'); yline(conf,'r--'); yline(-conf,'r--');
grid on; xlabel('Lag'); ylabel('R_{uu}(\tau)');
title('Autokorelasi Input u(k)', 'FontWeight', 'bold');
legend('R_{uu}','0','\pm 2/\surdN','Location','best');

nexttile;
stem(lags, Ry, 'filled'); hold on;
yline(0,'k-'); yline(conf,'r--'); yline(-conf,'r--');
grid on; xlabel('Lag'); ylabel('R_{yy}(\tau)');
title('Autokorelasi Output y(k)', 'FontWeight', 'bold');
legend('R_{yy}','0','\pm 2/\surdN','Location','best');

nexttile;
stem(lags, Ryu, 'filled'); hold on;
yline(0,'k-'); yline(conf,'r--'); yline(-conf,'r--');
grid on; xlabel('Lag'); ylabel('R_{yu}(\tau)');
title('Cross-korelasi Input-Output (y vs u)', 'FontWeight', 'bold');
legend('R_{yu}','0','\pm 2/\surdN','Location','best');

fprintf('✓ Analisis korelasi selesai.\n');
fprintf('========================================\n\n');

%% ===========================================
% 3. CEK STASIONERITAS
% ============================================
fprintf('========================================\n');
fprintf('3. CEK STASIONERITAS DATA\n');
fprintf('========================================\n');

% Uji ADF pada data asli
[h_y, p_y] = adftest(y);
[h_u, p_u] = adftest(u);

fprintf('ADF Test - Data Asli:\n');
fprintf('  Output y: h=%d, p-value=%.4f', h_y, p_y);
if h_y == 1
    fprintf(' → STASIONER\n');
else
    fprintf(' → TIDAK STASIONER\n');
end

fprintf('  Input u : h=%d, p-value=%.4f', h_u, p_u);
if h_u == 1
    fprintf(' → STASIONER\n');
else
    fprintf(' → TIDAK STASIONER\n');
end

% Cek apakah perlu stasionerkan
if (h_y == 1) && (h_u == 1)
    fprintf('\n✓ KESIMPULAN: Data SUDAH STASIONER.\n');
    fprintf('  Tidak perlu detrending.\n');
    y_stat = y;
    u_stat = u;
    is_detrended = false;
else
    fprintf('\n✗ KESIMPULAN: Data TIDAK STASIONER.\n');
    fprintf('  Melakukan detrending...\n\n');
    
    % Detrend data
    y_stat = detrend(y);
    u_stat = detrend(u);
    
    % Uji ulang setelah detrending
    [h_y_stat, p_y_stat] = adftest(y_stat);
    [h_u_stat, p_u_stat] = adftest(u_stat);
    
    fprintf('ADF Test - Setelah Detrending:\n');
    fprintf('  Output y: h=%d, p-value=%.4f', h_y_stat, p_y_stat);
    if h_y_stat == 1
        fprintf(' → STASIONER\n');
    else
        fprintf(' → MASIH TIDAK STASIONER\n');
    end
    
    fprintf('  Input u : h=%d, p-value=%.4f', h_u_stat, p_u_stat);
    if h_u_stat == 1
        fprintf(' → STASIONER\n');
    else
        fprintf(' → MASIH TIDAK STASIONER\n');
    end
    
    if (h_y_stat == 1) && (h_u_stat == 1)
        fprintf('\n✓ KESIMPULAN: Data BERHASIL DISTASIONERKAN.\n');
        is_detrended = true;
    else
        error(['Data MASIH TIDAK STASIONER setelah detrending.\n', ...
               'Pertimbangkan differencing atau transformasi lain.']);
    end
end

% Plot data stasioner
figure('Name', '3. Data Stasioner', 'Position', [100 100 900 500]);
subplot(2,1,1);
plot(y_stat, 'b-', 'LineWidth', 1.5); grid on;
title('Output Stasioner y_{stat}(k)', 'FontWeight', 'bold');
xlabel('Sample (k)'); ylabel('y_{stat}(k)');

subplot(2,1,2);
plot(u_stat, 'r-', 'LineWidth', 1.5); grid on;
title('Input Stasioner u_{stat}(k)', 'FontWeight', 'bold');
xlabel('Sample (k)'); ylabel('u_{stat}(k)');

fprintf('========================================\n\n');

%% ===========================================
% 4. TENTUKAN PEMILIHAN MODEL (ARX-IO)
% ============================================
fprintf('========================================\n');
fprintf('4. PEMILIHAN MODEL\n');
fprintf('========================================\n');
fprintf('Model dipilih: ARX (AutoRegressive with eXogenous input)\n');
fprintf('Struktur: A(q) y(k) = B(q) u(k-nk) + e(k)\n');
fprintf('Parameter: [na nb nk]\n');
fprintf('  na = orde AR output\n');
fprintf('  nb = orde input\n');
fprintf('  nk = delay (transport lag)\n');
fprintf('========================================\n\n');

% Buat iddata dari data stasioner
z2_stat = iddata(y_stat, u_stat, Ts);

%% ===========================================
% 5. SPLIT DATA TRAIN - TEST (70% / 30%)
% ============================================
fprintf('========================================\n');
fprintf('5. SPLIT DATA TRAIN-TEST\n');
fprintf('========================================\n');

N_stat = length(y_stat);
idx_split = round(0.7 * N_stat);

data_train = z2_stat(1:idx_split);
data_test  = z2_stat(idx_split+1:end);

y_train = data_train.y;
u_train = data_train.u;
y_test  = data_test.y;
u_test  = data_test.u;

fprintf('Total data: %d sampel\n', N_stat);
fprintf('Data Train: %d sampel (%.1f%%)\n', idx_split, 100*idx_split/N_stat);
fprintf('Data Test : %d sampel (%.1f%%)\n', N_stat-idx_split, 100*(N_stat-idx_split)/N_stat);
fprintf('✓ Split data selesai.\n');
fprintf('========================================\n\n');

%% ===========================================
% 6. PEMILIHAN ORDE ARX DENGAN AIC (GRID SEARCH)
% ============================================
fprintf('========================================\n');
fprintf('6. PEMILIHAN ORDE ARX DENGAN AIC\n');
fprintf('========================================\n');

na_list = 1:10;
nb_list = 1:10;
nk_list = 0:2;

best_model = [];
best_aic   = inf;
best_order = [];

fprintf('Grid Search: na=[%d:%d], nb=[%d:%d], nk=[%d:%d]\n', ...
        min(na_list), max(na_list), min(nb_list), max(nb_list), ...
        min(nk_list), max(nk_list));
fprintf('Mencari kombinasi orde terbaik...\n\n');

results = [];  % [na nb nk AIC FIT_train]

for na = na_list
    for nb = nb_list
        for nk = nk_list
            try
                % Training model ARX
                m = arx(data_train, [na nb nk]);
                
                % Hitung residual
                e_id = resid(data_train, m);
                if isprop(e_id, 'OutputData')
                    e = e_id.OutputData(:);
                else
                    e = e_id.y(:);
                end
                
                % Hitung AIC
                N_e = length(e);
                sigma2_hat = mean(e.^2);
                k_param = na + nb;
                aic_val = 2*k_param + N_e*log(sigma2_hat);
                
                % Hitung FIT
                [~, fit_train, ~] = compare(data_train, m);
                
                % Simpan hasil
                results = [results; na nb nk aic_val fit_train];
                
                % Update model terbaik
                if aic_val < best_aic
                    best_aic   = aic_val;
                    best_model = m;
                    best_order = [na nb nk];
                end
            catch
                % Skip jika error
            end
        end
    end
end

fprintf('✓ Grid Search selesai!\n');
fprintf('----------------------------------------\n');
fprintf('MODEL TERBAIK:\n');
fprintf('  Orde [na nb nk] = [%d %d %d]\n', best_order);
fprintf('  AIC             = %.4f\n', best_aic);
fprintf('----------------------------------------\n\n');

% Tampilkan 5 model terbaik
[~, idx_sorted] = sort(results(:,4));
top5 = results(idx_sorted(1:min(5,end)), :);

fprintf('Top 5 Model (berdasarkan AIC):\n');
fprintf('  na  nb  nk     AIC      FIT(%%)\n');
fprintf('  --------------------------------\n');
for i = 1:size(top5,1)
    fprintf('  %2d  %2d  %2d   %8.2f   %6.2f\n', ...
            top5(i,1), top5(i,2), top5(i,3), top5(i,4), top5(i,5));
end
fprintf('========================================\n\n');

%% Tampilkan parameter model
fprintf('Model ARX Terbaik:\n');
present(best_model);
fprintf('\n');

%% ===========================================
% 7. EVALUASI MSE, RMSE, MAE, MAPE
% ============================================
fprintf('========================================\n');
fprintf('7. EVALUASI METRIK ERROR\n');
fprintf('========================================\n');

% Prediksi TRAIN
[yhat_train_id, fit_train] = compare(data_train, best_model);
if isprop(yhat_train_id, 'OutputData')
    yhat_train = yhat_train_id.OutputData(:);
else
    yhat_train = yhat_train_id.y(:);
end

% Prediksi TEST
[yhat_test_id, fit_test] = compare(data_test, best_model);
if isprop(yhat_test_id, 'OutputData')
    yhat_test = yhat_test_id.OutputData(:);
else
    yhat_test = yhat_test_id.y(:);
end

% Fungsi metrik
MSE_func  = @(y,yh) mean((y - yh).^2);
RMSE_func = @(y,yh) sqrt(mean((y - yh).^2));
MAE_func  = @(y,yh) mean(abs(y - yh));
MAPE_func = @(y,yh) mean(abs((y - yh)./y))*100;

% Hitung metrik TRAIN
MSE_train  = MSE_func(y_train, yhat_train);
RMSE_train = RMSE_func(y_train, yhat_train);
MAE_train  = MAE_func(y_train, yhat_train);
MAPE_train = MAPE_func(y_train, yhat_train);

% Hitung metrik TEST
MSE_test  = MSE_func(y_test, yhat_test);
RMSE_test = RMSE_func(y_test, yhat_test);
MAE_test  = MAE_func(y_test, yhat_test);
MAPE_test = MAPE_func(y_test, yhat_test);

% Tampilkan hasil
fprintf('METRIK EVALUASI - DATA TRAIN:\n');
fprintf('  FIT  = %.2f %%\n', fit_train);
fprintf('  MSE  = %.6f\n', MSE_train);
fprintf('  RMSE = %.6f\n', RMSE_train);
fprintf('  MAE  = %.6f\n', MAE_train);
fprintf('  MAPE = %.4f %%\n\n', MAPE_train);

fprintf('METRIK EVALUASI - DATA TEST:\n');
fprintf('  FIT  = %.2f %%\n', fit_test);
fprintf('  MSE  = %.6f\n', MSE_test);
fprintf('  RMSE = %.6f\n', RMSE_test);
fprintf('  MAE  = %.6f\n', MAE_test);
fprintf('  MAPE = %.4f %%\n', MAPE_test);
fprintf('========================================\n\n');

% Plot perbandingan
figure('Name', '7. Model Validation', 'Position', [100 100 1200 500]);
subplot(1,2,1);
compare(data_train, best_model);
title(sprintf('TRAIN: ARX[%d %d %d] - FIT=%.2f%%', ...
      best_order(1), best_order(2), best_order(3), fit_train), ...
      'FontWeight', 'bold');
grid on;

subplot(1,2,2);
compare(data_test, best_model);
title(sprintf('TEST: ARX[%d %d %d] - FIT=%.2f%%', ...
      best_order(1), best_order(2), best_order(3), fit_test), ...
      'FontWeight', 'bold');
grid on;

%% ===========================================
% 8. CEK RESIDU DENGAN LJUNG-BOX TEST
% ============================================
fprintf('========================================\n');
fprintf('8. UJI RESIDU (LJUNG-BOX WHITE NOISE TEST)\n');
fprintf('========================================\n');

% Ekstrak residu dari data TEST
e_id_test = resid(best_model, data_test);
if isprop(e_id_test, 'OutputData')
    e_test = e_id_test.OutputData(:);
else
    e_test = e_id_test.y(:);
end

% Uji Ljung-Box untuk white noise
n_lags = min(20, length(e_test)-1);
[h_lb, pValue_lb] = lbqtest(e_test, 'Lags', n_lags);

fprintf('Ljung-Box Test (Lags=%d):\n', n_lags);
fprintf('  H (hypothesis) = %d\n', h_lb);
fprintf('  p-value        = %.4f\n\n', pValue_lb);

fprintf('KESIMPULAN:\n');
if h_lb == 0
    fprintf('  ✓ p-value (%.4f) > 0.05 → H0 DITERIMA\n', pValue_lb);
    fprintf('  ✓ Residu adalah WHITE NOISE\n');
    fprintf('  ✓ Model ARX BAIK (tidak ada autokorelasi)\n');
    fprintf('  ✓ Model sudah menangkap semua informasi dalam data\n');
else
    fprintf('  ✗ p-value (%.4f) < 0.05 → H0 DITOLAK\n', pValue_lb);
    fprintf('  ✗ Residu BUKAN WHITE NOISE\n');
    fprintf('  ✗ Model ARX KURANG BAIK (masih ada autokorelasi)\n');
    fprintf('  ✗ Masih ada pola yang belum tertangkap oleh model\n\n');
    fprintf('  SARAN PERBAIKAN:\n');
    fprintf('    1. Ubah orde model (kembali ke Step 6)\n');
    fprintf('    2. Pertimbangkan model nonlinier (NLARX)\n');
    fprintf('    3. Tambah variabel eksogen lain\n');
end
fprintf('========================================\n\n');

% Plot analisis residu
figure('Name', '8. Residual Analysis');
resid(best_model, data_test);
sgtitle('Analisis Residu Model ARX', 'FontWeight', 'bold', 'FontSize', 12);

%% ===========================================
% 9. BUAT MODEL MATEMATIKA
% ============================================
fprintf('========================================\n');
fprintf('9. MODEL MATEMATIKA ARX\n');
fprintf('========================================\n');

% Ekstrak parameter model
na_m = best_model.na;
nb_m = best_model.nb;
nk_m = best_model.nk;

A = best_model.A;
if iscell(best_model.B)
    B = best_model.B{1};
else
    B = best_model.B;
end

fprintf('Orde Model: na=%d, nb=%d, nk=%d\n\n', na_m, nb_m, nk_m);

fprintf('Koefisien Polinomial:\n');
fprintf('  A(q^-1) = [');
fprintf(' %.6f', A);
fprintf(' ]\n');
fprintf('  B(q^-1) = [');
fprintf(' %.6f', B);
fprintf(' ]\n\n');

fprintf('Bentuk Umum:\n');
fprintf('  A(q^-1) y(k) = B(q^-1) u(k-%d) + e(k)\n\n', nk_m);

% Cetak persamaan difference lengkap
fprintf('PERSAMAAN DIFFERENCE LENGKAP:\n');
fprintf('y(k) = ');

first = true;

% Bagian y(k-i)
for i = 2:length(A)
    ai = -A(i);
    if abs(ai) < 1e-10
        continue;
    end
    
    if first
        fprintf('%.6f*y(k-%d)', ai, i-1);
        first = false;
    else
        if ai >= 0
            fprintf(' + %.6f*y(k-%d)', ai, i-1);
        else
            fprintf(' - %.6f*y(k-%d)', abs(ai), i-1);
        end
    end
end

% Bagian u(k-j)
for j = 1:length(B)
    bj = B(j);
    if abs(bj) < 1e-10
        continue;
    end
    lag_u = nk_m + (j-1);
    
    if first
        fprintf('%.6f*u(k-%d)', bj, lag_u);
        first = false;
    else
        if bj >= 0
            fprintf(' + %.6f*u(k-%d)', bj, lag_u);
        else
            fprintf(' - %.6f*u(k-%d)', abs(bj), lag_u);
        end
    end
end

fprintf(' + e(k)\n');
fprintf('========================================\n\n');

%% ===========================================
% RINGKASAN AKHIR
% ============================================
fprintf('========================================\n');
fprintf('RINGKASAN HASIL IDENTIFIKASI SISTEM ARX\n');
fprintf('========================================\n');
fprintf('Data:\n');
fprintf('  Jumlah sampel       : %d\n', N);
fprintf('  Data stasioner      : %s\n', ternary(is_detrended, 'Ya (detrended)', 'Ya (asli)'));
fprintf('  Train / Test        : %d / %d sampel\n\n', idx_split, N_stat-idx_split);

fprintf('Model Terbaik:\n');
fprintf('  Struktur            : ARX [%d %d %d]\n', best_order);
fprintf('  AIC                 : %.4f\n\n', best_aic);

fprintf('Performa Train:\n');
fprintf('  FIT                 : %.2f %%\n', fit_train);
fprintf('  MSE                 : %.6f\n', MSE_train);
fprintf('  RMSE                : %.6f\n', RMSE_train);
fprintf('  MAE                 : %.6f\n', MAE_train);
fprintf('  MAPE                : %.4f %%\n\n', MAPE_train);

fprintf('Performa Test:\n');
fprintf('  FIT                 : %.2f %%\n', fit_test);
fprintf('  MSE                 : %.6f\n', MSE_test);
fprintf('  RMSE                : %.6f\n', RMSE_test);
fprintf('  MAE                 : %.6f\n', MAE_test);
fprintf('  MAPE                : %.4f %%\n\n', MAPE_test);

fprintf('Validasi Residu:\n');
fprintf('  Ljung-Box p-value   : %.4f\n', pValue_lb);
fprintf('  Residu white noise  : %s\n', ternary(h_lb==0, 'YA ✓', 'TIDAK ✗'));
fprintf('  Status model        : %s\n', ternary(h_lb==0, 'BAIK ✓', 'PERLU PERBAIKAN ✗'));

fprintf('========================================\n');
fprintf('IDENTIFIKASI SISTEM SELESAI!\n');
fprintf('========================================\n');

%% Helper function
function result = ternary(condition, true_val, false_val)
    if condition
        result = true_val;
    else
        result = false_val;
    end
end