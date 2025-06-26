clc; clear;

%% === 1. Grabación de audio ===
fs = 16000;
duracion = 2;

disp('🎙️ Comienza la grabación...');
recObj = audiorecorder(fs, 16, 1);
recordblocking(recObj, duracion);
disp('✅ Grabación completa.');

audio = getaudiodata(recObj);
audio = audio / max(abs(audio) + eps);  % Normalización

audiowrite('audio_grabado.wav', audio, fs);  % Opcional

%% === 2. Extracción de características (MFCC) ===
[coeffs, delta, deltaDelta] = mfcc(audio, fs);
features = [mean(coeffs); std(coeffs); ...
            mean(delta); std(delta); ...
            mean(deltaDelta); std(deltaDelta)];
features = features(:)';

%% === 3. Cargar modelos ===
if ~isfile('modelo_palabra.mat') || ...
   ~isfile('modelo_palabra_rf.mat') || ...
   ~isfile('modelo_palabra_knn.mat') || ...
   ~isfile('modelo_locutor.mat')
    error('❌ Faltan modelos entrenados. Ejecuta el script de entrenamiento primero.');
end

load('modelo_palabra.mat', 'bestSVMModel');
load('modelo_palabra_rf.mat', 'rfPalabraModel');
load('modelo_palabra_knn.mat', 'bestKNNModel');
load('modelo_locutor.mat', 'bestRFModel');

%% === 4. Predicciones ===
% Palabra
pred_svm  = predict(bestSVMModel, features);
pred_rf   = predict(rfPalabraModel, features);
pred_rf   = categorical(pred_rf);
pred_knn  = predict(bestKNNModel, features);

% Locutor
pred_locutor = predict(bestRFModel, features);
pred_locutor = categorical(pred_locutor);

%% === 5. Votación por mayoría ===
voto_mayoria = @(a, b, c) mode(categorical({char(a), char(b), char(c)}));
pred_final = voto_mayoria(pred_svm, pred_rf, pred_knn);

%% === 6. Mostrar resultados ===
fprintf('\n🎧 Clasificación del audio grabado:\n');
fprintf('🔤 SVM:       %s\n', string(pred_svm));
fprintf('🔤 RF:        %s\n', string(pred_rf));
fprintf('🔤 k-NN:      %s\n', string(pred_knn));
fprintf('🗳️  Votación:  %s\n', string(pred_final));
fprintf('🧑‍🏫 Locutor:   %s\n', string(pred_locutor));
