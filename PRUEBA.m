clc; clear;

%% === 1. Grabación de audio ===
fs = 16000;          % Frecuencia de muestreo (igual que en entrenamiento)
duracion = 2;        % Duración de la grabación en segundos

disp('🎙️ Comienza la grabación...');
recObj = audiorecorder(fs, 16, 1);  % 16 bits, 1 canal (mono)
recordblocking(recObj, duracion);
disp('✅ Grabación completa.');

% Obtener y normalizar audio
audio = getaudiodata(recObj);
audio = audio / max(abs(audio));  % Normalización

% Guardar audio (opcional)
audiowrite('audio_grabado.wav', audio, fs);

%% === 2. Extracción de características (MFCC) ===
[coeffs, delta, deltaDelta] = mfcc(audio, fs);
features = [mean(coeffs); std(coeffs); ...
            mean(delta); std(delta); ...
            mean(deltaDelta); std(deltaDelta)];
features = features(:)';  % Convertir a vector fila

%% === 3. Cargar modelo guardado ===
if ~isfile('mejor_modelo.mat')
    error('❌ Archivo "mejor_modelo.mat" no encontrado. Ejecuta primero el entrenamiento.');
end

load('mejor_modelo.mat', 'mejorModelo', 'tipoModelo', 'mejorAccuracy');

%% === 4. Clasificación ===
switch tipoModelo
    case 'SVM'
        pred = predict(mejorModelo, features);
    case 'RandomForest'
        pred = predict(mejorModelo, features);
        pred = categorical(pred);  % convertir string/cell a categoría
    otherwise
        error('Modelo desconocido: %s', tipoModelo);
end

%% === 5. Mostrar resultado ===
fprintf('🔊 Palabra detectada: %s\n', string(pred));
fprintf('🤖 Modelo usado: %s (precisión de entrenamiento %.2f%%)\n', tipoModelo, mejorAccuracy * 100);
