clc; clear;

% Cargar datos procesados
load('caracteristicas.mat');  % Variables: X, y

% Codificar etiquetas como categorías
y = categorical(y);

% Dividir en entrenamiento y prueba (80/20)
cv = cvpartition(y, 'HoldOut', 0.2);
Xtrain = X(training(cv), :);
ytrain = y(training(cv));
Xtest  = X(test(cv), :);
ytest  = y(test(cv));

%% ====== Entrenamiento con SVM ======
disp('Entrenando SVM...');
modeloSVM = fitcecoc(Xtrain, ytrain);  % SVM multiclase usando ECOC

% Predicción
ypredSVM = predict(modeloSVM, Xtest);

% Evaluación
accuracySVM = sum(ypredSVM == ytest) / numel(ytest);
fprintf('Precisión SVM: %.2f%%\n', accuracySVM * 100);

%% ====== Entrenamiento con Random Forest ======
disp('Entrenando Random Forest...');
modeloRF = TreeBagger(100, Xtrain, ytrain, ...
                      'OOBPrediction', 'On', ...
                      'Method', 'classification');

% Predicción
ypredRF = predict(modeloRF, Xtest);
ypredRF = categorical(ypredRF);  % TreeBagger devuelve celdas

% Evaluación
accuracyRF = sum(ypredRF == ytest) / numel(ytest);
fprintf('Precisión Random Forest: %.2f%%\n', accuracyRF * 100);

%% ====== Matrices de confusión ======
figure;
confusionchart(ytest, ypredSVM);
title('Matriz de confusión - SVM');

figure;
confusionchart(ytest, ypredRF);
title('Matriz de confusión - Random Forest');
