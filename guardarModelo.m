clc; clear;

% Cargar características
load('caracteristicas.mat');  % X, y, locutores

y_palabra = categorical(y);
y_locutor = categorical(locutores);

% División 80/20 estratificada por palabra
cvHoldout = cvpartition(y_palabra, 'HoldOut', 0.2);

Xtrain = X(training(cvHoldout), :);
Xtest  = X(test(cvHoldout), :);
ytrain_palabra = y_palabra(training(cvHoldout));
ytest_palabra  = y_palabra(test(cvHoldout));
ytrain_locutor = y_locutor(training(cvHoldout));
ytest_locutor  = y_locutor(test(cvHoldout));

%% ==== ENTRENAMIENTO PALABRA (SVM - RBF) ====
disp('🔍 Buscando mejores parámetros para SVM (Palabra)...');

kernelScaleVals = logspace(-2, 2, 5);      % [0.01 0.1 1 10 100]
boxConstraintVals = logspace(-2, 2, 5);

bestAccSVM = 0;
bestSVMModel = [];

for k = 1:length(kernelScaleVals)
    for b = 1:length(boxConstraintVals)
        t = templateSVM('KernelFunction', 'rbf', ...
                        'KernelScale', kernelScaleVals(k), ...
                        'BoxConstraint', boxConstraintVals(b));
        modelo = fitcecoc(Xtrain, ytrain_palabra, 'Learners', t, 'KFold', 5);
        acc = 1 - kfoldLoss(modelo);

        if acc > bestAccSVM
            bestAccSVM = acc;
            bestSVMModel = fitcecoc(Xtrain, ytrain_palabra, 'Learners', t);
        end
    end
end

% Evaluar
ypred_svm = predict(bestSVMModel, Xtest);
acc_svm = mean(ypred_svm == ytest_palabra);
fprintf('\n✅ SVM Palabra Accuracy: %.2f%%\n', acc_svm * 100);

% Guardar modelo SVM
save('modelo_palabra.mat', 'bestSVMModel', 'acc_svm');

%% ==== ENTRENAMIENTO PALABRA (Random Forest) ====
disp('🌲 Buscando mejores parámetros para RF (Palabra)...');

numTreesVals = [100, 200, 300];
minLeafVals = [1, 5, 10];

bestAccRF_palabra = 0;
rfPalabraModel = [];

for t = 1:length(numTreesVals)
    for l = 1:length(minLeafVals)
        modelo = TreeBagger(numTreesVals(t), Xtrain, ytrain_palabra, ...
                            'MinLeafSize', minLeafVals(l), ...
                            'Method', 'classification', ...
                            'OOBPrediction', 'On');
        acc = 1 - oobError(modelo, 'Mode', 'ensemble');
        if acc > bestAccRF_palabra
            bestAccRF_palabra = acc;
            rfPalabraModel = modelo;
        end
    end
end

% Evaluar
ypred_rf_palabra = predict(rfPalabraModel, Xtest);
ypred_rf_palabra = categorical(ypred_rf_palabra);
acc_rf_palabra = mean(ypred_rf_palabra == ytest_palabra);
fprintf('✅ RF Palabra Accuracy: %.2f%%\n', acc_rf_palabra * 100);

% Guardar modelo RF para palabra
save('modelo_palabra_rf.mat', 'rfPalabraModel', 'acc_rf_palabra');

%% ==== ENTRENAMIENTO LOCUTOR (Random Forest) ====
disp('🧑‍🏫 Entrenando RF para LOCUTOR...');

bestAccRF_locutor = 0;
bestRFModel = [];

for t = 1:length(numTreesVals)
    for l = 1:length(minLeafVals)
        modelo = TreeBagger(numTreesVals(t), Xtrain, ytrain_locutor, ...
                            'MinLeafSize', minLeafVals(l), ...
                            'Method', 'classification', ...
                            'OOBPrediction', 'On');
        acc = 1 - oobError(modelo, 'Mode', 'ensemble');
        if acc > bestAccRF_locutor
            bestAccRF_locutor = acc;
            bestRFModel = modelo;
        end
    end
end

% Evaluar
ypred_rf_locutor = predict(bestRFModel, Xtest);
ypred_rf_locutor = categorical(ypred_rf_locutor);
acc_rf_locutor = mean(ypred_rf_locutor == ytest_locutor);
fprintf('✅ RF Locutor Accuracy: %.2f%%\n', acc_rf_locutor * 100);

% Guardar modelo RF para locutor
save('modelo_locutor.mat', 'bestRFModel', 'acc_rf_locutor');

%% ==== MATRICES DE CONFUSIÓN ====
figure;
confusionchart(ytest_palabra, ypred_svm);
title('Matriz de confusión - Palabra (SVM)');

figure;
confusionchart(ytest_palabra, ypred_rf_palabra);
title('Matriz de confusión - Palabra (RF)');

figure;
confusionchart(ytest_locutor, ypred_rf_locutor);
title('Matriz de confusión - Locutor (RF)');

disp('🎯 Entrenamiento completado y modelos guardados.');
