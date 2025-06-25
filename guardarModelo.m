clc; clear;

% Cargar características y etiquetas
load('caracteristicas.mat');  % Variables: X, y
y = categorical(y);

% División fija de entrenamiento/prueba (80/20)
cvHoldout = cvpartition(y, 'HoldOut', 0.2);
Xtrain = X(training(cvHoldout), :);
ytrain = y(training(cvHoldout));
Xtest  = X(test(cvHoldout), :);
ytest  = y(test(cvHoldout));

%% ==== GRID SEARCH PARA SVM (RBF Kernel) ====
disp('Buscando mejores parámetros para SVM...');

% Hiperparámetros a explorar
kernelScaleVals = [0.1, 1, 10];
boxConstraintVals = [0.1, 1, 10];

bestAccSVM = 0;
bestSVMModel = [];

for k = 1:length(kernelScaleVals)
    for b = 1:length(boxConstraintVals)
        t = templateSVM('KernelFunction', 'rbf', ...
                        'KernelScale', kernelScaleVals(k), ...
                        'BoxConstraint', boxConstraintVals(b));
        modelo = fitcecoc(Xtrain, ytrain, 'Learners', t, 'KFold', 5);
        acc = 1 - kfoldLoss(modelo);

        fprintf('SVM RBF - Scale: %.2f, Box: %.2f => Acc: %.2f%%\n', ...
                kernelScaleVals(k), boxConstraintVals(b), acc * 100);

        if acc > bestAccSVM
            bestAccSVM = acc;
            bestSVMModel = fitcecoc(Xtrain, ytrain, 'Learners', t);
        end
    end
end

% Evaluación final
ypredSVM = predict(bestSVMModel, Xtest);
accFinalSVM = sum(ypredSVM == ytest) / numel(ytest);
fprintf('\n🔍 Mejor SVM => Acc: %.2f%%\n', accFinalSVM * 100);

%% ==== GRID SEARCH PARA RANDOM FOREST ====
disp('Buscando mejores parámetros para Random Forest...');

numTreesVals = [50, 100, 200];
minLeafVals = [1, 5, 10];

bestAccRF = 0;
bestRFModel = [];

for t = 1:length(numTreesVals)
    for l = 1:length(minLeafVals)
        modelo = TreeBagger(numTreesVals(t), Xtrain, ytrain, ...
                            'MinLeafSize', minLeafVals(l), ...
                            'Method', 'classification', ...
                            'OOBPrediction', 'On');
        oobErr = oobError(modelo, 'Mode', 'ensemble');
        acc = 1 - oobErr(end);

        fprintf('RF - Trees: %d, MinLeaf: %d => OOB Acc: %.2f%%\n', ...
                numTreesVals(t), minLeafVals(l), acc * 100);

        if acc > bestAccRF
            bestAccRF = acc;
            bestRFModel = modelo;
        end
    end
end

% Evaluación final
ypredRF = predict(bestRFModel, Xtest);
ypredRF = categorical(ypredRF);
accFinalRF = sum(ypredRF == ytest) / numel(ytest);
fprintf('\n🔍 Mejor RF => Acc: %.2f%%\n', accFinalRF * 100);

%% ==== Selección y guardado del mejor modelo ====
if accFinalSVM >= accFinalRF
    mejorModelo = bestSVMModel;
    tipoModelo = 'SVM';
    mejorAccuracy = accFinalSVM;
else
    mejorModelo = bestRFModel;
    tipoModelo = 'RandomForest';
    mejorAccuracy = accFinalRF;
end

% Guardar en archivo .mat
save('mejor_modelo.mat', 'mejorModelo', 'tipoModelo', 'mejorAccuracy');
fprintf('\n✅ Modelo guardado como mejor_modelo.mat (%s - %.2f%%)\n', tipoModelo, mejorAccuracy * 100);

%% ==== Matrices de confusión ====
figure;
confusionchart(ytest, ypredSVM);
title('Matriz de confusión - Mejor SVM');

figure;
confusionchart(ytest, ypredRF);
title('Matriz de confusión - Mejor Random Forest');
