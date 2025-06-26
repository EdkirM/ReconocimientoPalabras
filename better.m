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

%% ==== SVM ====
disp('🔍 Grid Search SVM (Palabra)...');

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

ypred_svm = predict(bestSVMModel, Xtest);
acc_svm = mean(ypred_svm == ytest_palabra);
fprintf('\n✅ SVM Palabra Accuracy: %.2f%%\n', acc_svm * 100);
save('modelo_palabra.mat', 'bestSVMModel', 'acc_svm');

%% ==== Random Forest ====
disp('🌲 Grid Search Random Forest (Palabra)...');

numTreesVals = [100, 200, 300];
minLeafVals = [1, 5, 10];

bestAccRF = 0;
rfPalabraModel = [];

for t = 1:length(numTreesVals)
    for l = 1:length(minLeafVals)
        modelo = TreeBagger(numTreesVals(t), Xtrain, ytrain_palabra, ...
                            'MinLeafSize', minLeafVals(l), ...
                            'Method', 'classification', ...
                            'OOBPrediction', 'On');
        acc = 1 - oobError(modelo, 'Mode', 'ensemble');
        if acc > bestAccRF
            bestAccRF = acc;
            rfPalabraModel = modelo;
        end
    end
end

ypred_rf = predict(rfPalabraModel, Xtest);
ypred_rf = categorical(ypred_rf);
acc_rf = mean(ypred_rf == ytest_palabra);
fprintf('✅ RF Palabra Accuracy: %.2f%%\n', acc_rf * 100);
save('modelo_palabra_rf.mat', 'rfPalabraModel', 'acc_rf');

%% ==== k-NN ====
disp('📌 Grid Search k-NN (Palabra)...');

kVals = [1, 3, 5, 7, 9];
distanceMetrics = {'euclidean', 'cityblock'};

bestAccKNN = 0;
bestKNNModel = [];

for k = 1:length(kVals)
    for d = 1:length(distanceMetrics)
        modelo = fitcknn(Xtrain, ytrain_palabra, ...
                         'NumNeighbors', kVals(k), ...
                         'Distance', distanceMetrics{d}, ...
                         'CrossVal', 'on', ...
                         'KFold', 5);
        acc = 1 - kfoldLoss(modelo);
        if acc > bestAccKNN
            bestAccKNN = acc;
            bestKNNModel = fitcknn(Xtrain, ytrain_palabra, ...
                                   'NumNeighbors', kVals(k), ...
                                   'Distance', distanceMetrics{d});
        end
    end
end

ypred_knn = predict(bestKNNModel, Xtest);
acc_knn = mean(ypred_knn == ytest_palabra);
fprintf('✅ k-NN Palabra Accuracy: %.2f%%\n', acc_knn * 100);
save('modelo_palabra_knn.mat', 'bestKNNModel', 'acc_knn');

%% ==== LOCUTOR (Random Forest) ====
disp('🧑‍🏫 Entrenando RF para Locutor...');

bestRFModel = [];
bestAccRF_locutor = 0;

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

ypred_locutor = predict(bestRFModel, Xtest);
ypred_locutor = categorical(ypred_locutor);
acc_rf_locutor = mean(ypred_locutor == ytest_locutor);
fprintf('✅ RF Locutor Accuracy: %.2f%%\n', acc_rf_locutor * 100);
save('modelo_locutor.mat', 'bestRFModel', 'acc_rf_locutor');

%% ==== FUNCIÓN DE VOTACIÓN ====
voto_mayoria = @(a,b,c) mode(categorical({char(a), char(b), char(c)}));

ypred_voto = strings(size(ypred_svm));
for i = 1:length(ypred_svm)
    ypred_voto(i) = voto_mayoria(ypred_svm(i), ypred_rf(i), ypred_knn(i));
end
ypred_voto = categorical(ypred_voto);

%% ==== MATRICES DE CONFUSIÓN ====
figure; confusionchart(ytest_palabra, ypred_svm);       title('SVM - Palabra');
figure; confusionchart(ytest_palabra, ypred_rf);        title('Random Forest - Palabra');
figure; confusionchart(ytest_palabra, ypred_knn);       title('k-NN - Palabra');
figure; confusionchart(ytest_palabra, ypred_voto);      title('Votación por mayoría');
figure; confusionchart(ytest_locutor, ypred_locutor);   title('Random Forest - Locutor');

disp('🎯 Entrenamiento y votación completados.');
