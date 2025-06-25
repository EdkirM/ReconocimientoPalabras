% Configuración inicial
clc; clear;

% Carpeta base donde están los audios
carpetaBase = 'Audios';  % Cambia esto a tu ruta

% Palabras objetivo
palabras = {'Casa', 'Lluvia', 'Nube', 'Perro', 'Tren'};

% Inicializa matrices
X = [];      % Características
y = {};      % Etiquetas (palabras)

% Parámetros
fsEsperado = 16000;  % Frecuencia de muestreo esperada

% Recorre carpetas por persona y palabra
personas = dir(carpetaBase);
personas = personas([personas.isdir] & ~startsWith({personas.name}, '.'));

for i = 1:length(personas)
    nombrePersona = personas(i).name;
    
    for j = 1:length(palabras)
        palabra = palabras{j};
        carpeta = fullfile(carpetaBase, nombrePersona, palabra);
        
        archivos = dir(fullfile(carpeta, '*.wav'));
        
        for k = 1:length(archivos)
            archivo = fullfile(carpeta, archivos(k).name);
            [audio, fs] = audioread(archivo);
            
            % Conversión a mono
            if size(audio, 2) > 1
                audio = mean(audio, 2);
            end
            
            % Normalización
            audio = audio / max(abs(audio));
            
            % Remuestreo si es necesario
            if fs ~= fsEsperado
                audio = resample(audio, fsEsperado, fs);
                fs = fsEsperado;
            end

            % Recorte de silencios (opcional)
            % usa Energy thresholding u otros métodos si deseas

            % Extracción de características MFCC
            try
                [coeffs, delta, deltaDelta] = mfcc(audio, fs);
                
                % Promedio y desviación estándar
                feats = [mean(coeffs); std(coeffs); ...
                         mean(delta); std(delta); ...
                         mean(deltaDelta); std(deltaDelta)];
                
                feats = feats(:)'; % Vector fila
                X = [X; feats];
                y = [y; palabra]; % Etiqueta correspondiente

            catch e
                warning('Error en archivo %s: %s', archivo, e.message);
            end
        end
    end
end

% Guardar datos
save('caracteristicas.mat', 'X', 'y');
disp('Extracción completa y guardada en caracteristicas.mat');
