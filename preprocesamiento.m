clc; clear;

% Carpeta base donde están los audios
carpetaBase = 'Audios';  % Cambia a tu ruta si es distinta

% Palabras objetivo
palabras = {'Casa', 'Lluvia', 'Nube', 'Perro', 'Tren'};

% Inicializa contenedores
X = [];         % Matriz de características
y = {};         % Etiquetas de palabras
locutores = {}; % Etiquetas de locutores

% Frecuencia de muestreo esperada
fsEsperado = 16000;

% Obtener lista de personas
personas = dir(carpetaBase);
personas = personas([personas.isdir] & ~startsWith({personas.name}, '.'));

% Recorre cada persona y palabra
for i = 1:length(personas)
    nombrePersona = personas(i).name;
    
    for j = 1:length(palabras)
        palabra = palabras{j};
        carpeta = fullfile(carpetaBase, nombrePersona, palabra);
        
        if ~isfolder(carpeta)
            continue;  % Si la carpeta no existe, saltar
        end
        
        archivos = dir(fullfile(carpeta, '*.wav'));
        
        for k = 1:length(archivos)
            archivo = fullfile(carpeta, archivos(k).name);
            
            try
                % Leer audio
                [audio, fs] = audioread(archivo);

                % Convertir a mono si es estéreo
                if size(audio, 2) > 1
                    audio = mean(audio, 2);
                end

                % Normalizar amplitud
                audio = audio / max(abs(audio) + eps);

                % Remuestrear si es necesario
                if fs ~= fsEsperado
                    audio = resample(audio, fsEsperado, fs);
                    fs = fsEsperado;
                end

                % Extraer MFCC + delta + delta-delta
                [coeffs, delta, deltaDelta] = mfcc(audio, fs);

                % Calcular estadísticas (media + std)
                feats = [mean(coeffs); std(coeffs); ...
                         mean(delta); std(delta); ...
                         mean(deltaDelta); std(deltaDelta)];
                feats = feats(:)';  % Vector fila

                % Almacenar datos
                X = [X; feats];
                y = [y; palabra];
                locutores = [locutores; nombrePersona];

            catch e
                warning('Error procesando %s: %s', archivo, e.message);
            end
        end
    end
end

% Guardar características y etiquetas
save('caracteristicas.mat', 'X', 'y', 'locutores');
disp('✅ Extracción completa guardada en caracteristicas.mat');
