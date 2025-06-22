import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import os
import matplotlib.pyplot as plt  # Adicionado para visualização de dados

# --- Parâmetros de Configuração ---
# Definindo parâmetros em um só lugar para facilitar experimentação
NUM_CLASSES = 10  # Dígitos de 0 a 9
INPUT_SHAPE = (28, 28)
EPOCHS = 10  # Aumentado para demonstrar EarlyStopping
BATCH_SIZE = 64  # Adicionado para controle de treinamento
VALIDATION_SPLIT_RATIO = 0.1  # 10% dos dados de treino para validação
MODEL_SAVE_DIR = "models"
MODEL_FILENAME = "modelo_mnist_custom.h5"
MODEL_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_FILENAME)

print(f"Iniciando treinamento do modelo para reconhecimento de dígitos MNIST...")
print(
    f"Configurações: Épocas={EPOCHS}, Batch Size={BATCH_SIZE}, Val. Split={VALIDATION_SPLIT_RATIO*100}%"
)

# --- 1. Carregamento e Pré-processamento dos Dados ---
# Carregando o famoso dataset MNIST de dígitos escritos à mão
print("\nCarregando dataset MNIST...")
try:
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    print(
        f"Dados carregados: {len(x_train)} amostras de treino, {len(x_test)} amostras de teste."
    )
except Exception as e:
    print(f"Erro ao carregar dados MNIST: {e}. Verifique sua conexão com a internet.")
    exit()

# Normalizando as imagens para o intervalo [0, 1]
# Isso ajuda o modelo a aprender melhor e mais rápido
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# As imagens estão em escala de cinza e 2D.
# Se estivéssemos usando convoluções, talvez adicionássemos uma dimensão de canal (28, 28, 1)
# Mas para camadas Dense, Flatten é suficiente.

# Convertendo os rótulos para o formato one-hot encoding
# Ex: o dígito 5 vira [0,0,0,0,0,1,0,0,0,0]
y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.to_categorical(
    y_test, NUM_CLASSES
)  # Não será usado no treinamento, mas é bom pré-processar

# --- 2. Visualização de Dados (Para demonstrar entendimento) ---
# Vamos mostrar algumas amostras do dataset para entender com o que estamos trabalhando
print("\nExemplos de imagens de treino")
plt.figure(figsize=(12, 4))  # Aumentei um pouco o tamanho para melhor visualização
num_examples_to_show = 5  # Quantidade de exemplos a serem mostrados

# Gera uma lista de índices aleatórios para as imagens de treino
random_indices = np.random.choice(len(x_train), num_examples_to_show, replace=False)
# `replace=False` garante que não escolheremos a mesma imagem duas vezes

for i, idx in enumerate(random_indices):  # Iteramos sobre os índices aleatórios
    plt.subplot(1, num_examples_to_show, i + 1)
    plt.imshow(x_train[idx], cmap="gray")  # Usamos o índice aleatório (idx)
    plt.title(f"Label: {np.argmax(y_train[idx])}")  # E o rótulo correspondente
    plt.axis("off")
plt.tight_layout()
plt.show()

# --- 3. Construção da Arquitetura da Rede Neural ---
# Esta é a parte central, onde definimos como o modelo vai "pensar"
print("\nConstruindo a arquitetura do modelo...")
model = keras.Sequential(
    [
        # Camada de Flatten: "Achata" a imagem 28x28 em um vetor de 784 pixels.
        # Essencial para conectar com as camadas Dense.
        keras.layers.Flatten(input_shape=INPUT_SHAPE, name="input_flatten_layer"),
        # Primeira camada densa (fully connected): 256 neurônios com ativação ReLU.
        # Experimentei com 128 e 64, mas 256 oferece mais capacidade sem ser excessivo.
        keras.layers.Dense(256, activation="relu", name="hidden_layer_1"),
        # Camada de Dropout para regularização: Ajuda a prevenir overfitting.
        # Desativa aleatoriamente 20% dos neurônios durante o treinamento.
        keras.layers.Dropout(0.2, name="dropout_layer_1"),
        # Segunda camada densa: Reduzimos para 128 neurônios.
        keras.layers.Dense(128, activation="relu", name="hidden_layer_2"),
        # Mais um dropout para robustez
        keras.layers.Dropout(0.2, name="dropout_layer_2"),
        # Camada de saída: 10 neurônios (um para cada dígito).
        # 'softmax' é ideal para classificação multi-classe, pois as saídas podem ser interpretadas como probabilidades.
        keras.layers.Dense(NUM_CLASSES, activation="softmax", name="output_layer"),
    ],
    name="mnist_digit_classifier_model",
)

# Resumo do modelo - muito útil para visualizar a estrutura
model.summary()

# --- 4. Compilação do Modelo ---
# Definindo o otimizador, a função de perda e as métricas para avaliação
print("\nCompilando o modelo...")
model.compile(
    optimizer="adam",  # Otimizador Adam é um bom ponto de partida, adaptável
    loss="categorical_crossentropy",  # Perda ideal para classificação multi-classe one-hot
    metrics=["accuracy"],
)  # Precisamos acompanhar a acurácia!

# --- 5. Callbacks para Treinamento Inteligente ---
# Adicionando EarlyStopping e ModelCheckpoint para melhorar o processo de treinamento.
# EarlyStopping: Para de treinar se a acurácia de validação não melhorar após algumas épocas.
# Isso evita o overfitting e economiza tempo.
early_stopping = EarlyStopping(
    monitor="val_accuracy", patience=5, verbose=1, restore_best_weights=True
)

# ModelCheckpoint: Salva o modelo com a melhor acurácia de validação durante o treinamento.
# Garante que temos o melhor modelo, mesmo que o EarlyStopping pare o treinamento.
model_checkpoint = ModelCheckpoint(
    filepath=MODEL_PATH,  # Salva com o nome definido
    monitor="val_accuracy",
    save_best_only=True,  # Salva apenas quando há melhora
    verbose=1,
)

callbacks_list = [early_stopping, model_checkpoint]

# --- 6. Treinamento do Modelo ---
print("\nIniciando o treinamento (pode levar alguns minutos)...")
history = model.fit(
    x_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT_RATIO,
    callbacks=callbacks_list,
    verbose=1,
)  # verbose=1 para ver o progresso

print("\nTreinamento concluído!")

# --- 7. Avaliação Final do Modelo (opcional, mas bom para relatório) ---
# Avaliando o modelo no conjunto de teste para ter uma métrica final
print(f"\nAvaliando o modelo no conjunto de teste ({len(x_test)} amostras)...")
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Acurácia final no conjunto de teste: {accuracy*100:.2f}%")
print(f"Perda final no conjunto de teste: {loss:.4f}")
