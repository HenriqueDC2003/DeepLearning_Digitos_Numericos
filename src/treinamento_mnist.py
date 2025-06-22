import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import os
import matplotlib.pyplot as plt  


NUM_CLASSES = 10  
INPUT_SHAPE = (28, 28)
EPOCHS = 10 
BATCH_SIZE = 64  
VALIDATION_SPLIT_RATIO = 0.1  
MODEL_SAVE_DIR = "models"
MODEL_FILENAME = "modelo_mnist_custom.h5"
MODEL_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_FILENAME)

print(f"Iniciando treinamento do modelo para reconhecimento de dígitos MNIST...")
print(
    f"Configurações: Épocas={EPOCHS}, Batch Size={BATCH_SIZE}, Val. Split={VALIDATION_SPLIT_RATIO*100}%"
)


print("\nCarregando dataset MNIST...")
try:
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    print(
        f"Dados carregados: {len(x_train)} amostras de treino, {len(x_test)} amostras de teste."
    )
except Exception as e:
    print(f"Erro ao carregar dados MNIST: {e}. Verifique sua conexão com a internet.")
    exit()


x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.to_categorical(
    y_test, NUM_CLASSES
)  

print("\nExemplos de imagens de treino")
plt.figure(figsize=(12, 4))  
num_examples_to_show = 5  


random_indices = np.random.choice(len(x_train), num_examples_to_show, replace=False)


for i, idx in enumerate(random_indices):  
    plt.subplot(1, num_examples_to_show, i + 1)
    plt.imshow(x_train[idx], cmap="gray")  
    plt.title(f"Label: {np.argmax(y_train[idx])}") 
    plt.axis("off")
plt.tight_layout()
plt.show()


print("\nConstruindo a arquitetura do modelo...")
model = keras.Sequential(
    [
        
        keras.layers.Flatten(input_shape=INPUT_SHAPE, name="input_flatten_layer"),
        keras.layers.Dense(256, activation="relu", name="hidden_layer_1"),
        keras.layers.Dropout(0.2, name="dropout_layer_1"),
        keras.layers.Dense(128, activation="relu", name="hidden_layer_2"),
        keras.layers.Dropout(0.2, name="dropout_layer_2"),
        keras.layers.Dense(NUM_CLASSES, activation="softmax", name="output_layer"),
    ],
    name="mnist_digit_classifier_model",
)

model.summary()


print("\nCompilando o modelo...")
model.compile(
    optimizer="adam",  
    loss="categorical_crossentropy", 
    metrics=["accuracy"],
)  

early_stopping = EarlyStopping(
    monitor="val_accuracy", patience=5, verbose=1, restore_best_weights=True
)


model_checkpoint = ModelCheckpoint(
    filepath=MODEL_PATH,  
    monitor="val_accuracy",
    save_best_only=True,  
    verbose=1,
)

callbacks_list = [early_stopping, model_checkpoint]


print("\nIniciando o treinamento (pode levar alguns minutos)...")
history = model.fit(
    x_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT_RATIO,
    callbacks=callbacks_list,
    verbose=1,
) 

print("\nTreinamento concluído!")


print(f"\nAvaliando o modelo no conjunto de teste ({len(x_test)} amostras)...")
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Acurácia final no conjunto de teste: {accuracy*100:.2f}%")
print(f"Perda final no conjunto de teste: {loss:.4f}")
