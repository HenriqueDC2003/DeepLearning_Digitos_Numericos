
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os  


MODEL_NAME = "modelo_mnist_custom.h5"  
NUM_CLASSES = 10 

print(f"Iniciando processo de reconhecimento de dígitos usando o modelo '{MODEL_NAME}'")


print(f"\nVerificando e carregando o modelo: '{MODEL_NAME}'...")
if not os.path.exists(MODEL_NAME):
    print(f"ERRO: O arquivo do modelo '{MODEL_NAME}' não foi encontrado.")
    print(
        "Por favor, execute 'treino_mnist.py' primeiro para treinar e salvar o modelo."
    )
    exit()

try:
    model = keras.models.load_model(MODEL_NAME)
    print(f"Modelo '{MODEL_NAME}' carregado com sucesso!")
except Exception as e:
    print(f"ERRO: Não foi possível carregar o modelo '{MODEL_NAME}'. Detalhes: {e}")
    print(
        "Verifique se o arquivo está íntegro e se foi salvo no formato H5 (não SavedModel)."
    )
    exit()

model.summary()

print("\nCarregando conjunto de dados de teste MNIST...")
try:
    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_test = x_test.astype("float32") / 255.0  
    print(f"{len(x_test)} imagens de teste carregadas e pré-processadas.")
except Exception as e:
    print(f"ERRO: Não foi possível carregar os dados de teste MNIST. Detalhes: {e}")
    print("Verifique sua conexão com a internet ou a instalação do TensorFlow.")
    exit()

index_to_predict = np.random.randint(0, len(x_test))

print(f"\nSelecionando imagem de teste do MNIST com índice: {index_to_predict}")
img_to_predict = x_test[index_to_predict]
true_label = y_test[index_to_predict]

input_for_model = np.expand_dims(img_to_predict, axis=0)

print("\nRealizando predição com o modelo...")
predictions = model.predict(input_for_model)
predicted_label = np.argmax(predictions[0])
confidence = predictions[0][predicted_label] * 100  

plt.figure(figsize=(6, 6))
plt.imshow(img_to_predict, cmap="gray")
plt.title(
    f"Rótulo Real: {true_label}\nPredição: {predicted_label} (Confiança: {confidence:.2f}%)",
    fontsize=14,
    color="green" if predicted_label == true_label else "red",
)
plt.axis("off")
plt.show()

print("\n--- Resultado da Predição ---")
print(f"Rótulo REAL da imagem: {true_label}")
print(f"Dígito PREVISTO pelo modelo: {predicted_label}")
print(f"Confiança da predição: {confidence:.2f}%")

if predicted_label == true_label:
    print("Parabéns! O modelo previu corretamente!")
else:
    print("Ops! O modelo cometeu um erro na predição.")

print("\nProbabilidades detalhadas para cada dígito:")
for i in range(NUM_CLASSES):
    print(f"Dígito {i}: {predictions[0][i]*100:.2f}%")

print("\nProcesso de reconhecimento finalizado.")

