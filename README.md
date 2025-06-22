# Reconhecimento de Dígitos Numéricos com Keras e MNIST

Este projeto implementa um modelo de Rede Neural Convolucional (CNN) utilizando a biblioteca Keras (com TensorFlow como backend) para o reconhecimento de dígitos manuscritos (0 a 9) do famoso conjunto de dados MNIST. O objetivo é demonstrar o ciclo completo de Deep Learning: pré-processamento de dados, construção e treinamento de um modelo, e sua posterior utilização para predição.

---

# Funcionalidades

* **Treinamento do Modelo:** Script `treino_mnist.py` para construir, compilar e treinar um modelo de rede neural, com visualização dinâmica de amostras de treino e gráficos de progresso.
* **Salvamento Inteligente do Modelo:** Utiliza `ModelCheckpoint` para salvar automaticamente a melhor versão do modelo durante o treinamento, e `EarlyStopping` para otimizar o tempo de treino e prevenir overfitting.
* **Reconhecimento de Dígitos:** Script `prever_mnist.py` para carregar o modelo treinado e realizar predições em novas imagens, exibindo o resultado visualmente e detalhando as probabilidades.
* **Estrutura Organizada:** Código-fonte, modelo treinado e exemplos de imagens organizados em diretórios claros.

---

## Pré-requisitos

Para executar este projeto localmente, você precisará ter o Python 3 instalado. Recomenda-se o uso de um ambiente virtual (`.venv`) para gerenciar as dependências.

* **Python 3.11**
* **pip** (gerenciador de pacotes do Python)

---

## Como Configurar e Executar

Siga os passos abaixo para configurar o ambiente e rodar o projeto.

### 1. Clonar o Repositório

Primeiro, clone este repositório para o seu ambiente local:

git clone (https://github.com/SeuUsuario/SeuRepositorio.git)](https://github.com/HenriqueDC2003/DeepLearning_Digitos_Numericos-.git)

cd SeuRepositorio

### Criar o ambiente virtual
python -m venv .venv

### Ativar o ambiente virtual (Windows)
.\.venv\Scripts\activate

### Ativar o ambiente virtual (macOS/Linux)
source ./.venv/bin/activate

### Instalar as Dependências
pip install -r requirements.txt

### Treinamento do Modelo
O modelo já foi treinado e o arquivo modelo_mnist_custom.h5 está incluído na pasta models/. No entanto, se você quiser re-treinar o modelo ou verificar o processo de treinamento:

### Certifique-se de que o ambiente virtual está ativado.

### Navegue até o diretório src/:
cd src

### Execute o script de treinamento:
python treino_mnist.py

### Este script irá:
Baixar o dataset MNIST (se não estiver disponível localmente).
Pré-processar os dados.
Construir e treinar o modelo de rede neural.
Exibir 5 imagens aleatórias de treino com seus rótulos.
Plotar gráficos de acurácia e perda durante o treinamento.
Salvar a melhor versão do modelo treinado como modelo_mnist_custom.h5 dentro da pasta ../models/.


### Reconhecimento (Predição) de Dígitos

### Execute o script de predição no diretório src/:
python prever_mnist.py

### Este script irá:
Carregar o modelo modelo_mnist_custom.h5 da pasta ../models/.
Selecionar aleatoriamente uma imagem do conjunto de testes do MNIST.
Exibir a imagem e a predição do modelo (com confiança), comparando-a com o rótulo real.
Mostrar as probabilidades detalhadas para cada dígito (0-9).
