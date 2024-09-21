# Psitest-OMR

Este repositório contém a lógica para treinamento do modelo de reconhecimento óptico de marca (OMR) para o aplicativo Psitest-Imagem.

O modelo é construindo utilizando `keras` e `tensorflow`.

Os dados são utilizados são o [MC Answer Boxes Dataset](https://sites.google.com/view/mcq-dataset).

A biblioteca `keras-tuner` é utilizada para a otimização dos hiperparâmetros do modelo.

A biblioteca `scikit-learn` é utilizada para obter algumas métricas de avaliação do modelo.

## Utilização

Para realizar o treinamento e avaliação do modelo, é necessário primeiro criar um ambiente virtual.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Então, é preciso instalar as dependências do projeto.

```bash
pip install -r requirements.txt
```

Para treinar o modelo, basta seguir o notebook `train.ipynb`.

Neste notebook, os dados serão baixados usando o pacote `gdown`, será realizada a busca pelos melhores hiperparâmetros do modelo utilizando o `keras-tuner` e o modelo será treinado e exportado para a pasta `models/`.
Os dados de teste também serão salvos, na pasta `datasets/`

Após o treinamento do modelo, é possível executar o notebook `predict.ipynb`, que faz uma demo da funcionalidade do modelo, mostrando algumas imagens dos dados de teste e as predições do modelo.

Também é possível avaliar o modelo utilizando o notebook `metrics.ipynb`, que calcula algumas métricas de avaliação do modelo.

## Notas

Existe um boa quantidade de dados para o treinamento do modelo, o que torna o processo relativamente lento e pesado. Por isso, é recomendado o uso de uma GPU para o treinamento do modelo. A GPU será utilizada automaticamente pelo tensorflow, caso esteja disponível. Para instruções de como instalar o tensorflow com suporte a GPU, veja a [documentação oficial](https://www.tensorflow.org/install/pip).
