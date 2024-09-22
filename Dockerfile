FROM quay.io/jupyter/tensorflow-notebook:cuda-python-3.11.10

# Crie um diretório de trabalho no contêiner
WORKDIR /app

# Copie o arquivo de requisitos para o contêiner
COPY requirements.txt /app

# Instale as dependências
RUN pip install --no-cache-dir -r requirements.txt
