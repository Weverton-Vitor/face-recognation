"""

Inteligência Artificial aplicada à Visão Computacional
Capítulo 4: Visão Computacional aplicada ao reconhecimento facial em imagens

Todos os direitos reservados à Facti, 2024

www.qualifacti.com.br

-----------------------------------------------------------------------------------------------------------------------

ETAPA CONSOLIDAR

Recado importante
Olá,
Como parte do capítulo de reconhecimento facial, gostaríamos de ressaltar a importância de realizar as atividades de
implementação fornecidas. Estas atividades são cuidadosamente desenhadas para reforçar o conteúdo apresentado. Lembre-se,
a implementação é uma habilidade que se aprimora com a prática. Ao aplicar os conceitos aprendidos, especialmente por meio
da escrita e execução de códigos, você ganhará uma compreensão mais profunda e prática dos modelos. Encorajamos todos a
dedicar tempo a essas atividades. Ao fazer isso, você não apenas reforçará o que foi ensinado, mas também desenvolverá
as habilidades essenciais de resolução de problemas e depuração de código.

Lembrem-se: não basta apenas aprender, é preciso codificar! O caminho para dominar os modelos começa com a experiência prática.

Atenciosamente,
Júlio e Marcelo

-----------------------------------------------------------------------------------------------------------------------

Estudo de Caso
Algoritmo de treinamento das imagens


ORIENTAÇÕES:

#1 - Antes de iniciar e executar o código, abra a aba Terminal, localizada na parte inferior do PyCharm e execute, na
sequência, os seguintes comandos para instalar os recursos da biblioteca do OpenCV:

pip install opencv-python

pip install opencv-contrib-python

#2 - Lembre-se de trazer a pasta cascades disponibilizada para dentro do PyCharm. Você pode arrastar a pasta para dentro
do projeto, no menu lateral esquerdo.

-----------------------------------------------------------------------------------------------------------------------
"""

# Importando as bibliotecas
import cv2
import os
import numpy as np

eigenface = cv2.face.EigenFaceRecognizer_create() # traz a função de reconhecimento Eigenface
fisherface = cv2.face.FisherFaceRecognizer_create() # traz a função de reconhecimento Fisherface
lbph = cv2.face.LBPHFaceRecognizer_create() # traz a função de reconhecimento LBPH

def getImagemComId():
    caminhos = [os.path.join('fotos', f) for f in os.listdir('fotos')] # irá percorrer todas as imagens da pasta fotos criada na captura
    faces = []
    ids = []
    for caminhoImagem in caminhos:
        imagemFace = cv2.cvtColor(cv2.imread(caminhoImagem), cv2.COLOR_BGR2GRAY) # transforma as imagens em escala de cinza
        id = int(os.path.split(caminhoImagem)[-1].split('.')[1]) # verifica qual o id do identificador criado na captura
        ids.append(id)
        faces.append(imagemFace)
    return np.array(ids), faces

ids, faces = getImagemComId()

print("Treinando...") # indicação que está havendo o treinamento, conforme o reconhecedor
eigenface.train(faces, ids)
eigenface.write('cascades\\classificadorEigen.yml') # realiza o treinamento e cria o classificador Eingeface

fisherface.train(faces, ids)
fisherface.write('cascades\\classificadorFisher.yml') # realiza o treinamento e cria o classificador Fisherface

lbph.train(faces, ids)
lbph.write('cascades\\classificadorLBPH.yml') # realiza o treinamento e cria o classificador LBPH

print("Treinamento realizado") # indica que o treinamento foi finalizado
#depois de treinado ele vai gerar os arquivos que vão aparecer no menu ao lado e serão utilizados para o algoritmo de reconhecimento