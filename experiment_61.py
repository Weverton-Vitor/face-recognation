"""

Inteligência Artificial aplicada à Visão Computacional
Capítulo 4: Visão Computacional aplicada ao reconhecimento facial em imagens

Todos os direitos reservados à Facti, 2024

www.qualifacti.com.br

-----------------------------------------------------------------------------------------------------------------------

ETAPA CONCEITUALIZAR

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

Atividade de experimentação 61
Escreva um programa, em Python, que utilize classificadores haarcascade para detectar carros em imagens.


ORIENTAÇÕES:

#1 - Antes de iniciar e executar o código, abra a aba Terminal, localizada na parte inferior do PyCharm e execute, na
sequência, os seguintes comandos para instalar os recursos da biblioteca do OpenCV:

pip install opencv-python

pip install opencv-contrib-python

#2 - Lembre-se de trazer a pasta fotos disponibilizada para dentro do PyCharm. Você pode arrastar a pasta para dentro do
projeto, no menu lateral esquerdo.

#3 - Lembre-se de trazer a pasta cascades disponibilizada para dentro do PyCharm. Você pode arrastar a pasta para dentro
do projeto, no menu lateral esquerdo.

-----------------------------------------------------------------------------------------------------------------------
"""

import cv2

classificadorCarro = cv2.CascadeClassifier('cascades\\carros.xml') # utiliza um haarcascade treinado para detectar carros

imagem = cv2.imread('outros\\carros1.jpg') # 1.01; # 9; #70,70 - atente-se para o caminho e extensão da imagem
#imagem = cv2.imread('outros\\carros2.jpg') # 1.053; #9
#imagem = cv2.imread('outros\\carros3.jpg') # 1.02; # 8
#imagem = cv2.imread('outros\\carros4.jpg') # 1.01; # 8
#imagem = cv2.imread('outros\\carros5.jpg')  # 1.01; # 9; #70,70

imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) # transforma a imagem colorida em escala de cinza
# comando para detectar carros na imagem em escala de cinza. Você pode alterar os parâmetros scaleFactor e minNeighbors
# para melhorar a precisão da detecção (alguns estão indicados como os números na frente da imagem acima)

detectado = classificadorCarro.detectMultiScale(imagemCinza, scaleFactor=1.01, minNeighbors=9, minSize=(70, 70))

for (x, y, l, a) in detectado:
    imagem = cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2) # comando para desenhar um retângulo na presença de carros

cv2.imshow(str(len(detectado)) + ' carro(s) encontrado(s)', imagem) # mostrará a quantidade de carros no título da janela
cv2.waitKey() # comando que aguarda o fechamento das janelas com as imagens