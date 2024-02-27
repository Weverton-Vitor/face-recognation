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

Atividade de experimentação 58
Escreva um programa, em Python, que transforme uma imagem colorida em uma imagem em escala de cinza.


ORIENTAÇÕES:

#1 - Antes de iniciar e executar o código, abra a aba Terminal localizada na parte inferior do PyCharm e execute, na
sequência, os seguintes comandos para instalar os recursos da biblioteca do OpenCV:

pip install opencv-python

pip install opencv-contrib-python

#2 - Lembre-se de trazer a pasta fotos disponibilizada para dentro do PyCharm. Você pode arrastar a pasta para dentro do
projeto, no menu lateral esquerdo.

-----------------------------------------------------------------------------------------------------------------------
"""

import cv2 # realiza o import da função cv2

print(cv2.__version__) # verifica qual a versão instalada

#help(cv2.face) # ajuda sobre a função cv2

imagem = cv2.imread('outros\\carros5.jpg') # comando para leitura de uma imagem, atente-se para o caminho e extensão da imagem
imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) # transforma a imagem colorida em escala de cinza

cv2.imshow("Original", imagem) # comando para abrir a imagem original colorida
cv2.imshow("Cinza", imagemCinza) # comando para abrir a imagem em escala de cinza

cv2.waitKey() # comando que aguarda o fechamento das janelas com as imagens