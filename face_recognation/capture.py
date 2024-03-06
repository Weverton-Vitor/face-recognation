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
Algoritmo de captura das imagens


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
import numpy as np

classificador = cv2.CascadeClassifier("../cascades/haarcascade_frontalface_default.xml") # utilizando o haarcascade para detectar a face
classificadorOlho = cv2.CascadeClassifier("../cascades/haarcascade_eye.xml") # utilizando o haarcascade para detectar olhos
camera = cv2.VideoCapture(0) # 0 siginifica que irá usar a câmera padrão e 1 webcam externa
amostra = 1
numeroAmostras = 25 # captura de 25 imagens da face, varie a posição ao capturar
id = input('Digite seu identificador: ') # comece com o identificador #1 para realizar captura sem máscara e #2 para com máscara
largura, altura = 200, 200 # dimensões das imagens capturadas
print("Capturando as faces...") # indicação de início da captura

while True:
    conectado, imagem = camera.read() # captura das imagens pela webcam
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) # transformação das imagens em escala de cinza
    facesDetectadas = classificador.detectMultiScale(imagemCinza, scaleFactor=1.5, minSize=(100, 100)) # detecta face nas imagens capturadas

    for (x, y, l, a) in facesDetectadas:
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2) # desenha um retângulo na face detectada
        regiao = imagem[y:y + a, x:x + l] # dimensões do retângulo
        regiaoCinzaOlho = cv2.cvtColor(regiao, cv2.COLOR_BGR2GRAY)
        olhosDetectados = classificadorOlho.detectMultiScale(regiaoCinzaOlho)
        for (ox, oy, ol, oa) in olhosDetectados:
            cv2.rectangle(regiao, (ox, oy), (ox + ol, oy + oa), (0, 255, 0), 2) # desenha um retângulo nos olhos detectados

        if cv2.waitKey(1) & 0xFF == ord('q'): # aperte a tecla Q para realizar as capturas das imagens
            if np.average(imagemCinza) > 100:
                imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, altura))
                # criar a pasta fotos: New > Directory > fotos
                cv2.imwrite("./fotos/pessoa." + str(id) + "." + str(amostra) + ".jpg", imagemFace) # nomeia as imagens capturadas
                print("[foto " + str(amostra) + " capturada com sucesso]")
                amostra += 1

    cv2.imshow("Face", imagem)
    cv2.waitKey(1)
    if amostra >= numeroAmostras + 1:
        break

print("Faces capturadas com sucesso") # mensagem de finalização da captura das 25 imagens
camera.release()
cv2.destroyAllWindows()