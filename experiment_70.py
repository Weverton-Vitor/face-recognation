"""
Inteligência Artificial aplicada à Visão Computacional
Capítulo 7: Visão Computacional aplicada ao rastreamento de objetos

Todos os direitos reservados à Facti, 2024

www.qualifacti.com.br

É importante esclarecer que estas atividades não compõem a avaliação e não haverá correção formal por parte dos instrutores;
o objetivo é a autoaprendizagem e prática.

-----------------------------------------------------------------------------------------------------------------------
ETAPA CONCEITUALIZAR

Recado importante
Olá,
Como parte do capítulo de rastreamento de objetos, gostaríamos de ressaltar a importância de realizar as atividades de
implementação fornecidas. Estas atividades são cuidadosamente desenhadas para reforçar o conteúdo apresentado. Lembre-se,
a implementação é uma habilidade que se aprimora com a prática. Ao aplicar os conceitos aprendidos, especialmente por meio
da escrita e execução de códigos, você ganhará uma compreensão mais profunda e prática dos modelos. Encorajamos todos a
dedicar tempo a essas atividades. Ao fazer isso, você não apenas reforçará o que foi ensinado, mas também desenvolverá
as habilidades essenciais de resolução de problemas e depuração de código.

Lembrem-se: não basta apenas aprender, é preciso codificar! O caminho para dominar os modelos começa com a experiência prática.

Atenciosamente,
Júlio e Marcelo

-----------------------------------------------------------------------------------------------------------------------
ORIENTAÇÕES:

#1 - Antes de iniciar e executar o código, abra a aba Terminal, localizada na parte inferior do PyCharm e execute, na
sequência, os seguintes comandos para instalar os recursos da biblioteca do OpenCV:

pip install opencv-python

pip install opencv-contrib-python

#2 - Para executar o código:
    * Clique em Run;
    * Ao iniciar a janela do vídeo, selecione com o mouse criando um circulo no objeto de interesse para o rastreamento;
    * Pressione ESC para encerrar a qualquer momento.

-----------------------------------------------------------------------------------------------------------------------

Atividade de experimentação 70

"""

# Importando bibliotecas
import cv2
import numpy as np

# Inicializa a captura pela webcam
cap = cv2.VideoCapture(0)

# Leitura do primeiro frame e transformação em esacala de cinza.
ret, frame = cap.read()
frame_gray_init = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Utilização do método de estimativa de fluxo óptico
parameters_lucas_kanade = dict(winSize=(15, 15),
                               maxLevel=4,
                               criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Define uma função para selecionar um ponto de interesse no frame com um clique do mouse
def select_point(event, x, y, flags, params):
    global point, selected_point, old_points
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)
        selected_point = True # Indica que um ponto foi selecionado
        old_points = np.array([[x, y]], dtype=np.float32) # Armazena o ponto como array NumPy
# Cria uma janela chamada 'Frame' e define a função 'select_point' como callback para eventos do mouse
cv2.namedWindow('Frame')
cv2.namedWindow('Frame')
cv2.setMouseCallback('Frame', select_point)

# Inicializa variáveis para controle do ponto selecionado
selected_point = False
point = ()
old_points = np.array([[]])

# Cria uma máscara com as mesmas dimensões e tipo do frame para desenhar o rastreamento
mask = np.zeros_like(frame)

# Loop principal para processamento de cada frame capturado pela webcam
while True:
    ret, frame = cap.read() # Lê o próximo frame
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Converte o frame atual para escala de cinza
    # Se um ponto foi selecionado
    if selected_point is True:
        cv2.circle(frame, point, 5, (0, 0, 255), 2) # Desenha um círculo no frame no ponto selecionado
        # Calcula o fluxo óptico de Lucas-Kanade para o frame atual
        new_points, status, errors = cv2.calcOpticalFlowPyrLK(frame_gray_init,
                                                              frame_gray,
                                                              old_points,
                                                              None,
                                                              **parameters_lucas_kanade)
        # Atualiza o frame inicial para o frame atual
        frame_gray_init = frame_gray.copy()
        old_points = new_points # Atualiza os pontos antigos para os novos pontos calculados
        # Extrai as coordenadas dos pontos
        x, y = new_points.ravel().astype(int)
        j, k = old_points.ravel().astype(int)

        mask = cv2.line(mask, (x, y), (j, k), (0, 255, 255), 2) # Desenho de uma linha indicando o rastreamento;
        frame = cv2.circle(frame, (x, y), 5, (0, 255, 0), -1) # Demarcação de um ponto do objeto de rastreamento;
    # Combina o frame e a máscara
    img = cv2.add(frame, mask)
    # Exibe o frame e a máscara em janelas separadas
    cv2.imshow("Frame", frame)
    cv2.imshow("Frame 2", mask)
    # Aguarda por uma tecla ser pressionada; se a tecla ESC for pressionada, interrompe o loop
    key = cv2.waitKey(1)
    if key == 27:
        break
# Libera a captura de vídeo e fecha todas as janelas
cap.release()
cv2.destroyAllWindows()