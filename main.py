# Imports
import cv2
import os # lib de criação de pastas
import numpy as np

# Função savePerson()
def savePerson():
    global identificacao
    global boolsaveimg
    print('Qual o seu nome?')
    name = input()
    identificacao = name
    boolsaveimg = True

# Função diretorio()
def diretorio(img):
    global identificacao
    if not os.path.exists('train'):
        os.makedirs('train')
    if not os.path.exists(f'train/{identificacao}'):
        os.makedirs(f'train/{identificacao}')
    # Criar arquivos na pasta
    files = os.listdir(f'train/{identificacao}')
    cv2.imwrite(f'train/{identificacao}/{str(len(files))}.jpg', img)

# Função trainData() - Percerre todas as pastas/pessoas do diretorio para treinar o modelo
def trainData():
    global recognizer
    global trained
    trained = True
    persons = os.listdir('train')
    ids = []
    faces = []
    for i, p in enumerate(persons):
        for f in os.listdir(f'train/{p}'):
         img = cv2.imread(f'train/{p}/{f}', 0)
         faces.append(img)
         ids.append(i)
    recognizer.train(faces, np.array(ids))


# Variáveis
identificacao = ''
boolsaveimg = False
saveCount = 0
trained = False
persons = os.listdir('train')

# Recognizer - Reconhecedor
# Necessita instalar o LBPH - pip install opencv_contrib_python
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Ler a webcam
cap = cv2.VideoCapture(0)

# Carregar o xml do Haar Cascade - classificador
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Loop
while(True):

    # Webcam para frame
    ret, frame = cap.read()

    # Frame em escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecção da face no frame 'gray'
    faces = face_cascade.detectMultiScale(gray, 1.3,5)

    # Iterar todas as faces no frame
    for (x,y,w,h) in faces:

        # Cortar face - Roi
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (50,50))

        # Colocar o retângulo na face do frame 'frame'
        cv2.rectangle(frame, (x,y), (x+w, y+h), (200,0,0), 3)

        # Teste
        if trained:
            idp, acc = recognizer.predict(roi)
            namePerson = persons[idp]
            cv2.putText(frame, namePerson, (x,y), 3,2, (0,255,0), 2, cv2.LINE_AA)

        # Checa boolsaveimg
        if boolsaveimg:
            diretorio(roi)
            saveCount += 1

        # Contados Stop save img
        if saveCount > 50:
            boolsaveimg = False
            saveCount = 0
    
    # Exibir o frame
    cv2.imshow('frame', frame)

    # Recupera botão pressionado
    key = cv2.waitKey(1)

    # Close loop
    if key == ord('c'):
        break

    # Salvar imagens
    if key == ord('s'):
        savePerson()

    # Treinamento do modelo
    if key == ord('t'):
        trainData()

# fechamento da janela e reinicialização da mesma
cap.release()
cv2.destroyAllWindows()
