# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""
from flask import Flask, render_template, Response, jsonify, request
from camera import VideoCamera
import cv2
from PIL import ImageGrab, Image
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
import webbrowser
import random
################
from IPython.display import display
from yolo import YOLO


#####################
emotion_message=" "
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT']=1
video_stream = VideoCamera()
@app.route('/')
def index():
    return render_template('index.html')
def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
@app.route('/video_feed')
def video_feed():
    return Response(gen(video_stream),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/after', methods=['Get','POST'])
def after():

    
    def objectDetection(file, model_path, class_path, anchors_path):
        yolo = YOLO(model_path=model_path, classes_path=class_path, anchors_path=anchors_path)
        image = Image.open(file)
        result_image = yolo.detect_image(image)
        x,y,w,h=yolo.detect_point(image)
        return result_image, x, y, w, h

    img = ImageGrab.grab()
    imgCrop = img.crop((600,300,1200,700))
    imgCrop.save('static/file.png')
##########################################
    img1, x, y, w, h= objectDetection('static/file.png', 'model_data/yolo.h5', 'model_data/coco_classes.txt', 'model_data/yolo_anchors.txt')    
    img1.save('static/file.png')
##########################################
    

    #cropped=img1[y:y+h,x:x+w]
    a=y+h
    b=x+w
    cropped = img1.crop((x,a, b,y))
    cropped.save('static/cropped.png')
    #image=cv2.imwrite('static/cropped.png', cropped)
    image=cv2.imread('static/cropped.png',0)
    
    image=cv2.resize(image, (64,64))

    image=image/255.0

    image=np.reshape(image,(1,64,64,1))

    model= load_model('emotion.hdf5')
    model1=load_model('age.h5')

    prediction=model.predict(image)

    label_map =["화남" ,"언짢음","무서움", "행복함", "슬픔", "놀람", "무표정"]
    label = ["화가 많이 나셨나봐요.. 화를 풀 수 있는 노래를 추천해 드릴게요.", "언짢으신 일이 있으신가요? 기분 풀 수 있는 노래를 추천해 드릴게요.", "많이 놀라셨나요? 진정할 수 있게 차분한 노래를 추천해 드릴게요.", "행복해보이시네요! 즐거운 노래를 계속 들어보세요!" ,"슬퍼보이시네요. 위로가 될 수 있는 노래를 추천해드릴게요.", "많이 놀라셨나요? 진정할 수 있게 차분한 노래를 추천해 드릴게요." ,"특별한 감정이 판단되지 않았어요. 멜론차트를 들려드릴게요!"]
    angry=["잠시 모든 걸 멈추고 숨을 고른다","화낼 만한 일인지 먼저 생각한다","감정과 반응 사이에 공간을 만드세요", "왜 화가 나는지 스스로에게 묻는다", "1분동안 무언가에 집중해보세요"]
    sad=["꼭 기억하고 약속해 줘. 넌 네가 믿는것보다 더 용감하며, 보기보다 강하고, 네 생각보다 더 똑똑하다는 것을","낙오자라는 세 글자에 슬퍼하지 말고, 사랑이란 두 글자에 얽매이지 말고, 삶이란 한 글자에 충실 하라","당신이 잘 보여야 하는 사람 말고, 당신을 행복하게 만들어주는 사람과 인생을 함께 하세요. ","모두들 괴롭고 외롭고 슬프다 그냥 아닌척 살아갈 뿐이다. 쓰러져 울기보다 밝은 척 살아가는 것이 더 나은 일이란 것을 알기에","저녁에 돌아갈 집이 있다는 것, 힘들 떄 마음속으로 생각할 사람이 있다는것, 외로울 때 혼자 부를 노래가 있다는 것"]
    age_ranges = ['1-2', '3-9', '10-20', '28-45', '21-27', '46-65', '66-116']
    prediction=np.argmax(prediction)
    face_age = age_ranges[np.argmax(model1.predict(image))] 
    final_prediction=label_map[prediction]

    resultMessage = label[prediction]
    if final_prediction=="화남":
        link  = f"https://www.youtube.com/results?search_query=차분해지는 노래"
        emotion_message=random.choice(angry)
    elif final_prediction=="언짢음":
        link  = f"https://www.youtube.com/results?search_query=언짢을때 듣는 노래"
        emotion_message=" "
    elif final_prediction=="무서움":
        link  = f"https://www.youtube.com/results?search_query=안정되는 노래"
        emotion_message=" "
    elif final_prediction=="행복함":
        link  = f"https://www.youtube.com/results?search_query=행복할때 듣는 노래"
        emotion_message=" "
    elif final_prediction=="슬픔":
        link  = f"https://www.youtube.com/results?search_query=위로해주는 노래"
        emotion_message=random.choice(sad)
    elif final_prediction=="놀람":
        link  = f"https://www.youtube.com/results?search_query=진정되는 노래"
        emotion_message=" "
        
    elif final_prediction=="무표정":
        if face_age=="1-2":
             link  = f"https://www.youtube.com/results?search_query=동요"
        elif face_age=="3-9":
            link = f"https://www.youtube.com/results?search_query=동요"
        elif face_age=="10-20":
            link  = f"https://www.youtube.com/results?search_query=최신 멜론 탑 100 차트"
        elif face_age=="21-27":
            link  = f"https://www.youtube.com/results?search_query=최신 멜론 탑 100 차트"
        elif face_age=="28-45":
            link  = f"https://www.youtube.com/results?search_query=2000년대 가요"
        elif face_age=="46-65":
            link  = f"https://www.youtube.com/results?search_query=1990년대 가요"
        elif face_age=="66-116":
            link  = f"https://www.youtube.com/results?search_query=1980년대 가요"
        emotion_message=" "

        
    return render_template('after.html', data=final_prediction,message=resultMessage, link=link, Phrases=emotion_message,age=face_age)


if __name__ == '__main__':
	app.run(host='127.0.0.1', debug=True,port="5000")