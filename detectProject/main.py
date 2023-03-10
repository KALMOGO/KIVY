from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.stacklayout import StackLayout
from kivy.uix.image import Image
from kivy.properties import ColorProperty
from kivy.core.window import Window
from kivy.uix.label import Label
from kivy.graphics.texture import Texture
from kivy.logger import Logger
from kivy.clock import Clock
from kivy.input.providers.mouse import MouseMotionEvent

# Recognation labrairies
import numpy as np
import pandas as pd
import cv2 
import os
import pickle

def recognation_fit(x_train:list,label_ids:dict, y_labels:list):
    """
        Recognation fit function on train images 

    Args:
        x_train (list): images list to be recognize
        label_ids (dict): 
        y_labels (_type_): id of images for the name
    """
    # model de reconnaissance faciale
    recognizer = cv2.face.LBPHFaceRecognizer_create()    
    
    with open("labels.pickle", "wb") as f:
        pickle.dump(label_ids, f)

    recognizer.train(x_train, np.array(y_labels))
    recognizer.save("trainner.yml")

def detect_faces(size_percent:int):
    
    """
        Dectect images to recognize on a video
        
    Returns:
        tuple: (Image_list_detected, Images_label_id, Image_label)
    """
    y_labels = []
    x_train = []
    current_id = 0
    label_ids = {}
    face_cascade = cv2.CascadeClassifier("C:/Users/DELL/Desktop/traitementImages/exercices/devoir/src/cascades/data/haarcascade_frontalface_alt2.xml")


    BASE_DIR = os.path.dirname(os.path.abspath('__file__')) # j'obtient le chemin absolue du file 

    image_dir = os.path.join(BASE_DIR, "images") # j'obtient le repertoire absolu du repertoire images

    #os.walk() permet de generer sous la forme d'itérateur le tree du repertoire : (racine, dossier, fichier)
    
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith("png") or file.endswith("jpg"): # str.endswith() & str.startswith()
                path = os.path.join(root, file) # generer le repertoire absolue de tous les fichiers du repertoire images

                label = os.path.basename(os.path.dirname(path)).replace(" ","-").lower() # repertoire courant de l'image

                if label not in label_ids:
                    label_ids[label] = current_id
                    current_id+=1

                id_ = label_ids[label]
                
                image = cv2.imread(path, 0) # chargement de l'image
                
                # pretraitement de l'imade lue
                image = resize_image(image,size_percent)
                image = img_improver(image)
                
                # reconnaitre  uniquement les visage sur une photo
                faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=3)

                for (x,y,w, h)  in faces:
                    roi = image[y:y+h, x:x+w] # delimitation de l'image sur le visage 
                    x_train.append(roi) # ajout de l'image dans la liste a reconnaitre
                    y_labels.append(id_) # labels des images
                    
    return (x_train,label_ids,y_labels)


def img_improver(img=None):
    
    assert img is not None, "Your image is not good !!!"
    img = cv2.equalizeHist(img)
    img = cv2.bilateralFilter (img,1,75, 100, borderType=cv2.BORDER_CONSTANT)
    return img


def resize_image(img=None, percent=0):
    
    assert img is not None, "ERROR: image invalid !!!"
    assert percent>0, "Error: the percent of reduction should be between 1 and 100 !!!" 

    scale_percent = percent # pourcentage de reduction de l'image : percent%
    #calculate the percent of original dimensions
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)

    # dsize
    dsize = (width, height)

    return cv2.resize(img, dsize)


class SAVEBUTTON(Button):
    def on_touch_down(self, touch):
        
        if touch.is_double_tap:
            BASE_DIR = os.path.dirname(os.path.abspath('__file__'))
            SAVE_PATH = os.path.join(BASE_DIR, "trainImages")

    def imge_save(self):
        pass

        
    def on_touch_up(self, touch):
        self.save_imge = True
        
from functools import partial  
        
class Main(App):
    def __init__(self, **kwargs):
        Window.set_system_cursor('hand')
        self.btn_clicked = None
        self.main_layout = BoxLayout(orientation='vertical')
        self.img_web_cam = None
        self.caputre = None
        self.save_imge = False
        self.computer = 0
        self.frame = None
        self.original_image = None

        self.x_train=None
        self.label_ids=None
        self.y_labels=None

        self.is_caputer = False
        self.is_run_dectection = False
        self.save_image_btn = None
        
        super().__init__(**kwargs)
        

    def build(self):
    
        # image section
        
        
        # capture des photo à reconnaitre
        SL = StackLayout(orientation ='lr-tb', size_hint_y=0.7)
        btn1 = Button(text='capture',font_size = 20,size_hint =(1/3, .3),
                    background_color=(71/255, 104/255, 237/255, 1), on_press=self.train_btn)
        
        btn2 = Button(text='stop',font_size = 20,size_hint =(1/3, .3),
                    background_color=(71/255, 104/255, 237/255, 1), on_press=self.stop_btn)
        
        btn3 = Button(text='train',font_size = 20,size_hint =(1/3, .3),
                    background_color=(71/255, 104/255, 237/255, 1), on_press=self.capture_image)
        
        Start_btn = Button(text='Start recognation',font_size = 20,
            background_color=(71/255, 104/255, 237/255, 1), on_press=self.recognation)
        
        SL.add_widget(btn1)
        SL.add_widget(btn2)
        SL.add_widget(btn3)
        SL.add_widget(Start_btn)

        # Label descriptive
        bar_top = Label(size_hint_y=0.09)
        bar_bottom = Label(text='<--- face detection system --->',size_hint_y=0.2)
        bar_bottom1 = Label(size_hint_y=.25)
        
        self.main_layout.add_widget(bar_top)
        self.main_layout.add_widget(bar_bottom)
        self.main_layout.add_widget(SL)
        
        # Video section 
        Clock.schedule_interval(self.update,1.0/33.0)
        
        return self.main_layout
    
    def imge_save(self, path, capture):
        self.computer += 1
        path = os.path.join(path,"trainImages", f'{self.computer}.png')
        cv2.imwrite(path, self.original_image)


    def capture_image(self, *args):
        self.x_train,self.label_ids,self.y_labels = detect_faces(50) 
        recognation_fit(self.x_train,self.label_ids,self.y_labels)


    def train_btn(self, *args):
        
        self._extracted_from_recognation_3(True, False)
        self.save_image_btn = Button(text='Click to save', size_hint=(1, 0.2), background_color=(0.0,128/255,0.0,1), 
                            font_size=20 )
        BASE_DIR = os.path.dirname(os.path.abspath('__file__'))
        SAVE_PATH = os.path.join(BASE_DIR,"images")

        self.save_image_btn.bind(on_press=partial(self.imge_save, SAVE_PATH))
        self.main_layout.add_widget(self.img_web_cam, len(self.main_layout.children) )
        self.main_layout.add_widget(self.save_image_btn, len(self.main_layout.children) )

        if self.caputre is None:
            self.caputre = cv2.VideoCapture(0)
            

    def stop_btn(self, *args):

        if self.is_caputer == True:
            self.main_layout.remove_widget(self.img_web_cam)
            self.main_layout.remove_widget(self.save_image_btn)
            
        if self.is_run_dectection == True:
            self.main_layout.remove_widget(self.img_web_cam)
            
        self.is_caputer = False
        self.is_run_dectection=False
        self.caputre.release()
        self.caputre = None
        
    def recognation(self, *args):
        self._extracted_from_recognation_3(False, True)
        self.main_layout.add_widget(self.img_web_cam, len(self.main_layout.children) )

        if self.caputre is None:
            self.caputre = cv2.VideoCapture(0)

    
    def _extracted_from_recognation_3(self, arg0, arg1):
        self.is_caputer = arg0
        self.is_run_dectection = arg1
        self.img_web_cam = Image(size_hint_y=2.5)

    def update(self, *args):
        
        if self.caputre is not None:
            if  self.is_caputer == True:
                
                ret, self.frame = self.caputre.read()  
                self.original_image = self.frame.copy()
                
                face_cascade = cv2.CascadeClassifier("C:/Users/DELL/Desktop/traitementImages/exercices/devoir/src/cascades/data/haarcascade_frontalface_alt2.xml")
                gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
                line = 2
                
                for (x,y,h,w) in faces:
                    color = (255,0,0) #BGR
                    self.frame = cv2.rectangle(self.frame, (x,y), ( x+w, y+h), color, line)
                                
                buf = cv2.flip(self.frame, 0).tobytes()
                imge_texture = Texture.create(size=(self.frame.shape[1], self.frame.shape[0]),colorfmt='bgr')
                imge_texture.blit_buffer(buf, colorfmt="bgr", bufferfmt="ubyte")
                self.img_web_cam.texture = imge_texture 
            
            
            if self.is_run_dectection == True :
                
                face_cascade = cv2.CascadeClassifier("C:/Users/DELL/Desktop/traitementImages/exercices/devoir/src/cascades/data/haarcascade_frontalface_alt2.xml")
                recognizer = cv2.face.LBPHFaceRecognizer_create()
                recognizer.read("trainner.yml")

                with open("labels.pickle", "rb") as f:
                    og_labels = pickle.load(f)
                    #print(og_labels)
                    labels = { value:key for key,value in  og_labels.items()}

                ret, self.frame = self.caputre.read() 
                gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
                line = 2

                for (x,y,h,w) in faces:

                    r_o_iGray = gray[y:y+h, x:x+w]

                    id_, conf = recognizer.predict(r_o_iGray)
                    if conf>=30:

                        font = cv2.FONT_HERSHEY_SIMPLEX
                        name = labels[id_]
                        color = (0,255,0)
                        trait = 2
                        self.frame = cv2.putText(self.frame, name, (x,y), font, 1,color, trait, cv2.LINE_AA)

                        color = (255,0,0) #BGR
                        self.frame = cv2.rectangle(self.frame, (x,y), ( x+w, y+h), color, line)

                buf = cv2.flip(self.frame, 0).tobytes()
                imge_texture = Texture.create(size=(self.frame.shape[1], self.frame.shape[0]),colorfmt='bgr')
                imge_texture.blit_buffer(buf, colorfmt="bgr", bufferfmt="ubyte")
                self.img_web_cam.texture = imge_texture 


if __name__=='__main__':
    Main().run()