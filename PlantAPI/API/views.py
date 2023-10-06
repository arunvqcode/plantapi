from django.shortcuts import render, HttpResponse, redirect
import pandas as pd
import numpy as np
from PyPDF2 import PdfReader
from io import BytesIO
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
from scipy.spatial import distance
import pickle
import math
import warnings
import cv2
import os

#Encoding and Split data into Train/Test Sets
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

#Tensorflow Keras CNN Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop

warnings.filterwarnings('ignore')

model_url = "https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2"
IMAGE_SHAPE = (224, 224)

# Create your views here.
def home(request):
    return render(request, 'index.html')


def upload(request):
    if request.method == 'POST':
        pdf_file = request.FILES['upload'].read()
        pdf = PdfReader(BytesIO(pdf_file))

        DataSet = pd.DataFrame(columns=['Page_NO', 'Title', 'Description', 'Image_URL'])
        for index in range(len(pdf.pages)):
            page = pdf.pages[index]

            text = page.extract_text()
            try:     
                splited_text = [word for word in text.split('\n') if len(word.strip()) > 0]
            
                plan_title = splited_text[0]
                plant_dec = ' '.join(splited_text[1:-1])
            except:
                plan_title = ''
                plant_dec = ''


            images = page.images
            img_urls = []
            if images:
                for image in images:
                    img = Image.open(BytesIO(image.data))
                    img.save(f'static/Images/{str(index) + image.name.split(".")[0] + ".jpeg"}', 'JPEG')
                    img_urls.append(f'static/Images/{str(index) + image.name.split(".")[0] + ".jpeg"}')
            else:
                img_urls = 0
            
            DataSet = DataSet.append({'Page_NO': index, 'Title' : plan_title, 'Description' : plant_dec, 'Image_URL': img_urls}, ignore_index=True)
            DataSet.to_csv('static/info.csv')
            
        
# ***************Uploading Vectors of the Pics********************
        ls_scores = []
        i = 0
        for img_urls in DataSet['Image_URL']:
            if img_urls != 0:
                for sub_url in img_urls :
                    scores = {'index': None, 'title': None, 'Image_url': None, 'Description': None}
                    scores['Image_url'] = sub_url
                    scores['index'] = DataSet['Page_NO'][i]
                    scores['title'] = DataSet['Title'][i]
                    scores['Description'] = DataSet['Description'][i]
                    ls_scores.append(scores)
            i = i+1
            
        
        file_path = 'static/array_list.pkl'
        with open(file_path, 'wb') as array_file:
            pickle.dump(ls_scores, array_file)
             
        return redirect("mat")

        
    return render(request, 'upload.html')



# *******************Matching Functions**************
# def extract(file):

#     layer = hub.KerasLayer(model_url)
#     model = tf.keras.Sequential([layer])
#     image = Image.open(file)
#     resized_image = image.convert('L').resize(IMAGE_SHAPE)

#     resized_image = np.stack((resized_image,)*3, axis=-1)
#     resized_image = np.array(resized_image)/255.0
#     embedding = model.predict(resized_image[np.newaxis, ...])

#     vgg16_feature_np = np.array(embedding)
#     flattended_feature = vgg16_feature_np.flatten()

#     return flattended_feature


def build(request):

    folder_dir = 'static/Images'
    data = []
    label = []

    SIZE = 128 #Crop the image to 128x128

    for folder in os.listdir(folder_dir):
        for file in os.listdir(os.path.join(folder_dir, folder)):
            label.append(folder)
            img = cv2.imread(os.path.join(folder_dir, folder, file))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            im = cv2.resize(img_rgb, (SIZE,SIZE))
            data.append(im)

    data_arr = np.array(data)
    label_arr = np.array(label)

    encoder = LabelEncoder()
    y = encoder.fit_transform(label_arr)
    y = to_categorical(y,11)
    X = data_arr/255

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=10)

    with open('static/encorder.pkl', 'wb') as f:
        pickle.dump(encoder, f)


    model = Sequential()
    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu', input_shape = (SIZE,SIZE,3)))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same',activation ='relu'))
    model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same',activation ='relu'))
    model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same',activation ='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(11, activation = "softmax"))


    datagen = ImageDataGenerator(
            rotation_range=20,
            zoom_range = 0.20,
            width_shift_range=0.3,
            height_shift_range=0.3,
            horizontal_flip=True,
            vertical_flip=True)

    datagen.fit(X_train)


    model.compile(optimizer=Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])
    batch_size=32
    epochs=64
    model.fit_generator(datagen.flow(X_train,y_train, batch_size=batch_size),
                                epochs = epochs,
                                validation_data = (X_test,y_test),
                                verbose = 1)
    
    model.save('static/model')

    return HttpResponse('Done')


def predict_flower_class(image_path, model, encoder):
    SIZE = 128

    # Load and preprocess the image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im = cv2.resize(img_rgb, (SIZE, SIZE))
    image_arr = np.array(im) / 255.0
    image_arr = np.expand_dims(image_arr, axis=0)  # Add batch dimension

    # Perform prediction
    prediction_probs = model.predict(image_arr)
    predicted_class_idx = np.argmax(prediction_probs)
    predicted_class = encoder.inverse_transform([predicted_class_idx])[0]

    return predicted_class




def match(request):
    if request.method == 'POST':
        dataset = pd.read_csv('static/info.csv')
        model = tf.keras.models.load_model('static/model')
        encoder = ''
        with open('static/encorder.pkl', 'rb') as f:
            encoder = pickle.load(f)

        img = request.FILES['image']
        img_name = img.name
        t_img = Image.open(img)
        t_img.save(f'static/target_image.jpeg')
        matched_img = predict_flower_class('static/target_image.jpeg', model, encoder)
        print('Matched image:', matched_img)

        # info = ['static/target_image.jpeg', img_index, img_title, Shortest_image,  shortest_dc, desc ]
        output = 'target Image :  {}<br><bt>Matched Image : {}'.format(img_name, matched_img)
        data = dataset[dataset['Title'].str.contains(matched_img.split(' ')[0])].values
        print('Dataset:', dataset)

        title = data[0][1]
        desc = data[0][2]
        img_ = data[0][3]
        print(img_)
        target_img = 'static/target_image.jpeg'
        info = {'info': [target_img, title, desc, img_]}
        return render(request, "result.html", context=info)  
           
    return render(request, "matching.html")     



def reslt(request):
    return render(request, 'result.html')

