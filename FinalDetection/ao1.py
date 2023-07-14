from flask import Flask,render_template,request
from keras.models import load_model
import tensorflow

import numpy as np
from keras.models import Sequential

from keras.preprocessing import image



app=Flask(__name__)

labels=['freshApple','freshBanana','freshOrange','rottenApple','rottenBanana','rottenOrange']
model=load_model("Fresh_Rotten_Fruits_MobileNetV2_Transfer_Learning2(98).h5")
model.make_predict_function()

def predict_label(img_path):
    '''img=cv2.imread("/kaggle/input/apples-bananas-oranges/original_data_set/freshbanana/Screen Shot 2018-06-12 at 10.00.42 PM.png")

    img=cv2.resize(img,(224,224))
    imgarray=np.array(img)
    imgarray=imgarray.reshape(1,224,224,3)
    a=model.predict(imgarray)
    index=a.argmax()'''
    i=tensorflow.keras.utils.load_img(img_path,target_size=(224,224))
    i=np.array(i)/255.0
    i=i.reshape(1,224,224,3)
    predictions = model.predict(i)  
    index=predictions.argmax()
    '''predicted_labels = np.argmax(predictions, axis=1)
    #p=model.predict_classes(i)
    tensorflow.keras.preprocessing.image.img_to_array
    predicted_labels = tuple(predicted_labels)
    probabilities = np.max(predictions, axis=1)
    for i in range(len(predicted_labels)):
       print(f"Prediction: Class {predicted_labels[i]}, Probability: {probabilities[i]}")'''
    
    print(index)
    print(predictions)
    
   
    return np.max(predictions);

def predfruit(img_path):
    i=tensorflow.keras.utils.load_img(img_path,target_size=(224,224))
    i=np.array(i)/255.0
    i=i.reshape(1,224,224,3)
    predictions = model.predict(i)
    index=predictions.argmax()

    if labels[index].startswith('f'):
        return 'fresh'
    else:
        return 'rotten'

def dfruit(img_path):
    i=tensorflow.keras.utils.load_img(img_path,target_size=(224,224))
    i=np.array(i)/255.0
    i=i.reshape(1,224,224,3)
    predictions = model.predict(i)  
    index=predictions.argmax()
    if labels[index].startswith('f'):
        return labels[index][5:]
    else:
        return labels[index][6:]

@app.route("/")
def main():
    return render_template("index.html")
@app.route("/submit",methods=['GET',"POST"])
def get_output():
    if request.method=="POST":
        img=request.files['my_image']
        try:
            img_path="static/"+img.filename
            img.save(img_path)
            p=predict_label(img_path) 
            k=predfruit(img_path)
            return render_template("ps.html",prediction=k,result=round(p*100,3),ness=k,fruit=dfruit(img_path),img_path=img_path)
        except:
            return render_template("ps.html")
    return render_template("ps.html")
if __name__=="__main__":
    app.  run(debug=True)


