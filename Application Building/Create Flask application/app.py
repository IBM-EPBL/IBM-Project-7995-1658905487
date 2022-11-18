from flask import Flask,render_template,request, jsonify

import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import requests

app = Flask(__name__,template_folder="../templates", 
            static_folder='../static')

model = load_model('nutrition.hdf5.h5')
print("Loaded model from disk")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/image')
def image1():
    return render_template("image.html")

@app.route('/imageprediction')
def imageprediction():
    return render_template("imageprediction.html")


@app.route('/predict',methods=['POST'])
def launch():
    if request.method=='POST':
        f=request.files['file']
        
        basepath=os.path.dirname('/')
        filepath=os.path.join(basepath, f.filename)
        f.save(filepath)
        
        img=image.load_img(filepath,target_size=(64,64))
        x=image.img_to_array(img)
        x=np.expand_dims(x,axis=0)
        
        pred=np.argmax(model.predict(x),axis=1)
        print("prediction",pred)
        index=['APPLES','BANANA','ORANGE','PINEAPPLE','WATERMELON']
        
        result=str(index[pred[0]])
        apiResult=nutrition(result)
        
        final_result = {
                "result" : result, 
                "apiResult" : apiResult
            }
        print(final_result)
        return final_result
    
def nutrition(index):
    
    url="https://calorieninjas.p.rapidapi.com/v1/nutrition"
    
    querystring = {"query":index}
    
    headers = {  'X-RapidAPI-Host': 'calorieninjas.p.rapidapi.com',
    'X-RapidAPI-Key': '8c43e02098mshcb4fea7ab8fdea2p175878jsn0d0669a8826c'}
    
    response = requests.request("GET",url,headers=headers,params=querystring)
    
    return response.json()['items']

if __name__== "__main__":
    app.run(debug=False)
    
        
        
        