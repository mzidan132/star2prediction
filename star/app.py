from flask import Flask,request,render_template
import numpy as np
import pandas
import sklearn
import pickle

# importing model
model = pickle.load(open('model.pkl','rb'))
sc = pickle.load(open('standscaler.pkl','rb'))
ms = pickle.load(open('minmaxscaler.pkl','rb'))

# creating flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])
def predict():
    
    temp = request.form['Temperature']
    N = request.form['Luminosity']
    rad = request.form['Radius']
    absm = request.form['Absolute magnitude']
    st = request.form['Star type']
    feature_list = [temp,N,rad,absm,st]
    single_pred = np.array(feature_list).reshape(1, -1)

    scaled_features = ms.transform(single_pred)
    final_features = sc.transform(scaled_features)
    prediction = model.predict(final_features)

    star_dict = {1: "A", 2: "B", 3: "F", 4: "G", 5: "K", 6: "M", 7: "O"}

    if prediction[0] in star_dict:
        crop = star_dict[prediction[0]]
        result = "{} is the star".format(crop)
    else:
        result = "Sorry, we could not determine the star."
    return render_template('index.html',result = result)




# python main
if __name__ == "__main__":
    app.run(debug=True)