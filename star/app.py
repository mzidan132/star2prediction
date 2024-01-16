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

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer
auto_mpg = pd.read_csv('auto-mpg.csv', na_values='?')
print(auto_mpg.shape)
# Impute missing values using the median


# Check unique values for the 'horsepower' attribute
print("Unique values for 'horsepower':")
print(auto_mpg['horsepower'].unique())


from sklearn.preprocessing import LabelEncoder

# Instantiate the LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the 'car name' column
auto_mpg['car name'] = label_encoder.fit_transform(auto_mpg['car name'])

imputer = SimpleImputer(strategy='median')
mpg = pd.DataFrame(imputer.fit_transform(auto_mpg), columns=auto_mpg.columns)

# Split the dataset into 75% train and 25% test
X = mpg.drop(['mpg'], axis=1)
y = mpg['mpg']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# DecisionTreeRegressor
dt = DecisionTreeRegressor(max_depth=8, min_samples_leaf=0.13)

# Fit dt with train data
dt.fit(X_train, y_train)



# python main
if __name__ == "__main__":
    app.run(debug=True)
