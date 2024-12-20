from flask import Flask,render_template,request,redirect  #flask--for creating web application
                                                #render_template--for rendering HTML templates
                                                #request--for handling HTTP requests
import numpy as np  #for numerical operation
import joblib   #for imported to load the pre-trained machine learning model.
app=Flask(__name__)   #An instance of the Flask class is created and app is initialized
@app.route('/predict',methods=['GET','POST'])  #The /predict route handles both GET and POST requests.
# @app.route--this is a root decorator for the end point

def predict():  
    
    if request.method=='POST':
        algo=str(request.form['algorithm'])
        print(algo)
        if algo == "knn":
            model=joblib.load('knnheartmodel.pkl')
        elif algo == "rf":
            model=joblib.load('rfcheartmodel.pkl')
        elif algo == "dt":
            model=joblib.load('dtcheartmodel.pkl')
        else:
            model=joblib.load('nbheartmodel.pkl')
        
        
        age=float(request.form['age'])
        sex=float(request.form['sex'])
        cp=float(request.form['cp'])
        trestbps=float(request.form['trestbps'])
        chol=float(request.form['chol'])
        fbs=float(request.form['fbs'])
        restecg=float(request.form['restecg'])
        talach=float(request.form['talach'])
        exang=float(request.form['exang'])
        oldpeak=float(request.form['oldpeak'])
        slope=float(request.form['slope'])
        ca=float(request.form['ca'])
        thal=float(request.form['thal'])
        
        # this extracted data is then used to create a numpy array containg the input features 
        testdata=np.array([[age,sex,cp,trestbps,chol,fbs,restecg,talach,exang,oldpeak,slope,ca,thal]])
        #the machine learning model is a loaded using joblib and a prediction is made using the input data
        # model=joblib.load('knnheartmodel.pkl')
        res=model.predict(testdata)
        target=res[0]  
        msg=f"Result of prediction = {target} Done using the {algo} Algorithm."
        print(f"Result of prediction = {target}, Done using the {algo} Algorithm.")
        #the result is then passed to the index.html template for rendring
        return render_template('index.html',result=msg)
    
def predict1():  
    
    if request.method=='POST':
        algo=str(request.form['algorithm'])
        print(algo)
        if algo == "knn":
            model=joblib.load('knnheartmodel.pkl')
        elif algo == "rf":
            model=joblib.load('rfcheartmodel.pkl')
        elif algo == "dt":
            model=joblib.load('dtcheartmodel.pkl')
        else:
            model=joblib.load('nbheartmodel.pkl')
        
        
        # this extracted data is then used to create a numpy array containg the input features 
        # testdata=np.array([[age,sex,cp,trestbps,chol,fbs,restecg,talach,exang,oldpeak,slope,ca,thal]])
        #the machine learning model is a loaded using joblib and a prediction is made using the input data
        # model=joblib.load('knnheartmodel.pkl')
        # res=model.predict(testdata)
        # target=res[0]  
        # msg=f"Result of prediction = {target} Done using the {algo} Algorithm."
        # print(f"Result of prediction = {target}, Done using the {algo} Algorithm.")
        #the result is then passed to the index.html template for rendring
        # return render_template('index.html',result=msg)

@app.route('/nodu')
def nodu(): #there is another route decorator for the root endpoint and it renders the index.html template
    return render_template('index.html')

@app.route('/')
def index(): #there is another route decorator for the root endpoint and it renders the index.html template
    return render_template('index5.html')

@app.route('/knn')
def knn(): #there is another route decorator for the root endpoint and it renders the index.html template
    return render_template('knn.html')

@app.route('/rfc')
def rfc(): #there is another route decorator for the root endpoint and it renders the index.html template
    return render_template('rfc.html')

@app.route('/dct')
def dct(): #there is another route decorator for the root endpoint and it renders the index.html template
    return render_template('dct.html')

@app.route('/nbc')
def nbc(): #there is another route decorator for the root endpoint and it renders the index.html template
    return render_template('nbc.html')

if __name__=='__main__':
    app.run(debug=True)
