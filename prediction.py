import numpy as np
import joblib
model=joblib.load('knnheartmodel.pkl')

testdata=np.array([[63,1,3,145,233,1,0,150,0,2.3,0,0,1]])

result=model.predict(testdata)

print(f"Result of prediction = {result[0]}")