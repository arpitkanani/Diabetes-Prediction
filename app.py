from flask import Flask,request,render_template,Response,redirect, url_for, session
from src.pipelines.prediction_pipeline import CustomData,PredictPipeline
import warnings 
warnings.filterwarnings('ignore')


application=Flask(__name__)
app=application
app.secret_key = "diabetes-secret-key"


#model=pickle.load(open('E:\Data Analysis\Classification-project\models\ModelForPrediction.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'POST':
        data=CustomData(
            Pregnancies = int(request.form.get('Pregnancies',0)),
            Glucose = float(request.form.get('Glucose',0)),
            BloodPressure = float(request.form.get('BloodPressure',0)),
            Insulin = float(request.form.get('Insulin',0)),
            BMI = float(request.form.get('BMI',0)),
            DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction',0)),
            Age = int(request.form.get('Age',0))
        )

        pred_df=data.get_data_as_dataframe()   
        prediction_pipeline=PredictPipeline()
            
        predict = prediction_pipeline.predict(pred_df)
        result = 'Diabetic' if int(predict[0]) == 1 else 'Non-Diabetic'

        session['prediction_result'] = result

        return redirect(url_for('predict_datapoint'))
    else:
        result = session.pop('prediction_result', None)  
        return render_template('home.html', result=result)

    


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)


