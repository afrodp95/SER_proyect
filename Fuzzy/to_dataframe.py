import joblib
import numpy as np 
import pandas as pd 

ravdess_data = joblib.load( "./Fuzzy/ravdess_speech_data.gz" )
ravdess_labels = joblib.load( "./Fuzzy/ravdess_numeric_labels.gz" )


mapping = {0:"neutral",1:"calm",2:"happy",3:"sad",4:"angry",5:"fearful",6:"disgust",7:"surprised"}

ravdess_data = pd.DataFrame(ravdess_data)
ravdess_data['emotion'] = ravdess_labels
ravdess_data['emotion'] = ravdess_data['emotion'].replace(mapping) 
ravdess_data['emotion'].value_counts()

ravdess_data.to_csv("./Fuzzy/radvess_data.csv")