import pickle
import pandas as pd

# import the ml model
with open('model/model.pkl', 'rb') as f:
    model = pickle.load(f)

MODEL_VERSION = '1.0.0'

class_labels = model.classes_.tolist()

def predict_output(user_input: dict) -> dict:
    input_df = pd.DataFrame([user_input])
    predict_class = model.predict(input_df)[0]
    
    # get probabilities for all classes
    probabilities = model.predict_proba(input_df)[0]
    confidence = max(probabilities)
    
    # create mapping 
    class_probs = dict(zip(class_labels, map(lambda x: round(x, 2), probabilities)))

    return {
        'predicted_category': predict_class,
        'confidence': round(confidence, 2),
        'class_probabilities': class_probs
    }