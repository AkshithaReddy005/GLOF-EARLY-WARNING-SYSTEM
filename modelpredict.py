import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import custom_object_scope

# Define the custom loss function
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        loss = alpha * tf.math.pow(1 - y_pred, gamma) * cross_entropy
        return tf.reduce_mean(loss, axis=1)
    return focal_loss_fixed

# Load scaler, label encoders, and feature names
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')
feature_names = joblib.load('feature_names.pkl')

# Load the model with custom object scope
with custom_object_scope({'focal_loss_fixed': focal_loss(gamma=2., alpha=0.25)}):
    model = tf.keras.models.load_model('best_bilstm_model_new_soft.h5')

def preprocess_input(input_str):
    # Split the input string and convert it to a list
    input_list = input_str.replace('"', '').split(',')
    
    # Create DataFrame
    data = pd.DataFrame([input_list], columns=feature_names)

    # Encode categorical features
    for column, le in label_encoders.items():
        if column in data.columns:
            known_labels = list(le.classes_)
            most_frequent_label = known_labels[0]  # Handle unseen labels
            data[column] = data[column].apply(lambda x: x if x in known_labels else most_frequent_label)
            data[column] = le.transform(data[column])

    # Ensure all columns are numeric before scaling
    data = data.apply(pd.to_numeric, errors='coerce')

    # Align columns by filling missing columns with zeros
    missing_cols = set(feature_names) - set(data.columns)
    for col in missing_cols:
        data[col] = 0

    data = data[feature_names]  # Reorder columns to match training data

    # Scale features
    data = pd.DataFrame(scaler.transform(data), columns=data.columns)

    # Reshape data for LSTM model
    data = data.values.reshape((data.shape[0], 1, data.shape[1]))
    return data

def predict(input_str):
    # Preprocess the input
    preprocessed_input = preprocess_input(input_str)
    
    # Make predictions
    predictions = model.predict(preprocessed_input)
    predicted_class = (predictions > 0.5).astype(int).flatten()[0]
    return predicted_class

def input_model(inpu):
    try:
        predicted_class = predict(inpu)
        print(f'Predicted Class: {predicted_class}')
        return {"message": predicted_class}
    except ValueError as e:
        print(f"Error: {e}")

# Example usage
input_str = '"43","GL093898E30128N",30.128,93.898,"M(e)","Brahmaputra","Yarlung Zangbo","Nyang",0.095,4224,0,1,"siliciclasticSeds","Plateau",143943,15522582,4249792,0.274,168,667382,0.157,0,4187,4756,5075,5085,5424,6218,2031,2.73,0.61,0.50371642588498,0.586816961265914,0.605873675652009,71,10,33,33,31,33,62.1,70,71,71,71,71,71,71,71,71,3.5,4.5,4.7,4.61,4.9,5.3,71,71,70,64,50,2,0,0,0,0,0,1.62,69,275.9,608.71,13.12,-11.98,25.09,9.52,-4.67,10.02,-6.58,579.7,107.7,3,85.1,321.9,12.8,309.4,12.8,-1.73,69,274.5,617.01,9.93,-15.3,25.21,6.35,-9.7,6.81,-9.93,756.5,144,4.3,87.1,429,15.6,414.9,15.6"'
print(input_model(input_str))



















































# import joblib
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# import pandas as pd
# from sklearn.utils import resample
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import classification_report, confusion_matrix, f1_score, ConfusionMatrixDisplay
# import tensorflow as tf





# # Load the CSV file
# df = pd.read_csv(r'C:\Users\sanat\OneDrive\Desktop\GLOF\dataset\GLOFData.csv')




# #dumping the random scalers 
# label_encoders = {}
# for column in df.select_dtypes(include=['object']).columns:
#     le = LabelEncoder()
#     df[column] = le.fit_transform(df[column])
#     label_encoders[column] = le

# scaler = StandardScaler()
# X = df.iloc[:, :-2] 
# scaler.fit(X)

# joblib.dump(scaler, 'scaler.pkl')












# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# import tensorflow as tf
# import joblib
# # C:\Users\sanat\OneDrive\Desktop\GLOF\best_bilstm_model_new_soft.h5
# #import the model, label and scaler 
# model = tf.keras.models.load_model('best_bilstm_model_new_soft.h5')
# label_encoders = joblib.load('label_encoders.pkl')
# scaler = joblib.load('scaler.pkl')

# def preprocess_input(input_str):
#     # Split the input string and convert it to a list
#     input_list = input_str.replace('"', '').split(',')
    
#     # Create DataFrame
#     data = pd.DataFrame([input_list], columns=X.columns)

#     # Encode categorical features
#     for column, le in label_encoders.items():
#         if column in data.columns:
#             # Get unique values the encoder has seen
#             known_labels = list(le.classes_)
            
#             # Map unseen labels to a default valid label, e.g., the most frequent one
#             most_frequent_label = known_labels[0]  # This is just an example, change as needed
            
#             # Encode labels, mapping unseen ones to most frequent or another label
#             data[column] = data[column].apply(lambda x: x if x in known_labels else most_frequent_label)
            
#             # Now encode the values
#             data[column] = le.transform(data[column])

#     # Ensure all columns are numeric before scaling
#     data = data.apply(pd.to_numeric, errors='coerce')

#     # Align columns by filling missing columns with zeros
#     missing_cols = set(X.columns) - set(data.columns)
#     for col in missing_cols:
#         data[col] = 0  # or np.nan if you need to handle it differently

#     data = data[X.columns]  # Reorder columns to match training data

#     # Scale features
#     data = pd.DataFrame(scaler.transform(data), columns=data.columns)

#     # Reshape data for LSTM model
#     data = data.values.reshape((data.shape[0], 1, data.shape[1]))
#     return data


# def predict(input_str):
#     # Preprocess the input
#     preprocessed_input = preprocess_input(input_str)
    

#     predictions = model.predict(preprocessed_input)
    

#     predicted_class = np.argmax(predictions, axis=1)[0]
#     return predicted_class

# # Example usage




# #input_str = '"43","GL093898E30128N",30.128,93.898,"M(e)","Brahmaputra","Yarlung Zangbo","Nyang",0.095,4224,0,1,"siliciclasticSeds","Plateau",143943,15522582,4249792,0.274,168,667382,0.157,0,4187,4756,5075,5085,5424,6218,2031,2.73,0.61,0.50371642588498,0.586816961265914,0.605873675652009,71,10,33,33,31,33,62.1,70,71,71,71,71,71,71,71,71,3.5,4.5,4.7,4.61,4.9,5.3,71,71,70,64,50,2,0,0,0,0,0,1.62,69,275.9,608.71,13.12,-11.98,25.09,9.52,-4.67,10.02,-6.58,579.7,107.7,3,85.1,321.9,12.8,309.4,12.8,-1.73,69,274.5,617.01,9.93,-15.3,25.21,6.35,-9.7,6.81,-9.93,756.5,144,4.3,87.1,429,15.6,414.9,15.6'

# # try:
# #     predicted_class = predict(input_str)
# #     print(f'Predicted Class: {predicted_class}')
# # except ValueError as e:
# #     print(f"Error: {e}")




# def input_model(inpu):
#     try:
#         predicted_class = predict(inpu)
#         print(f'Predicted Class: {predicted_class}')
#         return {"message":predicted_class}
#     except ValueError as e:
#         print(f"Error :{e}")

# # input(input_str)