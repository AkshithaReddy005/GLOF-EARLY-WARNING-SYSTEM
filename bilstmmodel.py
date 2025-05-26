import pandas as pd
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping





# Load the CSV file
df = pd.read_csv(r'C:\Users\sanat\OneDrive\Desktop\GLOF\dataset\GLOFData.csv')


df = df.sample(frac=1).reset_index(drop=True)



import joblib
from sklearn.preprocessing import LabelEncoder


label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

#label encoders
joblib.dump(label_encoders, 'label_encoders.pkl')

#data types
# print("Data types after encoding:")
# print(df.dtypes)



X = df.iloc[:, :-2]  
y = df.iloc[:, -2]   


scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

X = X.values.reshape((X.shape[0], 1, X.shape[1]))









X_y = pd.concat([pd.DataFrame(X.reshape(X.shape[0], -1)), y], axis=1)

# Separate majority and minority classes
X_y_majority = X_y[X_y[y.name] == 0]
X_y_minority = X_y[X_y[y.name] == 1]


X_y_minority_oversampled = resample(X_y_minority, 
                                    replace=True, 
                                    n_samples=len(X_y_majority), 
                                    random_state=42)


X_y_resampled = pd.concat([X_y_majority, X_y_minority_oversampled])


X_resampled = X_y_resampled.iloc[:, :-1].values.reshape(-1, 1, X.shape[2])
y_resampled = X_y_resampled.iloc[:, -1]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)


#Bi-directional LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2]))),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(set(y_resampled)), activation='softmax')  # Use softmax for multi-class classification
])


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#early stopping the model for better accuracy
checkpoint = ModelCheckpoint('best_bilstm_model.h5', monitor='val_loss', save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

#training the model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32, callbacks=[checkpoint, early_stopping])


y_pred = model.predict(X_test)
y_pred_classes = y_pred.argmax(axis=-1)

#F1 score
log_f1 = f1_score(y_test, y_pred_classes, average='macro')
print('F1 Score:', log_f1)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_classes)
print("Confusion Matrix:")
print(cm)

# Confusion Matrix
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm)
cm_display.plot()


print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes))