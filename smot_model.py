import pandas as pd
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import joblib

df = pd.read_csv(r'C:\Users\sanat\OneDrive\Desktop\GLOF\dataset\GLOFData.csv')

# Shuffle the dataset
df = df.sample(frac=1).reset_index(drop=True)

label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Prepare features and labels
X = df.iloc[:, :-2]
y = df.iloc[:, -2]

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler, label encoders, and feature names
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(X.columns, 'feature_names.pkl')


# Encode categorical variables
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Verify that all data is numeric
print("Data types after encoding:")
print(df.dtypes)

# Separate features (X) and labels (y)
X = df.iloc[:, :-2]  # All columns except the last two
y = df.iloc[:, -2]   # Second-to-last column

# Normalize/Standardize numerical features
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
# Split the dataset into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)







print("Before OverSampling, counts of label '1': {}".format(sum(y_train == 1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train == 0)))

# import SMOTE module from imblearn library
# pip install imblearn (if you don't have imblearn in your system)
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 2)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train.ravel())

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

print("After OverSampling, counts of label '1': {}".format(sum(y_train_res == 1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res == 0)))







X_train=X_train_res
y_train=y_train_res


# print(X_train)


import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

# Assuming X_train_res and y_train_res are already defined from SMOTE
print("Original X_train_res shape:", X_train_res.shape)

# Convert X_train and X_test to NumPy arrays
X_train_array = X_train.values  # or use X_train.to_numpy()
X_test_array = X_test.values

# Check current shape
print("Original X_train shape:", X_train_array.shape)

# Reshape to (samples, timesteps=1, features)
X_train_reshaped = X_train_array.reshape((X_train_array.shape[0], 1, X_train_array.shape[1]))
X_test_reshaped = X_test_array.reshape((X_test_array.shape[0], 1, X_test_array.shape[1]))

print("Reshaped X_train shape:", X_train_reshaped.shape)

# Ensure y_train_res and y_test are binary (0 or 1)
y_train_res = np.array(y_train_res).astype(np.float32)
y_test = np.array(y_test).astype(np.float32)





from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
import tensorflow as tf

# Adjusted class weights (optional, depending on class balance)
class_weights = {0: 1., 1: 75.}  # Experiment with this if needed

# Model Architecture
model = tf.keras.Sequential([
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True, input_shape=(1, X_train_res.shape[1]))),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=False)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(1, activation='sigmoid')  # Sigmoid activation for binary classification
])

def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_true = tf.cast(y_true, tf.float32)  # Convert y_true to float32
        y_pred = tf.cast(y_pred, tf.float32)  # Ensure y_pred is float32
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        loss = alpha * tf.math.pow(1 - y_pred, gamma) * cross_entropy
        return tf.reduce_mean(loss, axis=-1)  # Use axis=-1 for the correct reduction
    return focal_loss_fixed

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # Adjusted learning rate
model.compile(optimizer=optimizer, loss=focal_loss(gamma=2., alpha=0.25), metrics=['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint('best_bilstm_model_new_soft.h5', monitor='val_loss', save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model with class weights
history = model.fit(
    X_train_reshaped, 
    y_train_res, 
    validation_split=0.2, 
    epochs=50, 
    batch_size=32, 
    callbacks=[checkpoint, early_stopping], 
    class_weight=class_weights
)

# Predictions
y_pred = model.predict(X_test_reshaped)
y_pred_classes = (y_pred > 0.5).astype(int).flatten()  # Flatten for binary classification

# Evaluation
log_f1 = f1_score(y_test, y_pred_classes, average='macro')
print('F1 Score:', log_f1)

cm = confusion_matrix(y_test, y_pred_classes)
print("Confusion Matrix:")
print(cm)

cm_display = ConfusionMatrixDisplay(confusion_matrix=cm)
cm_display.plot()

print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes))



