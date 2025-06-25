# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, LabelEncoder

# # Load the CSV file
# df = pd.read_csv(r'C:\Users\sanat\OneDrive\Desktop\GLOF\dataset\GLOFData.csv')

# # Shuffle the dataset
# df = df.sample(frac=1).reset_index(drop=True)

# # Handle missing values (example: fill with mean)
# df.fillna(df.mean(), inplace=True)

# # Encode categorical variables
# label_encoders = {}
# for column in df.select_dtypes(include=['object']).columns:
#     le = LabelEncoder()
#     df[column] = le.fit_transform(df[column])
#     label_encoders[column] = le

# # Separate features (X) and labels (y)
# X = df.iloc[:, :-1]  # Assuming the last column is the label
# y = df.iloc[:, -1]

# # Normalize/Standardize numerical features
# scaler = StandardScaler()
# X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# # Split the dataset into training and testing sets with stratification
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# # Display the first row of X
# print(X.head(1))


# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Display the shapes of the resulting datasets
# print(f'X_train shape: {X_train.shape}')
# print(f'X_test shape: {X_test.shape}')
# print(f'y_train shape: {y_train.shape}')
# print(f'y_test shape: {y_test.shape}')





import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Load the CSV file
df = pd.read_csv(r'C:\Users\sanat\OneDrive\Desktop\GLOF\dataset\GLOFData.csv')

# Shuffle the dataset
df = df.sample(frac=1).reset_index(drop=True)


# Encode categorical variables
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Verify that all data is numeric
print("Data types after encoding:")
# print(df.dtypes)

# Separate features (X) and labels (y)
X = df.iloc[:, :-2]  # All columns except the last two
y = df.iloc[:, -2]   # Second-to-last column

# Normalize/Standardize numerical features
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Split the dataset into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Perform logistic regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Make predictions
y_pred = log_reg.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Display the first row of X

