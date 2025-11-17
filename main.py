import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score

# Tải dữ liệu + kiểm tra dữ liệu
data = pd.read_csv('data_PHANLOP_PCA.csv')
# print(data.head())
# print(data.info())
print(data.isnull().sum())

# Tiền xử lý dữ liệu
y = data['PhanKhuc_Target'] # tập đích
X = data.drop(['PhanKhuc_Target'], axis=1) #loại bỏ các cột không cần thiết

# Xử lý categorical features bằng one-hot encoding
# X = pd.get_dummies(X, columns=['ProductCategory', 'ProductBrand'])
print(X.head())

# Chia train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Chuẩn hóa dữ liệu 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


#=================================================
# Huấn luyện mô hình SVM

# Kernel RBF
model_rbf = svm.SVC(kernel='rbf', C=100.0)
# Huấn luyện mô hình
model_rbf.fit(X_train_scaled, y_train)
# Dự đoán
y_pred_rbf = model_rbf.predict(X_test_scaled)

print("\n--- Kết quả Kernel RBF ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rbf):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rbf))