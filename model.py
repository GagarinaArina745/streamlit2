import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Загрузка данных
data = pd.read_csv('train.csv')

# Выбираем нужные столбцы
features = ['ram', 'battery_power', 'px_height', 'px_width', 'mobile_wt', 'int_memory', 'talk_time', 'pc', 'clock_speed']
target = 'price_range'

df = data[features + [target]]

# Преобразование категориальных признаков в числовые
#   df[col] = df[col].astype(int)

# Разделение данных на обучающую и тестовую выборки
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Оценка модели
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Сохранение модели для использования в приложении Streamlit
joblib.dump(model, 'mobile_price_model.pkl') 