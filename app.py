import streamlit as st
import pandas as pd
import joblib
import requests
import io
import os

# Укажите здесь путь к локальному файлу или URL модели
MODEL_PATH_OR_URL = 'mobile_price_model.pkl'  # или 'https://example.com/path/to/model.pkl'

def load_model(path_or_url):
    if os.path.isfile(path_or_url):
        # Загружаем из файла
        return joblib.load(path_or_url)
    else:
        # Предполагаем, что это URL
        response = requests.get(path_or_url)
        response.raise_for_status()  # Проверка успешности запроса
        return joblib.load(io.BytesIO(response.content))

# Загружаем модель один раз при запуске приложения
model = load_model(MODEL_PATH_OR_URL)

st.title("Определение ценового диапазона мобильного телефона")

st.write("Введите характеристики мобильного телефона:")

ram = st.slider('ОЗУ (МБ)', min_value=512, max_value=8192)
battery_power = st.slider('Мощность батареи (mAh)', min_value=1000, max_value=2000, step=50)

px_height = st.number_input('Высота дисплея (px_height)', min_value=480, max_value=2160)
px_width = st.number_input('Ширина дисплея (px_width)', min_value=320, max_value=3840)
mobile_wt = st.number_input('Вес устройства (грамм)', min_value=50, max_value=300)

int_memory = st.slider('Внутренняя память (ГБ)', min_value=1, max_value=256)
talk_time = st.slider('Время работы батареи (часы)', min_value=1, max_value=48)
pc = st.slider('Мегапиксели основной камеры', min_value=0, max_value=108)
clock_speed = st.number_input('Частота процессора (GHz)', min_value=0.5, max_value=3.5, step=0.1)

# Преобразование входных данных в числовой формат
def convert_bool(val):
    return 1 if val == 'Да' else 0

input_data = {
    'ram': ram,
    'battery_power': battery_power,
    'px_height': px_height,  
    'px_width': px_width,    
    'mobile_wt': mobile_wt,  
    'int_memory': int_memory,
    'talk_time': talk_time,
    'pc': pc,
    'clock_speed': clock_speed
}

input_df = pd.DataFrame([input_data])

# Предсказание
st.write("Пометочка:")
st.write("0 — низкая стоимость")
st.write("1 — средняя стоимость")
st.write("2 — высокая стоимость")
st.write("3 — очень высокая стоимость")
if st.button('Определить ценовой диапазон'):
    prediction = model.predict(input_df)[0]
    st.write(f"Предполагаемый ценовой диапазон: {prediction}")
