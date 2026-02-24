import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Настройка страницы
st.set_page_config(
    page_title="Кошки vs Собаки",
    page_icon="🐱",
    layout="centered"
)

# Заголовок
st.title("🐱 Кошка или 🐶 Собака?")
st.markdown("### Загрузи фото, и нейросеть определит, кто на нём")

# Загрузка модели
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('models/cats_dogs_cnn_final.keras')
        return model
    except:
        try:
            model = tf.keras.models.load_model('models/cats_dogs_cnn_final.h5')
            return model
        except Exception as e:
            st.error(f"❌ Не удалось загрузить модель: {e}")
            return None

model = load_model()

# Размер изображения (должен совпадать с обучением)
IMG_SIZE = 150

# Загрузка файла
uploaded_file = st.file_uploader(
    "Выбери фото...", 
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file is not None and model is not None:
    # Показываем загруженное фото
    image = Image.open(uploaded_file)
    st.image(image, caption='Твоё фото', use_container_width=True)
    
    # Кнопка анализа
    if st.button("🔍 Определить", type="primary"):
        with st.spinner('Нейросеть думает...'):
            # Подготовка изображения
            img = image.resize((IMG_SIZE, IMG_SIZE))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Предсказание
            prediction = model.predict(img_array, verbose=0)[0][0]
            
            # Результат
            st.markdown("---")
            st.subheader("🎯 Результат:")
            
            col1, col2 = st.columns(2)
            
            if prediction > 0.5:
                prob = prediction * 100
                with col1:
                    st.success(f"### 🐶 **Собака**")
                with col2:
                    st.metric("Уверенность", f"{prob:.1f}%")
            else:
                prob = (1 - prediction) * 100
                with col1:
                    st.success(f"### 🐱 **Кошка**")
                with col2:
                    st.metric("Уверенность", f"{prob:.1f}%")
            
            # Прогресс-бар
            st.progress(float(prediction if prediction > 0.5 else 1 - prediction))

# Информация о проекте
with st.expander("ℹ️ О проекте"):
    st.markdown("""
    - **Модель**: Свёрточная нейросеть (CNN) с 4 слоями
    - **Архитектура**: Conv2D + MaxPooling + Dense
    - **Точность на обучении**: ~94%
    - **Датасет**: 25,000 изображений кошек и собак
    - **Библиотеки**: TensorFlow, Keras, Streamlit, PIL
    """)