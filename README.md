# 🐱 vs 🐶 Классификатор кошек и собак

[![Streamlit App](......)]

## 📝 Описание проекта
Веб-приложение на базе нейросети, которое определяет, кто изображён на фото — кошка или собака.

## 🛠️ Технологии
- **Python** 3.13
- **TensorFlow** / **Keras** (свёрточная нейросеть)
- **Streamlit** (веб-интерфейс)
- **PIL / OpenCV** (обработка изображений)

## 🧠 Модель
- Архитектура: 4 свёрточных слоя + 2 полносвязных
- Параметров: 6.8 млн
- Точность на обучении: **94%**
- Датасет: 25,000 изображений (Dogs vs Cats)

## 🚀 Запуск локально
```bash
git clone https://github.com/IrinaG1090/cats-vs-dogs-classifier.git
cd cats-vs-dogs-classifier
pip install -r requirements.txt
streamlit run app/app.py