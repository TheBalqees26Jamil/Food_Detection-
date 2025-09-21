# Food_Detection-


📘 README (English)
🍽️ Food Classification Project

This project provides a desktop application for classifying food images using a deep learning model (ResNet50).
It integrates a PyQt6 graphical interface with a FastAPI backend, which serves the trained model for predictions.

🚀 Features

Upload an image of a meal from your computer.

Detect the type of food using a trained ResNet50 model.

Display:

Meal name

Confidence percentage

Estimated calories

Clean and modern UI with buttons (Choose Photo, Detect).

Separate result window showing the uploaded image and classification results.

🛠️ Tech Stack 
Frontend (GUI): PyQt6

Backend (API): FastAPI + Uvicorn

Model: PyTorch (ResNet50 fine-tuned on 10 food classes)

Image Processing: Torchvision, PIL


📂 Project Structure:

Food_Project/
│
├── api.py                 # FastAPI backend (model + API endpoint)
├── main.py                # PyQt6 GUI (frontend)
├── models/
│   └── best_model_resnet50_final.pth   # Trained model
├── interfaces/
│   └── food.png           # Background image for GUI


⚙️ Setup & Installation

Clone the repository:
git clone <repo-link>
cd Food_Project


Install dependencies:
pip install -r requirements.txt


Main requirements:

PyQt6

FastAPI

Uvicorn

Torch, Torchvision

Pillow

Requests

Run the API (backend):
python api.py
You should see:
Uvicorn running on http://127.0.0.1:8000
Run the GUI (frontend):
python main.py


📊 Classes & Calories
The model supports 10 food classes:

apple_pie (300 kcal)

bibimbap (450 kcal)

cannoli (350 kcal)

chicken_curry (500 kcal)

falafel (350 kcal)

french_toast (400 kcal)

ice_cream (200 kcal)

ramen (550 kcal)

sushi (300 kcal)

tiramisu (450 kcal)

Usage Flow

Launch the app (main.py).

Select an image via Choose Photo.

Click Detect → the GUI sends the image to the API.

The API runs the model and returns: food class, confidence %, and calories.

A Result Window pops up displaying the image and results.

_____________________________________________________________________--

📘 README (عربي)
🍽️ مشروع تصنيف الأطعمة

هذا المشروع يقدم تطبيق سطح مكتب لتصنيف صور الأطعمة باستخدام نموذج تعلم عميق (ResNet50).
المشروع يدمج بين واجهة رسومية (PyQt6) و خادم خلفي (FastAPI) يعمل كوسيط بين الواجهة والموديل.


المزايا

رفع صورة وجبة من جهاز المستخدم.

التعرف على نوع الطعام باستخدام نموذج ResNet50 مدرّب مسبقًا.

عرض:

اسم الوجبة

نسبة الثقة

السعرات الحرارية المتوقعة

واجهة رسومية حديثة مع أزرار (Choose Photo, Detect).

نافذة منفصلة لعرض النتيجة مع الصورة والبيانات.


بيئة العمل:

الواجهة الأمامية (GUI): PyQt6

الخلفية (API): FastAPI + Uvicorn

النموذج: PyTorch (ResNet50 مدرّب على 10 أصناف طعام) كل صنف 150 صوره.

معالجة الصور: Torchvision, PIL

📂 هيكل المشروع:
Food_Project/
│
├── api.py                 # كود الخادم (الموديل + API)
├── main.py                # كود الواجهة الرسومية
├── models/
│   └── best_model_resnet50_final.pth   # النموذج المدرّب
├── interfaces/
│   └── food.png           # صورة الخلفية للواجهة

⚙️ خطوات التشغيل

تحميل المشروع:
git clone <repo-link>
cd Food_Project

تثبيت المكتبات المطلوبة:
pip install -r requirements.txt



أهم المكتبات:

PyQt6

FastAPI

Uvicorn

Torch, Torchvision

Pillow

Requests


تشغيل الخادم (API):

python api.py

سيظهر:
Uvicorn running on http://127.0.0.1:8000

python main.py


📊 الأصناف والسعرات الحرارية

النموذج يدعم 10 أصناف طعام:

فطيرة التفاح (300 سعرة)

بيبيمباب (450 سعرة)

كانولي (350 سعرة)

كاري الدجاج (500 سعرة)

فلافل (350 سعرة)

فرنش توست (400 سعرة)

آيس كريم (200 سعرة)

رامن (550 سعرة)

سوشي (300 سعرة)

تيراميسو (450 سعرة)

🖼️ كيفية الاستخدام

تشغيل الواجهة (main.py).

اختيار صورة باستخدام زر Choose Photo.

الضغط على Detect → الواجهة ترسل الصورة للـ API.

الخادم يعيد النتيجة: نوع الطعام + نسبة الثقة + السعرات.

نافذة جديدة تعرض الصورة والنتائج.




























































