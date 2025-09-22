
import  sys
from PyQt6 import QtWidgets, QtGui, QtCore
import requests

# ---------- Result Window ----------
class ResultWindow(QtWidgets.QWidget):
    def __init__(self, image_path, result_text):
        super().__init__()
        self.setWindowTitle("Classification Results")
        self.setFixedSize(450, 550)

        # Image display
        self.image_label = QtWidgets.QLabel(self)
        pixmap = QtGui.QPixmap(image_path).scaled(350, 350, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)
        self.image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        # Results display
        self.result_label = QtWidgets.QLabel(result_text, self)
        font = QtGui.QFont()
        font.setBold(True)
        font.setPointSize(12)
        self.result_label.setFont(font)
        self.result_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.result_label.setStyleSheet("color: black;")

        # Back button
        self.back_button = QtWidgets.QPushButton("Back", self)
        self.back_button.setFixedHeight(45)
        self.back_button.setStyleSheet("""
            QPushButton {
                background-color: rgb(17, 17, 17);
                color: white;
                border-radius: 15px;
                padding: 10px 20px;
                font-size: 14px;
                border: 2px solid rgb(255, 255, 255);
            }
            QPushButton:hover {
                background-color: rgb(255, 255, 255);
                border: 2px solid rgb(255, 255, 255);
                color: black;
            }
        """)
        self.back_button.clicked.connect(self.close)

        # Layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.result_label)
        layout.addWidget(self.back_button)
        self.setLayout(layout)


# ---------- Main Window ----------
class FoodClassifierUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Food Classifier")
        self.setGeometry(300, 100, 850, 750)

        # Background
        self.background_label = QtWidgets.QLabel(self)
        self.background_label.setGeometry(0, 0, 850, 750)
        pixmap = QtGui.QPixmap("interfaces/food.png")  # صورة الخلفية
        self.background_label.setPixmap(pixmap)
        self.background_label.setScaledContents(True)

        # Choose photo button
        self.b1 = QtWidgets.QPushButton("Choose Photo", self)
        self.b1.setGeometry(60, 600, 150, 50)
        self.b1.setStyleSheet("""
            QPushButton {
                background-color: rgb(17, 17, 17);
                color: white;
                border-radius: 15px;
                padding: 10px 20px;
                font-size: 14px;
                border: 2px solid rgb(255, 255, 255);
            }
            QPushButton:hover {
                background-color: rgb(255, 255, 255);
                border: 2px solid rgb(255, 255, 255);
                color: black;
            }
        """)

        # Detect button
        self.b2 = QtWidgets.QPushButton("Detect", self)
        self.b2.setGeometry(250, 600, 150, 50)
        self.b2.setStyleSheet(self.b1.styleSheet())

        # Connect buttons
        self.b1.clicked.connect(self.upload_image)
        self.b2.clicked.connect(self.classify_image)

        self.current_image_path = None

    # Upload image
    def upload_image(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            self.current_image_path = file_path

    # Classify image via API
    def classify_image(self):
        if not self.current_image_path:
            warning = QtWidgets.QMessageBox(self)
            warning.setWindowTitle("Warning")
            warning.setText("Please select an image first!")
            warning.setIcon(QtWidgets.QMessageBox.Icon.Warning)
            warning.setStyleSheet("QLabel{color: blue; font-weight: bold;}")
            warning.exec()
            return

        try:
            with open(self.current_image_path, "rb") as f:
                files = {"file": f}
                response = requests.post("http://127.0.0.1:8000/predict", files=files)

            if response.status_code == 200:
                data = response.json()
                result_text = f"Meal: {data['class_name']}\nConfidence: {data['confidence']}%\nCalories: {data['calories']} kcal"
            else:
                result_text = f"Error contacting API ({response.status_code})"

        except Exception as e:
            result_text = f"Error: {e}"

        self.result_window = ResultWindow(self.current_image_path, result_text)
        self.result_window.show()


# ---------- Main ----------
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = FoodClassifierUI()
    window.show()
    sys.exit(app.exec())















