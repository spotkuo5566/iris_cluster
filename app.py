from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# 載入模型
model = joblib.load("kmeans_iris_model.pkl")  # 你也可以改成 rf_model.pkl

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 從表單接收輸入
        sl = float(request.form["sepal_length"])
        sw = float(request.form["sepal_width"])
        pl = float(request.form["petal_length"])
        pw = float(request.form["petal_width"])

        # 準備特徵向量並預測
        features = np.array([[sl, sw, pl, pw]])
        prediction = model.predict(features)[0]

        target_names = ['setosa', 'versicolor', 'virginica']
        result = target_names[prediction]

        return render_template("index.html", result=result)
    except Exception as e:
        return render_template("index.html", result=f"錯誤：{str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
