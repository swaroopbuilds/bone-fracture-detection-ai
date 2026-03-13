from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

from fracture_box import detect_fracture_box
from fracture_heatmap import generate_heatmap
from doctor_ai import doctor_ai_response

app = Flask(__name__)

model = load_model("model.h5")

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {"png","jpg","jpeg"}

latest_diagnosis = None


def allowed_file(filename):
    return "." in filename and filename.rsplit(".",1)[1].lower() in ALLOWED_EXTENSIONS


def predict_fracture(img_path):

    img = image.load_img(img_path, target_size=(150,150))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    prediction = model.predict(img)

    class_index = np.argmax(prediction)

    classes = [
        "Arm_fracture",
        "Arm_normal",
        "Leg_fracture",
        "Leg_normal",
        "non_xray"
    ]

    result = classes[class_index]
    confidence = round(np.max(prediction)*100, 2)

    return result, confidence


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    global latest_diagnosis

    if "image" not in request.files:
        return render_template("index.html", error="No file uploaded")

    file = request.files["image"]

    if file.filename == "":
        return render_template("index.html", error="No file selected")

    if file and allowed_file(file.filename):

        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        result, confidence = predict_fracture(filepath)

        if result == "non_xray":
            return render_template(
                "index.html",
                error="Invalid image. Please upload a valid bone X-ray."
            )

        box_path = detect_fracture_box(filepath)
        heatmap_path = generate_heatmap(filepath)

        if "Arm" in result:
            bone_region = "Forearm / Arm (Radius/Ulna/Humerus)"
        elif "Leg" in result:
            bone_region = "Lower Leg (Tibia/Fibula)"
        else:
            bone_region = "Unknown"

        if "fracture" in result:

            fracture_status = "Fractured"

            if confidence < 60:
                severity = "Mild"
            elif confidence < 80:
                severity = "Moderate"
            else:
                severity = "Severe"

            suggestion = "Consult an orthopedic doctor immediately."

            analysis = [
                "Possible bone discontinuity detected",
                "Irregular bone edge pattern observed",
                "High fracture probability"
            ]

        else:

            fracture_status = "Normal"
            severity = "None"
            suggestion = "Bone appears normal."

            analysis = [
                "Bone alignment appears normal",
                "No fracture patterns detected",
                "Bone structure continuity preserved"
            ]


        diagnosis_summary = {
            "bone_region": bone_region,
            "fracture_detected": fracture_status,
            "severity": severity,
            "confidence": confidence,
            "recommendation": suggestion
        }

        latest_diagnosis = diagnosis_summary

        return render_template(
            "result.html",
            result=fracture_status,
            confidence=confidence,
            severity=severity,
            suggestion=suggestion,
            analysis=analysis,
            img_path=filepath,
            box_path=box_path,
            heatmap_path=heatmap_path,
            diagnosis_summary=diagnosis_summary
        )

    else:
        return render_template("index.html", error="Only JPG/PNG images are allowed")


@app.route("/ask_ai", methods=["POST"])
def ask_ai():

    data = request.get_json()
    question = data["question"]

    if latest_diagnosis is None:
        response = "Please analyze an X-ray first."
    else:
        response = doctor_ai_response(question, latest_diagnosis)

    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(debug=True)