from google import genai

# paste your API key here
client = genai.Client(api_key="AIzaSyDTyCHMlkULP9ongQcCbsouXnqyb6XjVXk")


def doctor_ai_response(question, diagnosis):

    # keywords allowed for medical questions
    allowed_keywords = [
        "fracture", "bone", "xray", "x-ray", "injury",
        "pain", "treatment", "heal", "recovery",
        "arm", "leg", "swelling", "doctor", "cast"
    ]

    # check if question is relevant
    if not any(word in question.lower() for word in allowed_keywords):
        return """
This AI assistant only answers questions related to the analyzed X-ray.

Please ask questions about:

• fracture severity  
• treatment options  
• recovery time  
• symptoms  
• bone injury  
• precautions after fracture
"""

    bone = diagnosis["bone_region"]
    fracture = diagnosis["fracture_detected"]
    severity = diagnosis["severity"]
    confidence = diagnosis["confidence"]

    prompt = f"""
You are an orthopedic AI doctor.

An AI analyzed an X-ray image with the following results:

Bone Region: {bone}
Fracture Detected: {fracture}
Severity Level: {severity}
AI Confidence: {confidence}%

Patient question:
{question}

Explain clearly in bullet points for easy reading.

Format the answer like this:

Condition
- explanation

Severity
- explanation

Possible Symptoms
- symptom 1
- symptom 2
- symptom 3

Treatment / Next Steps
- step 1
- step 2

Recovery
- healing time

Important Note
- this AI analysis is not a final diagnosis
- patient should consult a real orthopedic doctor
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        if response and hasattr(response, "text") and response.text:
            return response.text

        return """
Condition
- AI response could not be generated.

Severity
- Unable to determine through chatbot response.

Possible Symptoms
- Pain
- Swelling
- Difficulty in movement

Treatment / Next Steps
- Please consult an orthopedic doctor
- Recheck the uploaded X-ray

Recovery
- Depends on the type and severity of fracture

Important Note
- this AI analysis is not a final diagnosis
- patient should consult a real orthopedic doctor
"""

    except Exception as e:
        print("Doctor AI Error:", e)

        # fallback response so chatbot still works
        return f"""
Condition
- The analyzed X-ray indicates: {fracture} in {bone}.

Severity
- Estimated severity level is {severity} with AI confidence of {confidence}%.

Possible Symptoms
- Pain
- Swelling
- Difficulty moving the affected area

Treatment / Next Steps
- Follow the system recommendation
- Consult an orthopedic doctor for proper diagnosis

Recovery
- Recovery time depends on the severity and treatment method

Important Note
- The live AI service is currently unavailable
- This fallback explanation is based on the model prediction only
- patient should consult a real orthopedic doctor
"""