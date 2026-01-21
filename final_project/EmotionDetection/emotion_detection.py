import requests

def emotion_detector(text_to_analyze: str):
    
    url = "https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
    headers = {
        "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock",
        "Content-Type": "application/json",
    }

    input_json = {"raw_document": {"text": text_to_analyze}}

    response = requests.post(url, headers=headers, json=input_json)
    response.raise_for_status()

    # Convert response to a Python dictionary
    response_dict = response.json()

    # Extract emotion scores (Watson typically returns a dict under: response_dict["emotionPredictions"][0]["emotion"])
    emotion_scores = (
        response_dict.get("emotionPredictions", [{}])[0]
        .get("emotion", {})
    )

    # Pull required emotions with default 0.0 if missing
    anger_score = float(emotion_scores.get("anger", 0.0))
    disgust_score = float(emotion_scores.get("disgust", 0.0))
    fear_score = float(emotion_scores.get("fear", 0.0))
    joy_score = float(emotion_scores.get("joy", 0.0))
    sadness_score = float(emotion_scores.get("sadness", 0.0))

    # Determine dominant emotion (highest score)
    required = {
        "anger": anger_score,
        "disgust": disgust_score,
        "fear": fear_score,
        "joy": joy_score,
        "sadness": sadness_score,
    }
    dominant_emotion = max(required, key=required.get) if required else ""

    # Return in the requested format
    return {
        "anger": anger_score,
        "disgust": disgust_score,
        "fear": fear_score,
        "joy": joy_score,
        "sadness": sadness_score,
        "dominant_emotion": dominant_emotion,
    }
