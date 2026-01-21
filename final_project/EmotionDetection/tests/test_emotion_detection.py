import pytest

from EmotionDetection.emotion_detection import emotion_detector


ALLOWED_EMOTIONS = {"anger", "disgust", "fear", "joy", "sadness"}


@pytest.mark.parametrize(
    "text, expected_dominant",
    [
        ("I am so happy and excited today!", "joy"),
        ("I hate working long hours.", "anger"),
        ("This is disgusting and gross.", "disgust"),
        ("I am scared about what might happen.", "fear"),
        ("I feel very sad and depressed.", "sadness"),
    ],
)
def test_emotion_detector_dominant_emotion(text, expected_dominant):
    result = emotion_detector(text)

    # Basic contract checks
    assert isinstance(result, dict)

    for key in ["anger", "disgust", "fear", "joy", "sadness"]:
        assert key in result, f"Missing key '{key}' in result: {result}"
        assert isinstance(result[key], float), f"Expected float for '{key}', got {type(result[key])}: {result}"

    assert "dominant_emotion" in result, f"Missing key 'dominant_emotion' in result: {result}"
    assert result["dominant_emotion"] in ALLOWED_EMOTIONS, f"Unexpected dominant_emotion: {result}"

    # Expected dominant emotion for this statement
    assert result["dominant_emotion"] == expected_dominant, (
        f"Dominant emotion mismatch for text: {text!r}\n"
        f"Expected: {expected_dominant}\n"
        f"Got: {result['dominant_emotion']}\n"
        f"Full result: {result}"
    )
