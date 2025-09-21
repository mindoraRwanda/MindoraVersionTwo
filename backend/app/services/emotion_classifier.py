import torch
from sentence_transformers import SentenceTransformer, util

# Configure device for GPU usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"SentenceTransformer using device: {device}")

# Shared model cache to avoid loading multiple instances
_model_cache = {}

def get_emotion_model():
    model_name = "BAAI/bge-small-en"
    if model_name not in _model_cache:
        print(f"Loading emotion model: {model_name}")
        _model_cache[model_name] = SentenceTransformer(model_name, device=device)
    return _model_cache[model_name]

# Global model and embeddings
model = None
emotion_embeddings = None

def initialize_emotion_classifier():
    """Initializes the emotion classifier model and precomputes embeddings."""
    global model, emotion_embeddings
    
    if model is None:
        model = get_emotion_model()
        emotion_embeddings = {
            label: model.encode(prompts, convert_to_tensor=True)
            for label, prompts in emotion_examples.items()
        }
    return model

# Emotion categories and example prompts
emotion_examples = {
    "sadness": [
        "I feel like crying all the time.",
        "Nothing makes me happy anymore.",
        "I’m tired of everything.",
        "I don’t see the point in anything.",
        "I feel so low I can’t get out of bed."
    ],
    "anxiety": [
        "I'm always overthinking everything.",
        "I can't calm down, my mind won’t stop racing.",
        "I’m scared something bad will happen.",
        "I feel shaky and nervous all day.",
        "I worry about everything, even small things."
    ],
    "stress": [
        "I feel like I’m drowning in responsibilities.",
        "I can't take any more pressure.",
        "Everything is too much right now.",
        "I feel burned out.",
        "I’m overwhelmed and can’t focus."
    ],
    "fear": [
        "I’m terrified for no reason.",
        "I had a panic attack earlier.",
        "I feel unsafe, even at home.",
        "I’m afraid something is going to go wrong.",
        "My heart races and I can’t breathe when I think about it."
    ],
    "anger": [
        "I get irritated at everyone.",
        "I feel like screaming sometimes.",
        "Everything makes me so mad lately.",
        "I can’t control my temper.",
        "I feel like I’m going to explode."
    ],
    "guilt": [
        "I hate myself for what I did.",
        "I feel so ashamed.",
        "I can’t forgive myself.",
        "I’m a failure to everyone.",
        "People would hate me if they knew the truth."
    ],
    "craving": [
        "I really want to drink again.",
        "I’m trying to quit but I can’t stop.",
        "The urge to use is too strong.",
        "I feel like I need something to cope.",
        "I keep going back to old habits."
    ],
    "numbness": [
        "I don’t feel anything anymore.",
        "It’s like I’m empty inside.",
        "I can’t care about anything.",
        "Nothing matters to me now.",
        "I just go through the motions."
    ],
    "joy": [
        "Today actually felt okay.",
        "I laughed and it felt good.",
        "I feel lighter than usual.",
        "Something good finally happened.",
        "I think I’m starting to feel better."
    ],
    "neutral": [
        "Hello",
        "Hi",
        "How are you?",
        "Just checking in.",
        "Hey there"
        ]

}

def classify_emotion(user_input: str) -> str:
    """Classifies the emotion of a given text input."""
    global model, emotion_embeddings
    
    if model is None or emotion_embeddings is None:
        raise RuntimeError("Emotion classifier is not initialized. Call initialize_emotion_classifier() first.")

    input_embedding = model.encode(user_input, convert_to_tensor=True)

    best_label = "neutral"
    best_score = -1

    for label, embeddings in emotion_embeddings.items():
        cosine_scores = util.cos_sim(input_embedding, embeddings)
        avg_score = torch.mean(cosine_scores).item()

        if avg_score > best_score:
            best_score = avg_score
            best_label = label

    # Set a confidence threshold
    threshold = 0.35
    if best_score < threshold:
        return "neutral"

    return best_label
