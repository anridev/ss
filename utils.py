import re
import string

def preprocess_text(text: str) -> str:
    """
    Lightweight preprocessing:
    - lowercase
    - URL normalization
    - remove excessive repeated characters
    - strip punctuation
    """
    if not text:
        return ""
    
    # Lowercase
    text = text.lower()
    
    # URL normalization
    text = re.sub(r'https?://\S+|www\.\S+', ' [URL] ', text)
    
    # Remove excessive repeated characters (e.g., "freeeeee" -> "free")
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    
    # Handle spaced out characters (e.g., "F R E E" -> "FREE")
    # Only if they are single characters separated by spaces, and more than 2 in a row
    text = re.sub(r'\b(\w\s){2,}\w\b', lambda m: m.group().replace(' ', ''), text)
    
    # Strip punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
