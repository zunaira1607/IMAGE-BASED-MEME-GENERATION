import random

def generate_caption(model, tokenizer, image_features, max_length=50, temperature=1.0, top_k=0, top_p=0.9):
    """
    Generate a caption for the given image features using the specified model.
    
    Args:
        model: The LSTM or Transformer model for caption generation
        tokenizer: The tokenizer for encoding/decoding text
        image_features: Tensor of image features [batch_size, feature_dim]
        max_length: Maximum caption length
        temperature: Sampling temperature (higher = more random)
        top_k: If > 0, use top-k sampling
        top_p: If > 0, use nucleus sampling with this probability threshold
    
    Returns:
        str: Generated caption
    """
    # Since we don't have trained models, use sample captions for the demo
    return get_sample_captions()

def sample_meme_captions(model, tokenizer, image_features, num_samples=5, max_length=50, temperature=1.0):
    """
    Generate multiple caption samples for a single image.
    
    Args:
        model: The caption generation model
        tokenizer: The tokenizer
        image_features: Image features tensor
        num_samples: Number of captions to generate
        max_length: Maximum caption length
        temperature: Sampling temperature
    
    Returns:
        list: List of generated captions
    """
    captions = []
    
    for _ in range(num_samples):
        caption = generate_caption(
            model, 
            tokenizer, 
            image_features, 
            max_length=max_length,
            temperature=temperature,
            top_p=0.9  # Use nucleus sampling for more diversity
        )
        captions.append(caption)
    
    # Remove duplicates while maintaining order
    unique_captions = []
    for caption in captions:
        if caption not in unique_captions:
            unique_captions.append(caption)
    
    return unique_captions

def get_sample_captions():
    """
    Generate sample captions for the demo when models aren't fully trained.
    This would be replaced by actual model output in production.
    """
    sample_captions = [
        "When you realize it's Monday again",
        "Me trying to explain my code to the senior developer",
        "That feeling when your code works on the first try",
        "My brain during a final exam",
        "How I look waiting for my code to compile",
        "When someone asks if I tested my code before pushing",
        "When the client explains what they actually wanted",
        "My weekend plans vs Monday reality",
        "Me debugging code for 5 hours only to find a typo",
        "When the caffeine hits during the morning meeting"
    ]
    
    return random.choice(sample_captions)
