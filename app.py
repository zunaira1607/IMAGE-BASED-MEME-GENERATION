import streamlit as st
import os
import io
import base64
from PIL import Image

from mock_models import LSTMMemeCaptioner, TransformerMemeCaptioner
from tokenizers import CharacterTokenizer, WordTokenizer
from inference import generate_caption
from utils import create_meme

# Set page configuration
st.set_page_config(
    page_title="Deep Learning Meme Generator",
    page_icon="🤣",
    layout="wide"
)

# Title and description
st.title("Edgy Humor: Deep Learning Meme Generator")
st.markdown("""
This application generates humorous meme captions using deep learning models as described in research.
Upload an image and select your preferred model and tokenization method to generate a meme.
""")

# Sidebar for model selection
st.sidebar.header("Model Settings")

model_type = st.sidebar.selectbox(
    "Select Model Type",
    ["LSTM", "Transformer"]
)

tokenizer_type = st.sidebar.selectbox(
    "Select Tokenization Method",
    ["Character-level", "Word-level"]
)

# Load models (in a real implementation, these would be pre-trained)
@st.cache_resource
def load_models():
    # Simplified model loading since we're not using actual models
    device = "cpu"
    
    # Initialize models (these would normally be loaded from saved weights)
    lstm_char = LSTMMemeCaptioner(
        embedding_dim=256,
        hidden_dim=512,
        vocab_size=128,  # Character vocab size
        device=device
    )
    
    lstm_word = LSTMMemeCaptioner(
        embedding_dim=256,
        hidden_dim=512,
        vocab_size=10000,  # Word vocab size estimate
        device=device
    )
    
    transformer_char = TransformerMemeCaptioner(
        embedding_dim=256,
        num_heads=8,
        num_layers=6,
        vocab_size=128,
        device=device
    )
    
    transformer_word = TransformerMemeCaptioner(
        embedding_dim=256,
        num_heads=8,
        num_layers=6,
        vocab_size=10000,
        device=device
    )
    
    # Create tokenizers
    char_tokenizer = CharacterTokenizer()
    word_tokenizer = WordTokenizer()
    
    return {
        "LSTM_char": (lstm_char, char_tokenizer),
        "LSTM_word": (lstm_word, word_tokenizer),
        "Transformer_char": (transformer_char, char_tokenizer),
        "Transformer_word": (transformer_word, word_tokenizer)
    }

# Load models with caching
models = load_models()

# Image upload section
st.header("Upload an Image")
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    
    # Resize for display if needed
    display_image = image.copy()
    if max(display_image.size) > 800:
        display_image.thumbnail((800, 800))
    
    st.image(display_image, caption="Uploaded Image", use_container_width=True)
    
    # Generate button
    if st.button("Generate Meme"):
        with st.spinner("Processing image and generating caption..."):
            try:
                # Determine which model and tokenizer to use
                model_key = f"{model_type}_{'char' if tokenizer_type == 'Character-level' else 'word'}"
                model, tokenizer = models[model_key]
                
                # Generate caption
                # We'll use the sample captions directly since we don't have trained models
                caption = generate_caption(model, tokenizer, None)
                
                # Use our utility function to create the meme
                from utils import create_meme
                meme_image = create_meme(image, caption)
                
                # Display the meme
                st.header("Generated Meme")
                st.image(meme_image, caption=f"Generated caption: {caption}", use_container_width=True)
                
                # Download button for the meme
                buffered = io.BytesIO()
                meme_image.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                href = f'<a href="data:file/jpg;base64,{img_str}" download="generated_meme.jpg">Download Meme</a>'
                st.markdown(href, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error generating meme: {str(e)}")

# Information section about the models
st.header("About the Models")
st.markdown("""
### Model Types
- **LSTM**: Uses Long Short-Term Memory networks with global image embedding for caption generation.
- **Transformer**: Uses self-attention mechanism and encoder-decoder architecture for more contextual understanding.

### Tokenization Methods
- **Character-level**: Processes text character by character, which showed better generalization in the research.
- **Word-level**: Processes text word by word, which is better for maintaining semantic meaning.

Based on the research, LSTM with character-level tokenization performed best in human evaluations.
""")

# Footer
st.markdown("---")
st.markdown("Built based on research: Edgy Humor: Image-Based Meme Generation Using Deep Learning")
