import os
import torch
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

def set_seed(seed=42):
    """
    Set the random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_meme(image, caption, output_path=None):
    """
    Create a meme image with the given caption
    
    Args:
        image: PIL Image object
        caption: String caption to add to the image
        output_path: Path to save the meme (optional)
    
    Returns:
        PIL Image: The generated meme image
    """
    # Make a copy of the image to avoid modifying the original
    meme = image.copy()
    
    # Determine font size based on image width (approximately 1/20 of image width)
    font_size = max(16, int(meme.width / 20))
    
    # Create a drawing context
    draw = ImageDraw.Draw(meme)
    
    # Try to use a standard font, fall back to default if not available
    try:
        font = ImageFont.truetype("Arial", font_size)
    except IOError:
        font = ImageFont.load_default()
    
    # Calculate text width to center it (approximate)
    text_width = draw.textlength(caption, font=font)
    
    # Position text at the bottom of the image with some padding
    text_x = (meme.width - text_width) / 2
    text_y = meme.height - font_size * 2
    
    # Draw text outline for better visibility
    outline_positions = [(1,1), (-1,-1), (1,-1), (-1,1)]
    for dx, dy in outline_positions:
        draw.text((text_x + dx, text_y + dy), caption, font=font, fill="black")
    
    # Draw the main text in white
    draw.text((text_x, text_y), caption, font=font, fill="white")
    
    # Save the meme if output path is provided
    if output_path:
        meme.save(output_path)
    
    return meme

def apply_text_effects(image, caption, style="impact"):
    """
    Apply different text styling based on meme style
    
    Args:
        image: PIL Image
        caption: Text caption
        style: Text style ("impact", "modern", "minimal")
    
    Returns:
        PIL Image with styled text
    """
    meme = image.copy()
    draw = ImageDraw.Draw(meme)
    
    # Determine font size based on image width
    if style == "impact":
        # Classic Impact font style (default meme style)
        font_size = int(meme.width / 12)
        try:
            font = ImageFont.truetype("Arial", font_size)
        except IOError:
            font = ImageFont.load_default()
            
        # Split caption into top and bottom text if it contains a pipe
        if "|" in caption:
            top_text, bottom_text = caption.split("|", 1)
            top_text = top_text.strip().upper()
            bottom_text = bottom_text.strip().upper()
            
            # Position top text
            text_width = draw.textlength(top_text, font=font)
            text_x = (meme.width - text_width) / 2
            text_y = font_size / 2
            
            # Draw top text
            for dx, dy in [(2,2), (-2,-2), (2,-2), (-2,2)]:
                draw.text((text_x + dx, text_y + dy), top_text, font=font, fill="black")
            draw.text((text_x, text_y), top_text, font=font, fill="white")
            
            # Position bottom text
            text_width = draw.textlength(bottom_text, font=font)
            text_x = (meme.width - text_width) / 2
            text_y = meme.height - font_size * 1.5
            
            # Draw bottom text
            for dx, dy in [(2,2), (-2,-2), (2,-2), (-2,2)]:
                draw.text((text_x + dx, text_y + dy), bottom_text, font=font, fill="black")
            draw.text((text_x, text_y), bottom_text, font=font, fill="white")
        else:
            # If no pipe separator, just add text at the bottom
            caption = caption.upper()
            text_width = draw.textlength(caption, font=font)
            text_x = (meme.width - text_width) / 2
            text_y = meme.height - font_size * 1.5
            
            for dx, dy in [(2,2), (-2,-2), (2,-2), (-2,2)]:
                draw.text((text_x + dx, text_y + dy), caption, font=font, fill="black")
            draw.text((text_x, text_y), caption, font=font, fill="white")
            
    elif style == "modern":
        # Modern style with cleaner text
        font_size = int(meme.width / 15)
        try:
            font = ImageFont.truetype("Arial", font_size)
        except IOError:
            font = ImageFont.load_default()
            
        # Add semi-transparent background
        text_bg = Image.new('RGBA', meme.size, (0, 0, 0, 0))
        text_bg_draw = ImageDraw.Draw(text_bg)
        
        text_width = draw.textlength(caption, font=font)
        text_x = (meme.width - text_width) / 2
        text_y = meme.height - font_size * 2
        
        # Draw background rectangle
        padding = font_size / 2
        text_bg_draw.rectangle(
            [(text_x - padding, text_y - padding/2), 
             (text_x + text_width + padding, text_y + font_size + padding/2)],
            fill=(0, 0, 0, 128)
        )
        
        # Combine the background with the original image
        meme = Image.alpha_composite(meme.convert('RGBA'), text_bg)
        draw = ImageDraw.Draw(meme)
        
        # Draw text
        draw.text((text_x, text_y), caption, font=font, fill="white")
        
    elif style == "minimal":
        # Minimal style with small text
        font_size = int(meme.width / 25)
        try:
            font = ImageFont.truetype("Arial", font_size)
        except IOError:
            font = ImageFont.load_default()
            
        text_width = draw.textlength(caption, font=font)
        text_x = (meme.width - text_width) / 2
        text_y = meme.height - font_size * 1.5
        
        # Draw text with subtle shadow
        draw.text((text_x+1, text_y+1), caption, font=font, fill="black")
        draw.text((text_x, text_y), caption, font=font, fill="white")
    
    return meme

def get_device():
    """Get the device to use for tensor operations"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
