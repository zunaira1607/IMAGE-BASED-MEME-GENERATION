import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image

# Initialize ResNet model for feature extraction
@torch.no_grad()
def load_resnet():
    """
    Load a pre-trained ResNet model for image feature extraction.
    Returns the model without the final classification layer.
    """
    # Load pre-trained ResNet50
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    
    # Remove the final fully connected layer to get features
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    
    # Set to evaluation mode
    feature_extractor.eval()
    
    # Move to GPU if available
    if torch.cuda.is_available():
        feature_extractor = feature_extractor.cuda()
        
    return feature_extractor

# Create the transformation pipeline for images
def get_preprocessing_transforms():
    """
    Returns the transformation pipeline for preprocessing images for ResNet.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# Initialize the model and transforms
resnet_model = None
preprocess_transforms = get_preprocessing_transforms()

def preprocess_image(image):
    """
    Preprocess a PIL image for the ResNet model.
    
    Args:
        image: PIL Image object
    
    Returns:
        Tensor: Preprocessed image tensor
    """
    # Apply transformations
    image_tensor = preprocess_transforms(image).unsqueeze(0)
    
    # Move to GPU if available
    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()
    
    return image_tensor

def extract_image_features(image_tensor):
    """
    Extract features from an image using ResNet.
    
    Args:
        image_tensor: Preprocessed image tensor
    
    Returns:
        Tensor: Image feature tensor
    """
    global resnet_model
    
    # Initialize the model if not already done
    if resnet_model is None:
        resnet_model = load_resnet()
    
    with torch.no_grad():
        # Extract features
        features = resnet_model(image_tensor)
        # Flatten the features
        features = features.view(features.size(0), -1)
    
    return features
