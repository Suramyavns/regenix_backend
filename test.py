import torch
import clip
from PIL import Image

print('Loading models...')
device = "cpu"  # Change to "cuda" if using a GPU
# model = torch.load("vision_model.pt", map_location=device,weights_only=False)
# model.eval()  # Set to evaluation mode

model, preprocess = clip.load("ViT-B/32", device=device)  # Ensure same model variant

print('Model and Preprocess instance are loaded')

def test_clip(image_path, text_description, threshold=25):
    """
    Test the saved CLIP model by comparing an image with a text description.

    Args:
        image_path (str): Path to the image file.
        text_description (str): The description to compare against.
        threshold (float): Minimum similarity score to consider a match.

    Returns:
        dict: Similarity score and match result.
    """
    # Preprocess the image
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    # Tokenize the text
    text_tokens = clip.tokenize([text_description]).to(device)

    with torch.no_grad():
        # Get feature embeddings
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_tokens)

        # Compute similarity
        similarity = (text_features @ image_features.T).item()

    # Print results
    print(f"Text: {text_description}")
    print(f"Similarity Score: {similarity:.4f}")
    
    # Check if it meets the threshold
    match = similarity > threshold
    print("Match ✅" if match else "No Match ❌")

    return {"similarity_score": similarity, "match": match}

# Example Test Run
if __name__ == "__main__":
    image_path = input('Enter image file path: ')  # Replace with your image file
    text_description = "A bookshelf against a wall"  # Example description
    test_clip(image_path, text_description)

