from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import torch
import clip
from PIL import Image

app = Flask(__name__)
CORS(app)

# --- Global Model Loading ---
print("Loading models...")
# Use GPU if available; otherwise, fall back to CPU.
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

# If using GPU, convert model to half precision for speed and lower memory footprint.
if device == "cuda":
    model.half()

print("Model and Preprocess instance are loaded")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    # Validate the incoming request.
    if "image" not in request.files or "description" not in request.form:
        return jsonify({"error": "Image and description are required"}), 400

    image_file = request.files["image"]
    text = request.form.get("description")
    
    if not image_file or image_file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Open the image directly from the uploaded file.
    try:
        image = Image.open(image_file)
    except Exception as e:
        return jsonify({"error": f"Invalid image file: {str(e)}"}), 400

    # Preprocess the image and move it to the selected device.
    image_input = preprocess(image).unsqueeze(0).to(device)
    if device == "cuda":
        image_input = image_input.half()

    # Tokenize the text and move it to the device.
    text_tokens = clip.tokenize([text]).to(device)

    with torch.no_grad():
        # Obtain feature embeddings.
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_tokens)
        similarity = (text_features @ image_features.T).squeeze().item()

    # Allow threshold override via request; default is 25.
    try:
        threshold = float(request.form.get("threshold", 25))
    except ValueError:
        threshold = 25

    result = {
        "score": similarity,
        "match": similarity >= threshold,
        "message": "Processing complete"
    }

    return jsonify(result)

if __name__ == "__main__":
    # For development only; disable debug mode in production.
    app.run(port=9090, debug=False)
