from flask import Flask,jsonify,request,render_template
import torch
import clip,io
from PIL import Image

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/process", methods=["POST"])
def process(threshold=25):
    if "image" not in request.files or "description" not in request.form:
        return jsonify({"error": "Image and description are required"}), 400
    
    image_file = request.files.get('image')
    text = request.form.get('description')
    print(text,image_file)

    if image_file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Load image into memory using PIL
    image = preprocess(Image.open(io.BytesIO(image_file.read()))).unsqueeze(0).to(device)

    # Tokenize the text
    text_tokens = clip.tokenize([text]).to(device)

    with torch.no_grad():
        # Get feature embeddings
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_tokens)

        # Compute similarity
        similarity = (text_features @ image_features.T).item()

    processed_data = {
        "score": similarity,
        "match":similarity>=threshold,
        "message": "Processing complete"
    }

    return jsonify(processed_data)


if __name__=='__main__':
    print('Loading models...')
    device = "cpu"

    model, preprocess = clip.load("ViT-B/32", device=device)  # Ensure same model variant

    print('Model and Preprocess instance are loaded')

    app.run(port=9090,debug=True)