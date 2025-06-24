import os
from flask import Flask, request, render_template, jsonify
from PIL import Image # Python Imaging Library for image manipulation
import io # For working with streams of bytes
import base64 # For encoding/decoding base64 strings
import torch # PyTorch library for deep learning operations
import torchvision.transforms.functional as F # Functional API for torchvision transformations
import traceback # Module to print stack traces for debugging errors

# Import the Pix2Pix_Turbo model class from your local pix2pix_turbo.py file
from pix2pix_turbo import Pix2Pix_Turbo

# Initialize the Flask application
app = Flask(__name__)

# --- Model Loading ---
# Load the Pix2Pix_Turbo model when the application starts.
# This ensures the model is loaded only once and is ready for inference.
model = None # Initialize model as None to handle potential loading failures
try:
    # Instantiate the Pix2Pix_Turbo model.
    # The 'edge_to_image' type indicates the model expects edge maps as input
    # and generates full images. Other options like "sketch_to_image_stochastic"
    # might be available depending on the model's configuration.
    model = Pix2Pix_Turbo("edge_to_image") # Or "sketch_to_image_stochastic"
    print("Pix2Pix_Turbo model loaded successfully.")
except Exception as e:
    # Catch any exceptions that occur during model loading and print an error message.
    print(f"An error occurred while loading the model: {e}")
    model = None # Set model to None if loading fails

# Move the model to the GPU if CUDA is available for faster processing.
if torch.cuda.is_available():
    model.cuda()
    print("Model moved to GPU.")
else:
    print("GPU not found, model will run on CPU (performance may be slower).")

# --- Routes ---

@app.route('/')
def index():
    """
    Renders the main page of the application.
    This serves the index.html file, which contains the drawing canvas and UI.
    """
    return render_template('index.html')

@app.route('/generate_image', methods=['POST'])
def generate_image():
    """
    Handles POST requests to generate an image from a user's drawing and a text prompt.
    Receives base64 encoded image data and a text prompt from the frontend,
    processes it through the AI model, and returns the generated image as base64.
    """
    # Check if the model was loaded successfully. If not, return a server error.
    if not model:
        return jsonify({"error": "Model could not be loaded or an error occurred."}), 500

    try:
        # Parse the incoming JSON data from the request.
        data = request.json
        
        # Validate that essential data ('image' and 'prompt') are present in the request.
        if not data or 'image' not in data or 'prompt' not in data:
            return jsonify({"error": "Missing data: 'image' or 'prompt' not found."}), 400

        image_data = data['image'] # Get the base64 encoded drawing image
        prompt = data['prompt'] # Get the text prompt

        # Decode the base64 image data into a PIL Image object.
        # The split(',')[1] is used to remove the "data:image/png;base64," prefix.
        image_bytes = base64.b64decode(image_data.split(',')[1])
        input_image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB") # Open as PIL Image and ensure RGB format

        # Resize the input image to a target size suitable for the model.
        # It's best to keep it close to the size the model was trained on (e.g., 512x512).
        target_size = (512, 512) # Check the exact training size of your model if necessary
        input_image_pil = input_image_pil.resize(target_size)

        # IMPORTANT NOTE ON INPUT IMAGE:
        # If your Pix2Pix_Turbo model is specifically trained for "edge_to_image"
        # or "sketch_to_image_stochastic", it expects a *condition image*
        # (e.g., a Canny edge map or a sketch line drawing) as input,
        # NOT a direct RGB drawing from the user.
        #
        # Currently, 'input_image_pil' is the user's raw RGB drawing.
        # If your model expects Canny edges or another processed input,
        # you MUST apply that transformation here.
        #
        # Example for Canny Edge Detection (requires OpenCV: pip install opencv-python):
        # import cv2
        # import numpy as np
        # input_array = np.array(input_image_pil) # Convert PIL image to numpy array
        # canny_edges = cv2.Canny(input_array, 100, 200) # Apply Canny edge detection (adjust thresholds)
        # input_image_pil = Image.fromarray(canny_edges).convert("RGB") # Convert Canny output back to PIL RGB

        # Convert the PIL image to a PyTorch tensor.
        # ToTensor() converts pixels from [0, 255] to [0, 1] and changes (H, W, C) to (C, H, W).
        # unsqueeze(0) adds a batch dimension, making the tensor (1, C, H, W).
        # If your model expects input in the range [-1, 1], you would need to scale it:
        # c_t = c_t * 2 - 1
        c_t = F.to_tensor(input_image_pil).unsqueeze(0) 
        
        # Move the input tensor to the GPU if available.
        if torch.cuda.is_available():
            c_t = c_t.cuda()

        # Perform inference using the AI model within a torch.no_grad() context.
        # This disables gradient calculation, saving memory and speeding up inference.
        with torch.no_grad():
            output_image_tensor = model(c_t, prompt)
            # Convert the output tensor back to a PIL Image.
            # The model's output is typically in [-1, 1], so scale it back to [0, 1] for image display.
            output_pil = F.to_pil_image(output_image_tensor[0].cpu() * 0.5 + 0.5)

        # Encode the generated PIL image into a base64 string for sending back to the frontend.
        buffered = io.BytesIO()
        output_pil.save(buffered, format="PNG") # Save image to a byte buffer as PNG
        encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8') # Base64 encode and decode to UTF-8 string

        # Return the encoded image as a JSON response.
        return jsonify({"image": encoded_image})

    except Exception as e:
        # Log the full traceback of any error that occurs during image generation.
        print(f"An error occurred during image generation: {e}")
        traceback.print_exc() # Prints the detailed error trace to the console

        # Return a more descriptive error message to the client.
        return jsonify({"error": f"A server error occurred during image generation: {str(e)}"}), 500

# --- Application Entry Point ---
if __name__ == '__main__':
    # Run the Flask application in debug mode.
    # In debug mode, the server will reload on code changes and provide detailed error messages.
    # Set debug=False for production deployment.
    app.run(debug=True)
