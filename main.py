import os
from flask import Flask, request, send_file
from werkzeug.utils import secure_filename
import cv2
from ultralytics import YOLO
import numpy as np
import io
import base64
from ocr.ctpn_ocr import extract_nutrition_text
from llm.llm_vision import extract_nutrition_from_image
from PIL import Image
import datetime

app = Flask(__name__)

# Load YOLO model
model = YOLO("nutrition_label_detector.pt")

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    # Extract coordinates
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate area of each box
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # Calculate coordinates of intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    # Check if there is an intersection
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    # Calculate intersection area
    intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union area
    union_area = area1 + area2 - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area
    
    return iou

def save_results_to_file(original_filename, raw_text, corrected_text, nutrition_dict, llm_output):
    """
    Save the tabs' content to a text file in the results folder.
    
    Args:
        original_filename: Original filename of the uploaded image
        raw_text: Raw OCR text
        corrected_text: Corrected OCR text
        nutrition_dict: Dictionary of nutrition values from OCR
        llm_output: Dictionary of nutrition values from LLM
    """
    # Create results directory if it doesn't exist
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Get base filename without extension and sanitize it
    base_filename = os.path.splitext(os.path.basename(original_filename))[0]
    base_filename = secure_filename(base_filename)
    
    # Create a timestamp for uniqueness
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output filename
    output_filename = f"{base_filename}_{timestamp}.txt"
    output_path = os.path.join(results_dir, output_filename)
    
    # Format structured data
    structured_data = ""
    for nutrient, (amount, unit) in nutrition_dict.items():
        display_name = " ".join(word.capitalize() for word in nutrient.split("_"))
        structured_data += f"{display_name}: {amount} {unit}\n"
    
    # Format LLM output
    llm_data = ""
    if llm_output:
        # Handle different possible formats of LLM output
        if isinstance(next(iter(llm_output.values()), None), tuple):
            # Format: {nutrient: (amount, unit)}
            for nutrient, (amount, unit) in llm_output.items():
                display_name = " ".join(word.capitalize() for word in nutrient.split("_"))
                llm_data += f"{display_name}: {amount} {unit}\n"
        else:
            # Format: {nutrient: value}
            for nutrient, value in llm_output.items():
                display_name = " ".join(word.capitalize() for word in nutrient.split("_"))
                llm_data += f"{display_name}: {value}\n"
    
    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=== NUTRITION LABEL RESULTS ===\n\n")
        
        f.write("--- STRUCTURED DATA ---\n")
        f.write(structured_data if structured_data else "No structured data available.\n")
        f.write("\n")
        
        f.write("--- LLM OUTPUT ---\n")
        f.write(llm_data if llm_data else "No LLM output available.\n")
        f.write("\n")
        
        f.write("--- CORRECTED TEXT ---\n")
        f.write(corrected_text + "\n\n")
        
        f.write("--- RAW TEXT ---\n")
        f.write(raw_text + "\n")
    
    print(f"Results saved to {output_path}")

@app.route("/")
def index():
    """Home page with file upload form."""
    return """
    <!doctype html>
    <html>
      <head>
        <title>Nutrition Label Detector</title>
        <style>
          body { font-family: Arial, sans-serif; margin: 20px; }
          .container { max-width: 600px; margin: 0 auto; }
          h1 { color: #333; }
          input[type=file] { margin: 10px 0; }
          input[type=submit] { background-color: #4CAF50; color: white; padding: 10px 15px; border: none; cursor: pointer; }
        </style>
      </head>
      <body>
        <div class="container">
          <h1>Nutrition Label Detector</h1>
          <p>Upload an image to detect and crop nutrition labels.</p>
          <form action="/detect" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*">
            <br>
            <input type="submit" value="Detect Labels">
          </form>
        </div>
      </body>
    </html>
    """


@app.route("/detect", methods=["POST"])
def detect():
    """Process uploaded image using YOLO and return results."""
    if "file" not in request.files:
        return "No file part", 400

    file = request.files["file"]
    if file.filename == "":
        return "No selected file", 400

    # Read image file directly into memory
    file_bytes = file.read()
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Perform detection with explicit confidence threshold
    results = model(img, conf=0.5)  # Set confidence threshold to 0.5 for consistency across platforms

    # Log detection results
    print(f"Image size: {img.shape}")
    print(f"Number of detection results: {len(results)}")
    
    # Process results and create in-memory crops
    crops = []
    raw_texts = []
    corrected_texts = []
    nutrition_dicts = []
    llm_outputs = []

    for r_idx, r in enumerate(results):
        boxes = r.boxes
        print(f"Result {r_idx+1}: Found {len(boxes)} boxes")
        
        # Process each detected box directly
        for box_idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = float(box.conf[0])
            print(f"  Box {box_idx+1}: x1={x1}, y1={y1}, x2={x2}, y2={y2}, conf={conf:.4f}")
            
            # Print bounds info for debugging
            h, w = img.shape[:2]
            print(f"Image dimensions: {w}x{h}")
            
            # Ensure coordinates are within image boundaries
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            
            print(f"Box after validation: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            
            # Skip if dimensions are invalid
            if x1 >= x2 or y1 >= y2:
                print(f"Invalid box dimensions: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                continue
            
            # Extract crop
            try:
                print(f"Extracting crop from coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                crop = img[y1:y2, x1:x2]
                print(f"Crop extracted successfully, shape: {crop.shape}, dtype: {crop.dtype}")
                
                print(f"Crop statistics - min: {crop.min()}, max: {crop.max()}, mean: {crop.mean():.2f}")
                
                # Skip invalid crops
                if crop is None or crop.size == 0:
                    print("⚠️ Skipping: Crop is None or empty")
                    continue
                
                # Extract nutrition text using OCR with the crop image array
                print("Passing crop to OCR engine...")
                raw_text, corrected_text, nutrition_dict = extract_nutrition_text(crop)
                print(f"OCR completed. Text length: {len(raw_text)}, nutrition items: {len(nutrition_dict)}")
                
                # If OCR failed with error message
                if raw_text.startswith("Error"):
                    print(f"⚠️ OCR Error: {raw_text}")
                    
                raw_texts.append(raw_text)
                corrected_texts.append(corrected_text)
                nutrition_dicts.append(nutrition_dict)
                
                # Convert OpenCV crop (BGR) to PIL Image (RGB) and pass to LLM
                print("Converting crop to PIL Image for LLM...")
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                pil_crop = Image.fromarray(crop_rgb)
                print(f"PIL Image created successfully, size: {pil_crop.size}, mode: {pil_crop.mode}")
                
                llm_result = extract_nutrition_from_image(pil_crop)
                print(f"LLM processing completed. Result items: {len(llm_result) if llm_result else 0}")
                
                llm_outputs.append(llm_result if llm_result else {})
                
                # Save results to file
                save_results_to_file(
                    file.filename,
                    raw_text,
                    corrected_text,
                    nutrition_dict,
                    llm_result if llm_result else {}
                )
                
                # Convert crop to base64 for embedding in HTML
                _, buffer = cv2.imencode(".jpg", crop)
                crop_base64 = base64.b64encode(buffer).decode("utf-8")
                crops.append(crop_base64)
                print(f"Crop {len(crops)} processed successfully ✓")
                
            except Exception as e:
                import traceback
                print(f"❌ Error processing crop: {str(e)}")
                print(traceback.format_exc())
                continue

    # Return results page with embedded images
    if crops:
        result_html = """
        <!doctype html>
        <html>
          <head>
            <title>Nutrition Label Detection Results</title>
            <style>
              body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
              .container { max-width: 1000px; margin: 0 auto; }
              h1 { color: #333; text-align: center; margin-bottom: 30px; }
              .results { display: flex; flex-wrap: wrap; justify-content: space-between; }
              .result-item { margin: 15px 0; border: 1px solid #ddd; padding: 20px; border-radius: 8px; width: 100%; background-color: #fff; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
              .result-content { display: flex; flex-wrap: wrap; }
              .image-container { flex: 1; min-width: 250px; margin-right: 20px; }
              .text-container { flex: 2; min-width: 300px; }
              img { max-width: 100%; border: 1px solid #eee; }
              .back-btn { margin-top: 30px; text-align: center; }
              a { text-decoration: none; color: #4CAF50; }
              a:hover { text-decoration: underline; }
              .download-btn { margin-top: 10px; }
              .text-content { margin-top: 10px; font-size: 0.9em; white-space: pre-line; }
              .nutrition-table { width: 100%; border-collapse: collapse; margin-top: 15px; }
              .nutrition-table th, .nutrition-table td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
              .nutrition-table th { background-color: #f2f2f2; }
              .tabs { display: flex; margin-bottom: 15px; }
              .tab { padding: 8px 15px; cursor: pointer; background-color: #eee; margin-right: 5px; border-radius: 4px 4px 0 0; }
              .tab.active { background-color: #4CAF50; color: white; }
              .tab-content { display: none; }
              .tab-content.active { display: block; }
            </style>
            <script>
              function switchTab(resultIndex, tabName) {
                // Hide all tab contents
                document.querySelectorAll(`.result-${resultIndex} .tab-content`).forEach(content => {
                  content.classList.remove('active');
                });
                
                // Deactivate all tabs
                document.querySelectorAll(`.result-${resultIndex} .tab`).forEach(tab => {
                  tab.classList.remove('active');
                });
                
                // Activate selected tab and content
                document.getElementById(`${tabName}-${resultIndex}`).classList.add('active');
                document.getElementById(`${tabName}-content-${resultIndex}`).classList.add('active');
              }
            </script>
          </head>
          <body>
            <div class="container">
              <h1>Nutrition Label Detection Results</h1>
              <div class="results">
        """

        for i, (
            crop_base64,
            raw_text,
            corrected_text,
            nutrition_dict,
            llm_output,
        ) in enumerate(
            zip(crops, raw_texts, corrected_texts, nutrition_dicts, llm_outputs)
        ):
            # Create HTML for nutrition table
            nutrition_table = """
            <table class="nutrition-table">
                <tr>
                    <th>Nutrient</th>
                    <th>Amount</th>
                    <th>Unit</th>
                </tr>
            """

            for nutrient, (amount, unit) in nutrition_dict.items():
                # Convert keys like total_fat to "Total Fat" for display
                display_name = " ".join(
                    word.capitalize() for word in nutrient.split("_")
                )
                nutrition_table += f"""
                <tr>
                    <td>{display_name}</td>
                    <td>{amount}</td>
                    <td>{unit}</td>
                </tr>
                """

            nutrition_table += "</table>"

            # No nutrients detected message
            if not nutrition_dict:
                nutrition_table = (
                    "<p>No structured nutrition data could be extracted.</p>"
                )

            # LLM output table
            if llm_output:
                llm_table = """
                <table class="nutrition-table">
                    <tr>
                        <th>Nutrient</th>
                        <th>Amount</th>
                        <th>Unit</th>
                    </tr>
                """
                for nutrient, (amount, unit) in llm_output.items():
                    display_name = " ".join(
                        word.capitalize() for word in nutrient.split("_")
                    )
                    llm_table += f"""
                    <tr>
                        <td>{display_name}</td>
                        <td>{amount}</td>
                        <td>{unit}</td>
                    </tr>
                    """
                llm_table += "</table>"
            else:
                llm_table = "<p>No LLM output available.</p>"

            result_html += f"""
                <div class="result-item result-{i}">
                  <div class="result-content">
                    <div class="image-container">
                      <img src="data:image/jpeg;base64,{crop_base64}" alt="Detected label">
                      <div class="download-btn">
                        <a href="data:image/jpeg;base64,{crop_base64}" download="nutrition_label_{i}.jpg">Download Image</a>
                      </div>
                    </div>
                    <div class="text-container">
                      <div class="tabs">
                        <div class="tab active" id="structured-{i}" onclick="switchTab({i}, 'structured')">Structured Data</div>
                        <div class="tab" id="llm-{i}" onclick="switchTab({i}, 'llm')">LLM Output</div>
                        <div class="tab" id="corrected-{i}" onclick="switchTab({i}, 'corrected')">Corrected Text</div>
                        <div class="tab" id="raw-{i}" onclick="switchTab({i}, 'raw')">Raw Text</div>
                      </div>
                      
                      <div id="structured-content-{i}" class="tab-content active">
                        <h3>Structured Nutrition Data</h3>
                        {nutrition_table}
                      </div>
                      
                      <div id="llm-content-{i}" class="tab-content">
                        <h3>LLM Output</h3>
                        {llm_table}
                      </div>
                      
                      <div id="corrected-content-{i}" class="tab-content">
                        <h3>Corrected Text</h3>
                        <pre class="text-content">{corrected_text}</pre>
                      </div>
                      
                      <div id="raw-content-{i}" class="tab-content">
                        <h3>Raw OCR Text</h3>
                        <pre class="text-content">{raw_text}</pre>
                      </div>
                    </div>
                  </div>
                </div>
            """

        result_html += """
              </div>
              <div class="back-btn">
                <a href="/">← Back to upload</a>
              </div>
            </div>
          </body>
        </html>
        """
        return result_html
    else:
        return (
            "No nutrition labels detected in the image. <a href='/'>Try again</a>",
            404,
        )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
