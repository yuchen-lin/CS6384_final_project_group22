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

app = Flask(__name__)

# Load YOLO model
model = YOLO("nutrition_label_detector.pt")


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

    # Perform detection
    results = model(img)

    # Process results and create in-memory crops
    crops = []
    raw_texts = []
    corrected_texts = []
    nutrition_dicts = []
    llm_outputs = []

    for r in results:
        boxes = r.boxes

        for i, box in enumerate(boxes):
            # Get coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Extract crop
            crop = img[y1:y2, x1:x2]

            # Extract nutrition text using OCR with the crop image array
            raw_text, corrected_text, nutrition_dict = extract_nutrition_text(crop)
            raw_texts.append(raw_text)
            corrected_texts.append(corrected_text)
            nutrition_dicts.append(nutrition_dict)

            # Convert OpenCV crop (BGR) to PIL Image (RGB) and pass to LLM
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_crop = Image.fromarray(crop_rgb)
            llm_result = extract_nutrition_from_image(pil_crop)
            llm_outputs.append(llm_result if llm_result else {})

            # Convert crop to base64 for embedding in HTML
            _, buffer = cv2.imencode(".jpg", crop)
            crop_base64 = base64.b64encode(buffer).decode("utf-8")
            crops.append(crop_base64)

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
