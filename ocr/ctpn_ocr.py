# Install EasyOCR
# !pip install easyocr opencv-python matplotlib

# -*- coding:utf-8 -*-
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import easyocr
import json
import re
from symspellpy import SymSpell, Verbosity
from symspellpy.symspellpy import SuggestItem

from .ctpn_model import CTPN_Model
from .ctpn_utils import gen_anchor, bbox_transfor_inv, clip_box, filter_bbox, nms, TextProposalConnectorOriented
from .ctpn_utils import resize
from .config import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
reader = easyocr.Reader(['en'])

# Initialize symspellpy for spelling correction with in-memory dictionary
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

# Create in-memory dictionary with nutrition terms
nutrition_terms = [
    "calories", "total fat", "saturated fat", "trans fat", "cholesterol", 
    "sodium", "carbohydrate", "dietary fiber", "sugars", "protein", 
    "vitamin", "calcium", "iron", "potassium", "serving", "size", "amount",
    "daily value", "percent", "total", "nutrition facts", "servings", "container",
    "ingredients", "carbohydrates", "saturated", "polyunsaturated", "monounsaturated",
    "fat", "per", "cup", "tbsp", "tablespoon", "teaspoon", "gram", "milligram",
    "vitamin a", "vitamin b", "vitamin b1", "vitamin b2", "vitamin b3", "vitamin b6",
    "vitamin b12", "vitamin c", "vitamin d", "vitamin e", "vitamin k", "folate", "folic acid",
    "magnesium", "zinc", "copper", "selenium", "phosphorus", "manganese", "iodine",
    "kcal", "kj", "mg", "g", "oz", "ml", "l", "mcg", "iu", "serving size",
    "added sugar", "natural sugar", "sugar alcohol", "net carbs", "total carbs",
    "dietary cholesterol", "reference amount", "nutrition label", "ingredients list",
    "percent daily value", "serving per container", "serving suggestion",
    "milk", "bread", "cheese", "butter", "oil", "meat", "fish", "rice", "pasta", "egg",
    "low fat", "no sugar", "gluten free", "organic", "non-gmo", "high protein", "low sodium"
]

# Add terms directly to the dictionary
for term in nutrition_terms:
    sym_spell.create_dictionary_entry(term, 1000000)  # High frequency to prioritize these terms

def detect_text(image):
    """
    Detect text regions in an image
    
    Args:
        image: Image array (numpy array)
        
    Returns:
        tuple: (image, text_boxes)
    """

    model = CTPN_Model()
    weights = './ocr/checkpoints/pretrained_wgts.tar'
    
    # Check if weights file exists
    if not os.path.exists(weights):
        print(f"Weight file not found at: {weights}")
        # Fallback to EasyOCR if weights not found
        results = reader.readtext(image)
        boxes = []
        for (bbox, text, prob) in results:
            # Convert EasyOCR bbox format to the expected format
            x1, y1 = bbox[0]
            x2, y2 = bbox[2]
            # Format: x1, y1, x2, y2, score, x1, y1, x2, y2
            box = [x1, y1, x2, y2, prob, x1, y1, x2, y2]
            boxes.append(box)
        return image, np.array(boxes) if boxes else np.array([[0, 0, 10, 10, 0.5, 0, 0, 10, 10]])
    
    model.load_state_dict(torch.load(weights, map_location=device)['model_state_dict'])
    model.to(device)
    model.eval()

    image = resize(image, width=600)
    image_c = image.copy()
    h, w = image.shape[:2]

    image = image.astype(np.float32) - IMAGE_MEAN
    image = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    with torch.no_grad():
        cls, regr = model(image)
        cls_prob = F.softmax(cls, dim=-1).cpu().numpy()
        regr = regr.cpu().numpy()

    anchor = gen_anchor((int(h / 16), int(w / 16)), 16)
    bbox = bbox_transfor_inv(anchor, regr)
    bbox = clip_box(bbox, [h, w])

    fg = np.where(cls_prob[0, :, 1] > 0.5)[0]
    select_anchor = bbox[fg, :]
    select_score = cls_prob[0, fg, 1]
    keep_index = filter_bbox(select_anchor.astype(np.int32), 16)

    select_anchor = select_anchor[keep_index]
    select_score = select_score[keep_index]
    nmsbox = np.hstack((select_anchor, np.reshape(select_score, (-1, 1))))
    keep = nms(nmsbox, 0.3)

    select_anchor = select_anchor[keep]
    select_score = select_score[keep]

    textConn = TextProposalConnectorOriented()
    text_boxes = textConn.get_text_lines(select_anchor.astype(np.int32), select_score, [h, w])
    return image_c, text_boxes

def recognize_text(image, boxes):
    """
    Recognize text in detected regions
    
    Args:
        image: Image array
        boxes: Detected text boxes
        
    Returns:
        list: List of dictionaries with recognized text
    """
    results = []
    for box in boxes:
        try:
            pts = np.array(box[:8], dtype=np.int32).reshape(4, 2)
            
            # Get the min and max coordinates
            x_coords = pts[:, 0]
            y_coords = pts[:, 1]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            # Ensure coordinates are within image boundaries
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(image.shape[1], x_max)
            y_max = min(image.shape[0], y_max)
            
            # Check if crop dimensions are valid
            if x_min >= x_max or y_min >= y_max:
                print(f"Invalid crop dimensions: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}")
                continue  # Skip invalid crop
                
            # Check if crop area is too small
            if (x_max - x_min) < 5 or (y_max - y_min) < 5:
                print(f"Crop area is too small: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}")
                continue  # Skip crops that are too small
                
            cropped = image[y_min:y_max, x_min:x_max]
            
            # Check if cropped image is valid
            if cropped is None or cropped.size == 0 or cropped.shape[0] == 0 or cropped.shape[1] == 0:
                print(f"Invalid cropped image: {cropped}")
                continue  # Skip empty crops
                
            ocr_out = reader.readtext(cropped)

            if ocr_out:
                _, text, conf = ocr_out[0]
                results.append({
                    'box': pts.tolist(),
                    'text': text,
                    'confidence': round(conf, 4)
                })
        except Exception as e:
            print(f"Error processing box: {str(e)}")
            continue
            
    return results

def correct_spelling(text):
    """Correct spelling in the extracted text using symspellpy"""
    lines = text.split('\n')
    corrected_lines = []
    
    for line in lines:
        if not line.strip():
            continue
            
        # Preserve numbers and units, only correct words
        parts = re.findall(r'(\d+(?:\.\d+)?)|([a-zA-Z]+)|([^a-zA-Z0-9\s])', line)
        corrected_line = ""
        
        for part in parts:
            if part[0]:  # Numeric part
                corrected_line += part[0] + " "
            elif part[1]:  # Word part
                suggestions = sym_spell.lookup(part[1].lower(), Verbosity.CLOSEST, max_edit_distance=2)
                if suggestions:
                    corrected_line += suggestions[0].term + " "
                else:
                    corrected_line += part[1] + " "
            elif part[2]:  # Symbol part
                corrected_line += part[2] + " "
                
        corrected_lines.append(corrected_line.strip())
        
    return '\n'.join(corrected_lines)

def extract_nutrition_dict(text):
    """Extract nutrition information into a structured dictionary"""
    nutrition_dict = {}
    
    # Common nutrition labels and their variations
    nutrition_patterns = {
        'calories': [r'calories'],
        'total_fat': [r'total\s*fat'],
        'saturated_fat': [r'saturated\s*fat'],
        'trans_fat': [r'trans\s*fat'],
        'cholesterol': [r'cholesterol'],
        'sodium': [r'sodium'],
        'total_carbs': [r'total\s*carbohydrates?', r'carbs?'],
        'dietary_fiber': [r'dietary\s*fiber'],
        'sugars': [r'sugars?'],
        'protein': [r'protein'],
        'vitamin_d': [r'vitamin\s*d'],
        'calcium': [r'calcium'],
        'iron': [r'iron'],
        'potassium': [r'potassium']
    }
    
    lines = text.split('\n')
    
    for line in lines:
        line = line.lower().strip()
        if not line:
            continue
            
        # Try to identify nutritional information in each line
        for nutrient, patterns in nutrition_patterns.items():
            for pattern in patterns:
                match = re.search(f'({pattern})[:\s]*(\d+(?:\.\d+)?)(\s*\w+)?', line, re.IGNORECASE)
                if match:
                    amount = match.group(2)
                    unit = match.group(3).strip() if match.group(3) else ""
                    nutrition_dict[nutrient] = (amount, unit)
                    break
    
    return nutrition_dict

def extract_nutrition_text(image):
    """
    Extract nutrition text from an image
    
    Args:
        image: Image array (numpy ndarray) or path to image file (str)
        
    Returns:
        tuple: (raw_text, corrected_text, nutrition_dict)
    """
    try:
        # Detect text regions
        image, boxes = detect_text(image)
        
        # Recognize text in the detected regions
        ocr_results = recognize_text(image, boxes)
        
        # Combine all text into a single string
        raw_text = "\n".join([result['text'] for result in ocr_results])
        
        # Skip empty results
        if not raw_text:
            return "No text detected", "", {}
        
        # Correct spelling
        corrected_text = correct_spelling(raw_text)
        
        # Extract structured nutrition information
        nutrition_dict = extract_nutrition_dict(corrected_text)
        
        return raw_text, corrected_text, nutrition_dict
    
    except Exception as e:
        return f"Error processing image: {str(e)}", "", {}