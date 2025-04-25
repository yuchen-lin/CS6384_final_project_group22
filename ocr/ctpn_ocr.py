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
                    'confidence': round(conf, 4),
                    'x_min': x_min,
                    'x_max': x_max,
                    'y_min': y_min,
                    'y_max': y_max,
                    # Calculate center point for easier spatial reasoning
                    'center_x': (x_min + x_max) // 2,
                    'center_y': (y_min + y_max) // 2
                })
        except Exception as e:
            print(f"Error processing box: {str(e)}")
            continue
            
    return results

def correct_spelling(text):
    """Correct spelling in the extracted text using symspellpy"""
    lines = text.split('\n')
    corrected_lines = []
    
    # First, preprocess to correct common OCR errors in nutrition labels
    for i, line in enumerate(lines):
        if not line.strip():
            continue
            
        # Fix common OCR misreadings in nutrition values
        # Replace 'Og', 'Omg', etc. with '0g', '0mg' (letter O to number 0)
        line = re.sub(r'(\s|^)O([gm])', r'\1 0\2', line)
        line = re.sub(r'(\s|^)O(\s*%)', r'\1 0\2', line)
        
        # Fix specific measurement unit issues
        line = re.sub(r'(\d+)(\s*)Omg', r'\1\2 0mg', line)
        line = re.sub(r'(\d+)(\s*)O(\s*g)', r'\1\2 0\3', line)
        
        # First ensure consistent spacing between numbers and units by adding a space
        line = re.sub(r'(\d+)([gm])', r'\1 \2', line)
        
        # Preserve common nutrition label units that might get corrected incorrectly
        line = re.sub(r'(\d+)\s*([gm])$', r'\1 \2', line)
        lines[i] = line
    
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
                # Don't "correct" common measurement units
                if part[1].lower() in ['g', 'mg', 'mcg', 'iu', 'oz', 'ml']:
                    corrected_line += part[1] + " "
                else:
                    suggestions = sym_spell.lookup(part[1].lower(), Verbosity.CLOSEST, max_edit_distance=2)
                    if suggestions:
                        corrected_line += suggestions[0].term + " "
                    else:
                        corrected_line += part[1] + " "
            elif part[2]:  # Symbol part
                corrected_line += part[2] + " "
        
        # FIXED: Apply consistent spacing for all measurement units
        # For 'g' units - keep the space for consistency with 'mg'
        # Comment out the line that removes spaces between numbers and units
        # corrected_line = re.sub(r'(\d+)\s+([gm])([^a-zA-Z]|$)', r'\1\2\3', corrected_line)
        
        corrected_lines.append(corrected_line.strip())
        
    return '\n'.join(corrected_lines)

def extract_nutrition_dict(text):
    """Extract nutrition information into a structured dictionary"""
    nutrition_dict = {}
    
    print("\n===== NUTRITION TEXT PARSING DETAILS =====")
    
    # Common nutrition labels and their variations
    nutrition_patterns = {
        'calories': [r'calories', r'energy', r'kcal', r'cal(\.|s)?'],
        'total_fat': [r'total\s*fat', r'fat,?\s*total', r'fat\s*content', r't\.?\s*fat'],
        'saturated_fat': [r'saturated\s*fat', r'sat\.?\s*fat', r'sat\s*fat', r'saturates'],
        'trans_fat': [r'trans\s*fat', r'trans-fat', r'trans\s*fatty\s*acid'],
        'cholesterol': [r'cholesterol', r'cholest\.?'],
        'sodium': [r'sodium', r'salt', r'\bna\b'],
        'total_carbs': [r'total\s*carbohydrates?', r'carbs?', r'carbohydrates?,?\s*total', r'total\s*carbs?', r't\.?\s*carbs?'],
        'dietary_fiber': [r'dietary\s*fiber', r'fiber,?\s*dietary', r'fibre', r'fiber(\s|$)', r'roughage'],
        'sugars': [r'sugars?', r'total\s*sugars?', r'added\s*sugars?'],
        'protein': [r'protein', r'proteins'],
        'vitamin_d': [r'vitamin\s*d', r'vit\.?\s*d'],
        'calcium': [r'calcium', r'\bca\b'],
        'iron': [r'iron', r'\bfe\b'],
        'potassium': [r'potassium', r'\bk\b'],
        'vitamin_a': [r'vitamin\s*a', r'vit\.?\s*a'],
        'vitamin_c': [r'vitamin\s*c', r'vit\.?\s*c', r'ascorbic\s*acid'],
        'vitamin_e': [r'vitamin\s*e', r'vit\.?\s*e'],
        'thiamin': [r'thiamin', r'vitamin\s*b1', r'vit\.?\s*b1', r'thiamine'],
        'riboflavin': [r'riboflavin', r'vitamin\s*b2', r'vit\.?\s*b2'],
        'niacin': [r'niacin', r'vitamin\s*b3', r'vit\.?\s*b3'],
        'vitamin_b6': [r'vitamin\s*b6', r'vit\.?\s*b6', r'pyridoxine'],
        'folate': [r'folate', r'folic\s*acid', r'vitamin\s*b9', r'vit\.?\s*b9'],
        'vitamin_b12': [r'vitamin\s*b12', r'vit\.?\s*b12', r'cobalamin'],
        'biotin': [r'biotin', r'vitamin\s*b7', r'vit\.?\s*b7'],
        'pantothenic_acid': [r'pantothenic\s*acid', r'vitamin\s*b5', r'vit\.?\s*b5', r'pantothenate'],
        'phosphorus': [r'phosphorus', r'phosphorous', r'\bp\b'],
        'iodine': [r'iodine'],  # Removed \bi\b to avoid false matches
        'magnesium': [r'magnesium'],  # Removed \bmg\b to avoid confusion with the unit 'mg'
        'zinc': [r'zinc', r'\bzn\b'],
        'selenium': [r'selenium', r'\bse\b'],
        'copper': [r'copper', r'\bcu\b'],
        'manganese': [r'manganese', r'\bmn\b'],
        'chromium': [r'chromium', r'\bcr\b'],
        'molybdenum': [r'molybdenum', r'\bmo\b'],
        'chloride': [r'chloride', r'\bcl\b'],
        'choline': [r'choline'],
        'serving_size': [r'serving\s*size', r'portion\s*size', r'serving', r'portion'],
        'servings_per_container': [r'servings?\s*per\s*container', r'portions?\s*per\s*container', r'servings?\s*per\s*pack(age)?'],
        'calories_from_fat': [r'calories\s*from\s*fat', r'cal\.?\s*from\s*fat']
    }
    
    # Pre-processing for common OCR misreadings specific to values
    processed_text = text
    # Replace all instances of 'O' followed by g, mg, etc. with '0'
    processed_text = re.sub(r'(\s|^)O([gm%])', r'\1 0\2', processed_text)
    processed_text = re.sub(r'(\d+)(\s*)Omg', r'\1\2 0mg', processed_text)
    
    print("\nInput text for nutrition extraction:")
    print("-" * 40)
    print(processed_text)
    print("-" * 40)
    
    lines = processed_text.split('\n')
    
    print("\nParsing each line for nutrition information:")
    print("-" * 40)
    
    for line_idx, line in enumerate(lines):
        original_line = line
        line = line.lower().strip()
        if not line:
            print(f"Line {line_idx+1}: [Empty line, skipping]")
            continue
        
        print(f"Line {line_idx+1}: '{original_line}'")
        nutrient_match_found = False
        all_matches = []
            
        # Try to identify nutritional information in each line
        for nutrient, patterns in nutrition_patterns.items():
            # Sort patterns by length (descending) to prioritize more specific patterns
            sorted_patterns = sorted(patterns, key=len, reverse=True)
            
            for pattern in sorted_patterns:
                # More flexible matching for values, handling both O and 0
                match = re.search(r'(' + pattern + r')[:\s]*([O0-9]+(?:\.[O0-9]+)?)(\s*\w+)?', line, re.IGNORECASE)
                if match:
                    # Store all matches to pick the best one
                    match_length = len(match.group(1))
                    all_matches.append((nutrient, pattern, match, match_length))
        
        # Sort matches by length of the matched pattern (descending)
        if all_matches:
            # Pick the longest match (most specific)
            all_matches.sort(key=lambda x: x[3], reverse=True)
            nutrient, pattern, match, _ = all_matches[0]
            
            nutrient_match_found = True
            print(f"  ✓ Matched '{nutrient}' with pattern '{pattern}'")
            
            # Log the groups captured
            print(f"    - Captured groups: {match.group(1)}, {match.group(2)}, {match.group(3) if match.group(3) else 'None'}")
            
            # Convert any 'O' in the value to '0'
            amount_str = match.group(2)
            # Add null check to prevent TypeError on macOS
            if amount_str is None:
                print(f"  ⚠️ Warning: Matched pattern '{pattern}' but amount is None in line: '{line}'")
                continue
            
            original_amount = amount_str    
            amount = re.sub(r'O', '0', amount_str)
            if original_amount != amount:
                print(f"    - Converted 'O' to '0' in amount: '{original_amount}' → '{amount}'")
            
            # Get the unit and ensure it's parsed correctly
            unit_str = match.group(3) if match.group(3) else ""
            unit = unit_str.strip()
            
            # Ensure common units are preserved
            if not unit and (amount.endswith('g') or amount.endswith('mg')):
                # Handle cases where the unit is attached to the number
                if amount.endswith('g'):
                    original_amount = amount
                    unit = 'g'
                    amount = amount[:-1]
                    print(f"    - Extracted unit 'g' from amount: '{original_amount}' → amount='{amount}', unit='{unit}'")
                elif amount.endswith('mg'):
                    original_amount = amount
                    unit = 'mg'
                    amount = amount[:-2]
                    print(f"    - Extracted unit 'mg' from amount: '{original_amount}' → amount='{amount}', unit='{unit}'")
            
            nutrition_dict[nutrient] = (amount, unit)
            print(f"  📊 Added to nutrition dict: {nutrient} = ({amount}, '{unit}')")
                
        if not nutrient_match_found:
            print(f"  ✗ No nutrition information found in this line")
    
    print("\n===== EXTRACTION SUMMARY =====")
    print(f"Extracted {len(nutrition_dict)} nutrition items:")
    for nutrient, (amount, unit) in nutrition_dict.items():
        print(f"  • {nutrient}: {amount} {unit}")
    print("==============================\n")
    
    return nutrition_dict

def extract_nutrition_text(image):
    """
    Extract nutrition text from an image
    
    Args:
        image: Image array (numpy ndarray)
        
    Returns:
        tuple: (raw_text, corrected_text, nutrition_dict)
    """
    try:
        # Add logging to diagnose the issue
        print(f"Image type: {type(image)}")
        if image is None:
            print("ERROR: Image is None")
            return "Error: No image data provided", "", {}
        
        if isinstance(image, np.ndarray):
            print(f"Image shape: {image.shape}, Image size: {image.size}")
        else:
            print(f"Image is not a numpy array but a {type(image)}")
        
        # Detect text regions
        print("Starting text detection...")
        image, boxes = detect_text(image)
        print(f"Text detection completed. Found {len(boxes)} boxes")
        
        # Recognize text in the detected regions
        print("Starting text recognition...")
        ocr_results = recognize_text(image, boxes)
        print(f"Text recognition completed. Recognized {len(ocr_results)} text regions")
        
        # Process OCR results spatially - keeping the original raw text for reference
        raw_text = "\n".join([result['text'] for result in ocr_results])
        
        # Skip empty results
        if not raw_text:
            print("No text was detected in the image")
            return "No text detected", "", {}
        
        print(f"Raw text extracted: {raw_text[:100]}...")
        
        # Correct spelling
        print("Starting spelling correction...")
        # Create a corrected version of each OCR result
        corrected_ocr_results = []
        for result in ocr_results:
            corrected_text = correct_spelling(result['text'])
            result_copy = result.copy()
            result_copy['corrected_text'] = corrected_text
            corrected_ocr_results.append(result_copy)
        
        # Create a full corrected text for display purposes
        corrected_text = "\n".join([result['corrected_text'] for result in corrected_ocr_results])
        print("Spelling correction completed")
        
        # Extract structured nutrition information using spatial awareness
        print("Extracting nutrition information using spatial awareness...")
        nutrition_dict = extract_nutrition_dict_spatial(corrected_ocr_results)
        print(f"Extracted {len(nutrition_dict)} nutrition items")
        
        return raw_text, corrected_text, nutrition_dict
    
    except Exception as e:
        import traceback
        print(f"ERROR in extract_nutrition_text: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return f"Error processing image: {str(e)}", "", {}

def extract_nutrition_dict_spatial(ocr_results):
    """Extract nutrition information using spatial relationships between text boxes
    
    Args:
        ocr_results: List of OCR results with position information
        
    Returns:
        dict: Dictionary of nutrition values
    """
    nutrition_dict = {}
    
    print("\n===== SPATIAL NUTRITION TEXT PARSING =====")
    
    # Common nutrition labels and their variations (same as before)
    nutrition_patterns = {
        'calories': [r'calories', r'energy', r'kcal', r'cal(\.|s)?'],
        'total_fat': [r'total\s*fat', r'fat,?\s*total', r'fat\s*content', r't\.?\s*fat'],
        'saturated_fat': [r'saturated\s*fat', r'sat\.?\s*fat', r'sat\s*fat', r'saturates'],
        'trans_fat': [r'trans\s*fat', r'trans-fat', r'trans\s*fatty\s*acid'],
        'cholesterol': [r'cholesterol', r'cholest\.?'],
        'sodium': [r'sodium', r'salt', r'\bna\b'],
        'total_carbs': [r'total\s*carbohydrates?', r'carbs?', r'carbohydrates?,?\s*total', r'total\s*carbs?', r't\.?\s*carbs?'],
        'dietary_fiber': [r'dietary\s*fiber', r'fiber,?\s*dietary', r'fibre', r'fiber(\s|$)', r'roughage'],
        'sugars': [r'sugars?', r'total\s*sugars?', r'added\s*sugars?'],
        'protein': [r'protein', r'proteins'],
        'vitamin_d': [r'vitamin\s*d', r'vit\.?\s*d'],
        'calcium': [r'calcium', r'\bca\b'],
        'iron': [r'iron', r'\bfe\b'],
        'potassium': [r'potassium', r'\bk\b'],
        'vitamin_a': [r'vitamin\s*a', r'vit\.?\s*a'],
        'vitamin_c': [r'vitamin\s*c', r'vit\.?\s*c', r'ascorbic\s*acid'],
        'vitamin_e': [r'vitamin\s*e', r'vit\.?\s*e'],
        'thiamin': [r'thiamin', r'vitamin\s*b1', r'vit\.?\s*b1', r'thiamine'],
        'riboflavin': [r'riboflavin', r'vitamin\s*b2', r'vit\.?\s*b2'],
        'niacin': [r'niacin', r'vitamin\s*b3', r'vit\.?\s*b3'],
        'vitamin_b6': [r'vitamin\s*b6', r'vit\.?\s*b6', r'pyridoxine'],
        'folate': [r'folate', r'folic\s*acid', r'vitamin\s*b9', r'vit\.?\s*b9'],
        'vitamin_b12': [r'vitamin\s*b12', r'vit\.?\s*b12', r'cobalamin'],
        'biotin': [r'biotin', r'vitamin\s*b7', r'vit\.?\s*b7'],
        'pantothenic_acid': [r'pantothenic\s*acid', r'vitamin\s*b5', r'vit\.?\s*b5', r'pantothenate'],
        'phosphorus': [r'phosphorus', r'phosphorous', r'\bp\b'],
        'iodine': [r'iodine'],  # Removed \bi\b to avoid false matches
        'magnesium': [r'magnesium'],  # Removed \bmg\b to avoid confusion with the unit 'mg'
        'zinc': [r'zinc', r'\bzn\b'],
        'selenium': [r'selenium', r'\bse\b'],
        'copper': [r'copper', r'\bcu\b'],
        'manganese': [r'manganese', r'\bmn\b'],
        'chromium': [r'chromium', r'\bcr\b'],
        'molybdenum': [r'molybdenum', r'\bmo\b'],
        'chloride': [r'chloride', r'\bcl\b'],
        'choline': [r'choline'],
        'serving_size': [r'serving\s*size', r'portion\s*size', r'serving', r'portion'],
        'servings_per_container': [r'servings?\s*per\s*container', r'portions?\s*per\s*container', r'servings?\s*per\s*pack(age)?'],
        'calories_from_fat': [r'calories\s*from\s*fat', r'cal\.?\s*from\s*fat']
    }
    
    # First, identify all text blocks that match nutrition labels
    nutrition_label_blocks = []
    
    print("\nIdentifying nutrition label blocks:")
    for i, result in enumerate(ocr_results):
        text = result['corrected_text'].lower().strip()
        if not text:
            continue
            
        print(f"OCR block {i+1}: '{result['corrected_text']}' at x={result['center_x']}, y={result['center_y']}")
        
        # Find matching nutrition patterns
        matched_nutrients = []
        for nutrient, patterns in nutrition_patterns.items():
            for pattern in sorted(patterns, key=len, reverse=True):
                match = re.search(r'\b' + pattern + r'\b', text, re.IGNORECASE)
                if match:
                    matched_nutrients.append((nutrient, match.group(), match.start(), match.end()))
                    break
        
        if matched_nutrients:
            for nutrient, matched_text, start, end in matched_nutrients:
                print(f"  ✓ Found nutrition label: '{nutrient}' ({matched_text})")
                nutrition_label_blocks.append({
                    'nutrient': nutrient,
                    'matched_text': matched_text,
                    'block_index': i,
                    'x': result['center_x'],
                    'y': result['center_y'],
                    'x_min': result['x_min'],
                    'x_max': result['x_max'],
                    'y_min': result['y_min'],
                    'y_max': result['y_max'],
                    'center_x': result['center_x'],  # Explicitly add center_x
                    'center_y': result['center_y']   # Explicitly add center_y
                })
    
    # Sort nutrition blocks by y-coordinate (top to bottom)
    nutrition_label_blocks.sort(key=lambda block: block['y'])
    
    # Now for each nutrition label, find the closest value to its right within a reasonable y-range
    print("\nMatching nutrition labels with values:")
    
    # Regular expression for numbers with optional units
    value_pattern = r'([O0-9]+(?:\.[O0-9]+)?)(\s*[a-zA-Z%]+)?'
    processed_blocks = set()  # Track which blocks we've used for values
    
    for label_block in nutrition_label_blocks:
        print(f"Finding value for '{label_block['nutrient']}' at x={label_block['x']}, y={label_block['y']}:")
        
        # First check if the nutrition label text itself contains a value
        block_idx = label_block['block_index']
        label_text = ocr_results[block_idx]['corrected_text'].lower().strip()
        
        # Look for value in the same text as the label, but only after the matched nutrition text
        matched_text = label_block['matched_text'].lower()
        text_after_match = label_text[label_text.find(matched_text) + len(matched_text):]
        
        # Only search for value pattern if there is text after the nutrition label
        value_in_label = None
        if text_after_match.strip():
            # First look for values with g or mg units
            g_value_match = re.search(r'([O0-9]+(?:\.[O0-9]+)?)\s*[gm]', text_after_match)
            if g_value_match:
                amount_str = g_value_match.group(1)
                # Determine the unit
                unit_match = re.search(r'([gm])', text_after_match[g_value_match.end()-1:g_value_match.end()])
                unit_str = unit_match.group(1) if unit_match else ""
                if unit_str == 'm':
                    # Check if it's actually 'mg'
                    if len(text_after_match) > g_value_match.end() and text_after_match[g_value_match.end():g_value_match.end()+1] == 'g':
                        unit_str = 'mg'
                
                # Convert 'O' to '0' in the amount
                amount = re.sub(r'O', '0', amount_str)
                
                value_in_label = g_value_match
            else:
                # If no g/mg value, look for any value
                value_in_label = re.search(value_pattern, text_after_match)
            
        # If value is found in the same text, use it and continue to next label
        if value_in_label:
            amount_str = value_in_label.group(1)
            
            # Convert 'O' to '0' in the amount
            amount = re.sub(r'O', '0', amount_str)
            
            # If we found g/mg value earlier, use that unit
            if 'unit_str' in locals() and unit_str:
                unit = unit_str
            else:
                # Otherwise extract unit from the match
                unit_str = value_in_label.group(2) if len(value_in_label.groups()) > 1 and value_in_label.group(2) else ""
                unit = unit_str.strip()
            
            print(f"  ✓ Found value '{amount} {unit}' in the same text as '{label_block['nutrient']}'")
            nutrition_dict[label_block['nutrient']] = (amount, unit)
            processed_blocks.add(block_idx)
            continue  # Skip to next nutrition label
        
        # If no value in the label text, look for blocks that are to the right and within a similar y-coordinate range
        y_tolerance = 20  # Allow 20px up or down
        potential_value_blocks = []
        
        for i, result in enumerate(ocr_results):
            # Skip if we've already used this block
            if i in processed_blocks:
                continue
                
            # Skip if this is the label block itself
            if i == label_block['block_index']:
                continue
                
            # Check if required keys exist
            if 'center_x' not in result or 'center_y' not in result or 'center_x' not in label_block or 'center_y' not in label_block:
                print(f"  ⚠️ Skipping block {i} due to missing center coordinates")
                continue
                
            # Check if block is to the right of the label
            if result['center_x'] <= label_block['x_max']:
                continue
                
            # Check if block is within a reasonable y-range
            if abs(result['center_y'] - label_block['center_y']) > y_tolerance:
                continue
                
            # Check if block contains a number pattern
            text = result['corrected_text'].lower().strip()
            
            # First look for values with g or mg units
            g_value_match = re.search(r'([O0-9]+(?:\.[O0-9]+)?)\s*[gm]', text)
            if g_value_match:
                amount_str = g_value_match.group(1)
                # Determine the unit
                unit_match = re.search(r'([gm])', text[g_value_match.end()-1:g_value_match.end()])
                unit_str = unit_match.group(1) if unit_match else ""
                if unit_str == 'm':
                    # Check if it's actually 'mg'
                    if len(text) > g_value_match.end() and text[g_value_match.end():g_value_match.end()+1] == 'g':
                        unit_str = 'mg'
                
                # Convert 'O' to '0' in the amount
                amount = re.sub(r'O', '0', amount_str)
                
                potential_value_blocks.append({
                    'block_index': i,
                    'distance': result['center_x'] - label_block['x_max'],  # horizontal distance
                    'amount': amount,
                    'unit': unit_str,
                    'text': text,
                    'priority': 1  # Higher priority for g/mg values
                })
                continue  # Skip checking for other patterns in this block
            
            # If no g/mg value, look for any value
            value_match = re.search(value_pattern, text)
            if value_match:
                amount_str = value_match.group(1)
                unit_str = value_match.group(2) if value_match.group(2) else ""
                
                # Convert 'O' to '0' in the amount
                amount = re.sub(r'O', '0', amount_str)
                unit = unit_str.strip()
                
                # Extract unit from amount if it's attached (like "5g" or "10mg")
                if not unit and (amount.endswith('g') or amount.endswith('mg')):
                    if amount.endswith('g'):
                        unit = 'g'
                        amount = amount[:-1]
                    elif amount.endswith('mg'):
                        unit = 'mg'
                        amount = amount[:-2]
                
                potential_value_blocks.append({
                    'block_index': i,
                    'distance': result['center_x'] - label_block['x_max'],  # horizontal distance
                    'amount': amount,
                    'unit': unit,
                    'text': text,
                    'priority': 2  # Lower priority for other values
                })
        
        # Sort by horizontal distance (closest first)
        potential_value_blocks.sort(key=lambda block: (block['priority'], block['distance']))
        
        if potential_value_blocks:
            best_match = potential_value_blocks[0]
            processed_blocks.add(best_match['block_index'])
            
            print(f"  ✓ Found value '{best_match['amount']} {best_match['unit']}' for '{label_block['nutrient']}'")
            print(f"    from text block: '{best_match['text']}' at distance {best_match['distance']}px")
            
            nutrition_dict[label_block['nutrient']] = (best_match['amount'], best_match['unit'])
        else:
            print(f"  ✗ No value found for '{label_block['nutrient']}'")
    
    print("\n===== SPATIAL EXTRACTION SUMMARY =====")
    print(f"Extracted {len(nutrition_dict)} nutrition items:")
    for nutrient, (amount, unit) in nutrition_dict.items():
        print(f"  • {nutrient}: {amount} {unit}")
    print("==============================\n")
    
    return nutrition_dict