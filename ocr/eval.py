
import numpy as np
import cv2
from .ctpn_ocr import extract_nutrition_text, detect_text
from sklearn.metrics import mean_squared_error

# Ground truth data
test_1 = {
    "gt_text_by_box" :["0 %", "0 %", "16 servings per container", "iron 0 mg", "0 %", "serving size", 
                  "dietary fiber 0 g", "trans fat 0 g", "calcium 0 mg", "cholesterol 0 mg",
                  "nutrition facts", "60", "0 %", "total carbohydrate 17 g", "0 %", "calories",
                  "sodium 0 mg", "total fat 0 mg", "potassium 0 mg", "vitamin d 0 mcg", "0 %",
                  "protein 0 g", "0 %", "amount per serving", "1 tbsp. (21g)", "6%",
                  "% daily value", "total sugars 17 g", "0 %", "0 %", "saturated fat 0 g", "34 %"],

    "gt_dict" : {
                "calories": ("60", ""),
                "total_fat": ("0", "g"),
                "saturated_fat": ("0", "g"),
                "trans_fat": ("0", "g"),
                "cholesterol": ("0", "mg"),
                "sodium": ("0", "mg"),
                "total_carbs": ("17", "g"),
                "dietary_fiber": ("0", "g"),
                "sugars": ("17", "g"),
                "protein": ("0", "g"),
                "vitamin_d": ("0", "mcg"),
                "calcium": ("0", "mg"),
                "iron": ("0", "mg"),
                "potassium": ("0", "mg")
            },

    "gt_boxes" : []
}

test_2 = {
    "gt_text_by_box" :["21 %", "","saturated fat 8 g", "total fat 16 g", "total carbohydrate 79 g", "", "1 mg 6 %",
                       "calcium 30 mg 2 %", "amount per serving", "vit. d", "sodium 1,860 mg", "", "1 package (120 g)",
                       "", "iron", "", "", "servings per container", "", "nutrition facts", "cholesterol 0 mg",
                       "", "4", "510", "protein 12 g", "dietary fiber 2 g", "calories", "includes 2 g added sugars 4 %",
                       "trans fat 0 g"],

    "gt_dict" : {
                "calories": ("510", ""),
                "total_fat": ("16", "g"),
                "saturated_fat": ("8", "g"),
                "trans_fat": ("0", "g"),
                "cholesterol": ("0", "mg"),
                "sodium": ("1860", "mg"),
                "total_carbs": ("79", "g"),
                "dietary_fiber": ("2", "g"),
                "sugars": ("4", "g"),
                "protein": ("12", "g"),
                "vitamin_d": ("0", "mcg"),
                "calcium": ("30", "mg"),
                "iron": ("1", "mg"),
                "potassium": ("250", "mg")
            },

    "gt_boxes" : []
}

test_3 = {
    "gt_text_by_box" :["nutrition", "6 %", "2 %", "7 %", "0 %", "", "",
                        "% daily value*/% valor diario*", "0 %", "serving size/tamano por racion",
                        "total sugars/azucares totales < 1 g", "amount per serving/contidad por racion", "", 
                        "dietary fiber/fibra dietetica < 1 g",
                        "includes 0 g added sugars/incluye 0 g de azucares anadidos 0 %",
                        "protein/proteinas 1 g", "total fat/grasa total 9 g",
                        "about 7 servings per container/ aprox. 7 raciones por envase",
                        "", "trans fat/grasa trans 0 g", "facts", "vitamin d/vitamina d 0 mcg",
                        "iron/hierro 0.1 mg", "calcium/calcio 0 mg", "datos de nutricion",
                        "cholesterol/colesterol 0 mg", "saturated fat/grasa saturada 2.5 g",
                        "sodium/sodio 160 mg", "2 %", "13 %", "", "potassium/potasio 110 mg",
                        "", "12 %", "total carbohydrate/carbohidrato total 16 g"],

    "gt_dict" : {
                "calories": ("150", ""),
                "total_fat": ("9", "g"),
                "saturated_fat": ("2.5", "g"),
                "trans_fat": ("0", "g"),
                "cholesterol": ("0", "mg"),
                "sodium": ("160", "mg"),
                "total_carbs": ("16", "g"),
                "dietary_fiber": ("<1", "g"),
                "sugars": ("<1", "g"),
                "protein": ("1", "g"),
                "vitamin_d": ("0", "mcg"),
                "calcium": ("0", "mg"),
                "iron": ("0.1", "mg"),
                "potassium": ("110", "mg")
            },

    "gt_boxes" : []
}

test_cases = {
    '1': test_1,
    '2': test_2,
    '3': test_3,
}

# ====================================

def levenshtein_distance(s1, s2):
    len_s1, len_s2 = len(s1), len(s2)
    if len_s1 == 0: return len_s2
    if len_s2 == 0: return len_s1
    dp = [[0] * (len_s2 + 1) for _ in range(len_s1 + 1)]
    for i in range(len_s1 + 1): dp[i][0] = i
    for j in range(len_s2 + 1): dp[0][j] = j
    for i in range(1, len_s1 + 1):
        for j in range(1, len_s2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1,
                             dp[i][j - 1] + 1,
                             dp[i - 1][j - 1] + cost)
    return dp[len_s1][len_s2]

# Metric functions
def calculate_cer(predicted, ground_truth):
    if not ground_truth:
        return float('inf') if predicted else 0.0
    return levenshtein_distance(predicted, ground_truth) / len(ground_truth)

def calculate_wer(predicted, ground_truth):
    pred_words = predicted.split()
    gt_words = ground_truth.split()
    if not gt_words:
        return float('inf') if pred_words else 0.0
    return levenshtein_distance(pred_words, gt_words) / len(gt_words)

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    return interArea / float(boxAArea + boxBArea - interArea + 1e-5)

def evaluate(pred_text, gt_text, pred_dict, gt_dict, pred_boxes = None, gt_boxes = None):
    results = {}

    # CER & WER
    results["CER"] = calculate_cer(pred_text, gt_text)
    results["WER"] = calculate_wer(pred_text, gt_text)

    # RMSE with tuple-style values (amount, unit)
    gt_vals = []
    pred_vals = []
    missing_keys = []

    for key, gt_val in gt_dict.items():
        if key in pred_dict:
            try:
                pred_val = float(pred_dict[key][0])
                gt_val_float = float(gt_val[0])
                pred_vals.append(pred_val)
                gt_vals.append(gt_val_float)
            except Exception as e:
                continue  # Skip if value can't be converted
        else:
            missing_keys.append(key)

    # RMSE
    if gt_vals and pred_vals:
        results["RMSE"] = np.sqrt(mean_squared_error(gt_vals, pred_vals))
    else:
        results["RMSE"] = None

    # Completeness
    results["Completeness"] = len(pred_vals) / len(gt_dict)

    # Optionally log missing fields
    results["Missing Fields"] = missing_keys

    # IoU
    # if pred_boxes and gt_boxes:
    #     ious = []
    #     for gt in gt_boxes:
    #         max_iou = max([calculate_iou(pred, gt) for pred in pred_boxes])
    #         ious.append(max_iou)
    #     results["IoU"] = np.mean(ious)
    # else:
    #     results["IoU"] = None

    return results

def get_rect_from_ctpn_box(box):
    """
    Extract a rectangular bounding box from an 8-coordinate CTPN box.
    Format: [x0, y0, x1, y1, x2, y2, x3, y3]
    Returns: [x_min, y_min, x_max, y_max]
    """
    xs = box[0::2]  # all x-coordinates: x0, x1, x2, x3
    ys = box[1::2]  # all y-coordinates: y0, y1, y2, y3
    return [min(xs), min(ys), max(xs), max(ys)]

def set_gt(nOftestcase):
    testcase = test_cases[nOftestcase]
    gt_text_by_box = testcase["gt_text_by_box"]
    gt_text = "\n".join(gt_text_by_box)
    gt_dict = testcase["gt_dict"]
    gt_boxes = testcase["gt_boxes"]

    return gt_text, gt_dict, gt_boxes

def eval(nOftestcase):
    image = cv2.imread(f"ocr/testing_img/test_{nOftestcase}.png")
    
    gt_text, gt_dict, gt_boxes = set_gt(nOftestcase)
    _, pred_text, nutrition_dict = extract_nutrition_text(image)
    # _, text_boxes = detect_text(image)
    # pred_boxes = [get_rect_from_ctpn_box(box) for box in text_boxes]
    
    # print("\n\n\n\n\nExtracted text:", pred_text)
    # print("\n==============================\n")
    # print("Ground truth:", gt_text)

    # print("Extracted nutrition dict:", nutrition_dict)
    # print("\n==============================\n")
    # print("Ground truth:", gt_dict)

    results = evaluate(pred_text, gt_text, nutrition_dict, gt_dict)
    print("\nEvaluation Results:")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"{k}: {v:.3f}")
        else:
            print(f"{k}: {v}")

# Run pipeline
if __name__ == "__main__":
    n = input("which test case to test[1,2,3,4]: ")
    eval(n)