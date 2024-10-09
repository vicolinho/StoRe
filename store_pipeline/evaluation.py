import os
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score


def evaluate_predictions(predictions_dictionary, gold_links):
    """
    Evaluate the performance of active learning predictions by calculating F1 score, precision, and recall.

    Parameters:
    predictions_folder (str): The path to the folder containing the prediction CSV files.

    Returns:
    None
    """
    # List to store individual DataFrames from all prediction files
    tps = 0
    fps = 0
    fns = 0
    for task, predictions in predictions_dictionary.items():
        predicted_matches = predictions[0]
        class_non_matches = predictions[1]
        pair_confidence = predictions[2]
        for pred_match in predicted_matches:
            pair = tuple(sorted((pred_match[0], pred_match[1])))
            if pair in gold_links:
                tps += 1
            else:
                fps += 1
        for non_match in class_non_matches:
            pair = tuple(sorted((non_match[0], non_match[1])))
            if pair in gold_links:
                fns += 1
    p = tps / (tps + fps)
    r = tps / (tps + fns)
    f1 = 2 * p * r / (p + r)
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {p:.4f}")
    print(f"Recall: {r:.4f}")
    return tps, fps, fns, p, r, f1


def evaluate_prediction_per_model(predictions_dictionary, gold_links):
    """
    Evaluate the performance of active learning predictions by calculating F1 score, precision, and recall.

    Parameters:
    predictions_folder (str): The path to the folder containing the prediction CSV files.

    Returns:
    None
    """
    # List to store individual DataFrames from all prediction files
    model_results = {}
    for task, predictions in predictions_dictionary.items():
        predicted_matches = predictions[0]
        class_non_matches = predictions[1]
        pair_confidence = predictions[2]
        tps = 0
        fps = 0
        fns = 0
        for pred_match in predicted_matches:
            pair = tuple(sorted((pred_match[0], pred_match[1])))
            if pair in gold_links:
                tps += 1
            else:
                fps += 1
        for non_match in class_non_matches:
            pair = tuple(sorted((non_match[0], non_match[1])))
            if pair in gold_links:
                fns += 1
        if tps != 0:
            p = tps / (tps + fps)
            r = tps / (tps + fns)
            f1 = 2 * p * r / (p + r)
        else:
            p, r, f1 = 0, 0, 0
        model_results[task] = tps, fps, fns, p, r, f1
    return model_results

# Example usage (assuming the predictions folder path is defined):
# evaluate_predictions('/path/to/predictions/folder')
