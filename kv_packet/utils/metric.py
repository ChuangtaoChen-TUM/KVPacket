from collections import Counter

def f1_score(gold_tokens, pred_tokens):
    """
    Computes precision, recall, and F1-score between two lists of tokens,
    correctly handling multi-word occurrences.
    
    Args:
        gold_tokens (list of str): The correct answer tokens.
        pred_tokens (list of str): The predicted answer tokens.
    
    Returns:
        (precision, recall, f1): Tuple of floats.
    """
    # If either list is empty, return 0s
    if not gold_tokens or not pred_tokens:
        return 0.0, 0.0, 0.0

    gold_counts = Counter(gold_tokens)
    pred_counts = Counter(pred_tokens)

    # True positives: the intersection of the two bags of words
    # This is the sum of the minimum counts for each common token
    common_tokens = gold_counts & pred_counts
    tp = sum(common_tokens.values())

    # The total number of predicted tokens is tp + fp
    # The total number of gold tokens is tp + fn
    num_pred = len(pred_tokens)
    num_gold = len(gold_tokens)

    precision = tp / num_pred
    recall = tp / num_gold
    
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def f1_states(
    gold_tokens: list[str],
    pred_tokens: list[str]
) -> tuple[int, int, int]:
    """
    Computes the number of true positives, false positives, and false negatives.
    """
    gold_counts = Counter(gold_tokens)
    pred_counts = Counter(pred_tokens)
    
    # True positives: the intersection of the two bags of words
    common_tokens = gold_counts & pred_counts
    tp = sum(common_tokens.values())
    
    # False positives: tokens in prediction but not in gold
    # This is the total number of predicted tokens minus the true positives
    fp = len(pred_tokens) - tp
    
    # False negatives: tokens in gold but not in prediction
    # This is the total number of gold tokens minus the true positives
    fn = len(gold_tokens) - tp
    
    return tp, fp, fn


def calculate_metrics(tp, fp, fn):
    """
    Calculates precision, recall, and F1 from TP, FP, and FN counts.
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1