def format_confidence(confidence):
    """Format confidence score with color based on value"""
    if confidence >= 0.8:
        color = "#ff4444"  # Strong red for high confidence fake
    elif confidence >= 0.6:
        color = "#ff8c00"  # Orange for medium confidence
    else:
        color = "#00ff9d"  # Green for low confidence (likely real)
    return f'<span style="color: {color}; font-weight: bold;">{confidence:.1%}</span>'

def format_prediction(prediction):
    """Format prediction with color"""
    color = "#ff4444" if prediction == "FAKE" else "#00ff9d"
    return f'<span style="color: {color}; font-weight: bold;">{prediction}</span>'