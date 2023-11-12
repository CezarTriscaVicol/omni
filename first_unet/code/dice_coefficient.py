
def dice_coefficient(pred, target):
    """Calculate Dice coefficient."""
    smooth = 1e-7  # To avoid division by zero
    pred_binary = (pred > 0.5).float()
    intersection = (pred_binary * target).sum()
    return (2. * intersection + smooth) / (pred_binary.sum() + target.sum() + smooth)