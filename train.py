
# I used cross entropy loss. I originally tried using Dice loss to try to account for the imbalance in image area occupied by heart vs not heart, but I did not get good results. I may try retraining with some sort of IoU-like metric so that the loss function takes into account more than just per-pixel error.