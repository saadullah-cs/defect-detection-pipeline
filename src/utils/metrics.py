import logging
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)

def save_confusion_matrix(
    y_true: List[int], 
    y_pred: List[int], 
    classes: List[str], 
    output_path: Path
) -> None:
    """
    Computes and saves a stylized confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="whitegrid")
    
    # Generate a custom diverging colormap
    cmap = sns.color_palette("Blues", as_cmap=True)
    
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap=cmap, 
        xticklabels=classes, 
        yticklabels=classes,
        cbar_kws={'label': 'Number of Samples'},
        linewidths=.5
    )
    
    plt.title('Defect Classification Confusion Matrix', pad=20, fontsize=16)
    plt.ylabel('True Defect Type', weight='bold')
    plt.xlabel('Predicted Defect Type', weight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Confusion matrix saved successfully to {output_path}")