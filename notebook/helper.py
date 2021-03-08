import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Optional

def mnist_show_me(list_of_images:np.array=None, title:Optional[str]=None, 
                    cmap:Optional[str]="gray") -> None:
    """display images

    Args:
        list_of_images (np.array): list of images 

    Returns:
        [type]: plt.subplots figure
    """
    if list_of_images is not None:
        rows = 3 if not len(list_of_images) % 3 else 2
        cols = len(list_of_images) // rows
        rows, cols = int(rows), int(cols)
        fig, axes = plt.subplots(rows, cols)
        count = 0
        for i in range(rows):
            for j in range(cols):
                plt.subplot(rows, cols, count+1)
                plt.imshow(list_of_images[count], cmap=cmap)
                count += 1
        plt.suptitle(f"{title}")

def confusion_matrix(y_true:np.array, y_preds:np.array) -> None:
    plt.figure(figsize=(10,7))
    y_actu = pd.Series(y_true, name='Actual')
    y_pred = pd.Series(y_preds, name='Predicted')
    cm = pd.crosstab(y_actu, y_pred)
    ax = sns.heatmap(cm, annot=True, fmt="d")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


if __name__ == "__main__":
    fig, axes = mnist_show_me([[1, 2], [2, 2]])
    print(axes)
    print(axes[0])