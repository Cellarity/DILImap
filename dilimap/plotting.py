"""Plotting functions for data visualization and enrichment analysis"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve as _roc_curve, auc


def roc_curve(
    y_true,
    y_pred,
    threshold_points=None,
    threshold_name=None,
    color='brown',
    label=None,
    axvline=None,
    inverse=False,
    show=True,
):
    """
    Plots a ROC curve with optional threshold annotations.

    Args:
        y_true (array): Ground truth binary labels.
        y_pred (array): Predicted probabilities or scores.
        threshold_points (list): A list of threshold values to annotate on the ROC curve.
        threshold_name (str): The label to use for the threshold in annotations (default is 'cutoff').
        color (str): Color of the ROC curve line (default is 'brown').
        label (str): Custom label for the ROC curve (default is 'ROC curve').
        inverse (bool): If True, inverts the y_pred scores (useful if higher values indicate negative class).
        show (bool): If True, displays the plot. If False, suppresses the plot display.

    Returns:
        Displays the ROC curve.
    """

    # Optionally invert predictions
    if inverse:
        y_pred = (-y_pred).copy()

    mask = ~np.isnan(y_true)
    y_pred = y_pred[mask].copy()
    y_true = y_true[mask].copy()

    # Compute ROC curve and AUC
    fpr, tpr, thresholds = _roc_curve(y_true, y_pred, drop_intermediate=False)
    roc_auc = auc(fpr, tpr)

    # Set label for ROC curve
    label = label or 'ROC curve'
    plt.plot(fpr, tpr, color=color, lw=2, label=f'{label} (AUC = {roc_auc:.2f})')

    # Plot random classifier line
    if show:
        plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Random Classifier')

    # Annotate threshold points
    threshold_name = threshold_name or 'cutoff'
    if threshold_points is not None:
        for threshold in threshold_points:
            idx = np.argmin(np.abs(thresholds - ((-threshold) if inverse else threshold)))
            plt.scatter(fpr[idx], tpr[idx], marker='o', color='navy', s=30)
            text = plt.text(
                fpr[idx] + 0.02,
                tpr[idx] - 0.01,
                f'{threshold_name} = {threshold}',
                color='navy',
                fontsize=10,
            )
            text.set_bbox({'facecolor': 'white', 'alpha': 0.6, 'edgecolor': 'navy'})
    if axvline:
        plt.axvline(0.15, linestyle='--', color='grey', linewidth=0.8)

    # Final plot settings
    if show:
        plt.xlabel('False Positive Rate (FPR)', fontsize=12)
        plt.ylabel('True Positive Rate (TPR)', fontsize=12)
        plt.title('ROC Curve', fontsize=14)
        plt.legend(loc='lower right')
        plt.grid(True, linewidth=0.2)
        plt.show()


def boxplot_with_swarm(
    data,
    x,
    y,
    hue_order=None,
    palette=None,
    box_alpha=0.2,
    swarm_size=3,
    box_width=0.3,
    axhline=None,
    xlabel='',
    ylabel='',
    show=True,
    swarm_kwargs=None,
    show_box=True,
    show_swarm=True,
):
    """
    Creates a box plot overlaid with a swarm plot for a given dataset.

    Args:
        data (DataFrame): DataFrame containing the data to plot.
        x (str): Column name to use for the x-axis (categorical).
        y (str): Column name to use for the y-axis (numeric).
        hue (str): Column name for grouping categories (optional).
        hue_order (list of str): Order of categories in the hue (optional).
        palette (list of colors): List of colors to use for different categories (optional).
        box_alpha (float): Transparency level for the box plot.
        swarm_size (float): Size of the dots in the swarm plot.
        box_width (float): Width of the box plot.
        axhline (float): Optional y-value to draw a horizontal line (e.g., cutoff).
        xlabel (str): Label for the x-axis (default is empty).
        ylabel (str): Label for the y-axis (default is 'Value').

    Returns:
        Displays the plot.
    """
    # Prepare the data (ensure hue column and its order, if specified)
    df_tmp = data.copy()

    swarm_kwargs = swarm_kwargs or {}

    if hue_order is not None:
        df_tmp = df_tmp[df_tmp[x].isin(hue_order)]
        df_tmp[x] = df_tmp[x].astype('str').astype('category')
        df_tmp[x] = df_tmp[x].cat.reorder_categories(hue_order, ordered=True)
        df_tmp[x] = df_tmp[x].str.replace(r' \(', '\n(', regex=True)
        hue_order = [o.replace(' (', '\n(') for o in hue_order]

    # Plot swarm plot
    if show_swarm:
        sns.swarmplot(
            data=df_tmp,
            x=x,
            y=y,
            hue=x,
            hue_order=hue_order,
            order=hue_order,
            palette=palette,
            size=swarm_size,
            linewidth=0.2,
            dodge=False,
            legend=None,
            **swarm_kwargs,
        )

    # Plot box plot
    if show_box:
        sns.boxplot(
            data=df_tmp,
            x=x,
            y=y,
            hue=x,
            hue_order=hue_order,
            order=hue_order,
            palette=palette,
            boxprops={'alpha': box_alpha},
            width=box_width,
            dodge=False,
            fliersize=0,
        )

    # Optional threshold line
    if axhline is not None:
        plt.axhline(axhline, linestyle='--', color='k')

    # Labels and aesthetics
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title=x)
    if show:
        plt.show()
