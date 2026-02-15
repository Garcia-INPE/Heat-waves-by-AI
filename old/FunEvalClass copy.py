import os
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

LINE_LEN = 55
DIR_RESULTS = 'results/classification'
if not os.path.exists(DIR_RESULTS):
    os.makedirs(DIR_RESULTS)


def eval_confusion_matrix(y_test, y_pred, cmap, classifier_name, classifier_acronym, v, show_res=True):
    # y_pred=rf_pred; cmap='Blues'; classifier_name="Random Forest"; classifier_acronym="RF"; show_res=True
    """
    Create and plot the confusion matrix.

    Parameters:
    - y_test: true labels
    - y_pred: predicted labels
    - cmap: color map for the plot
    - classifier_name: string to identify the classifier in the title
    - classifier_acronym: acronym to prefix metric names
    - v: version or timestamp to append to filenames
    - show_res: whether to display the plot and print results

    Returns:
    - A dictionary containing the confusion matrix elements.
    """

    # --------------------------------------------------------------
    # Plot confusion matrix
    # --------------------------------------------------------------
    # Clean the plt buffers to avoid overlapping plots
    plt.close('all')
    # Although fig is not use, it's better to capture it iot prevent showing a blank plot immediately
    fig, ax = plt.subplots(figsize=(10, 8))
    cm = ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=['No IOC', 'IOC'],
        cmap=cmap, ax=ax, values_format='d', text_kw={'fontsize': 30})
    cm.ax_.set_title(
        f'Confusion Matrix - {classifier_name} - v{v}', fontsize=20, fontweight='bold')
    cm.ax_.tick_params(axis='both', which='major',
                       labelsize=20)  # tick label size
    cm.ax_.set_xlabel(cm.ax_.get_xlabel(), fontsize=20)  # x-axis label size
    cm.ax_.set_ylabel(cm.ax_.get_ylabel(), fontsize=20)  # y-axis label size

    plt.tight_layout()
    plt.savefig(
        os.path.join(DIR_RESULTS, f'{classifier_acronym}-Confusion_Matrix_{v}.png'), dpi=300)
    if show_res:
        plt.show()
    plt.close("all")

    # --------------------------------------------------------------
    # Build a string with the Confusion Matrix as table
    # --------------------------------------------------------------
    res_text = "\n" + "-" * LINE_LEN + "\n"
    res_text += f"CONFUSION MATRIX - {classifier_name}\n"
    res_text += "-" * LINE_LEN + "\n"
    res_text += pd.DataFrame(cm.confusion_matrix, index=['Actual No IOC', 'Actual IOC'], columns=[
        'Predicted No IOC', 'Predicted IOC']).to_string() + "\n"
    res_text += "-" * LINE_LEN + "\n"

    # --------------------------------------------------------------
    # Feed a string with the Confusion Matrix breakdown as table
    # --------------------------------------------------------------
    res_text += "\n\n" + "-" * LINE_LEN + "\n"
    res_text += f"CONFUSION MATRIX BREAKDOWN - {classifier_name}\n"
    res_text += "-" * LINE_LEN + "\n"
    res_text += f"  • TN: {cm.confusion_matrix[0, 0]} - Correctly identified non-IOC cases\n"
    res_text += f"  • FP: {cm.confusion_matrix[0, 1]} - Incorrectly predicted as IOC\n"
    res_text += f"  • FN: {cm.confusion_matrix[1, 0]} - Missed IOC events\n"
    res_text += f"  • TP: {cm.confusion_matrix[1, 1]} - Correctly identified IOC events\n"
    res_text += "-" * LINE_LEN + "\n"

    # --------------------------------------------------------------
    # Feed a string with the Classification Report
    # --------------------------------------------------------------
    res_text += "\n\n" + "-" * LINE_LEN + "\n"
    res_text += f"CLASSIFICATION REPORT - {classifier_name}\n"
    res_text += "-" * LINE_LEN + "\n"
    res_text += classification_report(y_test, y_pred)
    res_text += "-" * LINE_LEN + "\n"

    with open(os.path.join(DIR_RESULTS, f"{classifier_acronym}-Confusion_Matrix_Results_{v}.txt"), "w", encoding="utf-8") as f:
        print(res_text, file=f)

    if show_res:
        print(res_text)

    return {
        f'{classifier_acronym}_tn': cm.confusion_matrix[0, 0],
        f'{classifier_acronym}_fp': cm.confusion_matrix[0, 1],
        f'{classifier_acronym}_fn': cm.confusion_matrix[1, 0],
        f'{classifier_acronym}_tp': cm.confusion_matrix[1, 1]
    }


def eval_metrics(y_test, y_pred, classifier_name, classifier_acronym, v, show_res=True):
    # y_pred=rf_pred; classifier_name="Random Forest"; classifier_acronym="RF"
    """ 
    Calculate and return evaluation metrics.

    Parameters:
    - y_test: true labels
    - y_pred: predicted labels
    - classifier_name: string to identify the classifier in the print statements
    - classifier_acronym: acronym to prefix metric names
    - v: version or identifier for saving results
    - show_res: whether to print the results

    Returns:
    - A dictionary containing accuracy, precision, recall, and F1-score.
    """

    res_accuracy = accuracy_score(y_test, y_pred)
    res_precision = precision_score(y_test, y_pred)
    res_recall = recall_score(y_test, y_pred)
    res_f1_score = f1_score(y_test, y_pred)

    # --------------------------------------------------------------
    # Build a string with the metrics
    # --------------------------------------------------------------
    res_text = "\n" + "-" * LINE_LEN + "\n"
    res_text += f"MODEL EVALUATION SUMMARY - {classifier_name}\n"
    res_text += "-" * LINE_LEN + "\n"
    res_text += f"Accuracy:  {res_accuracy:.3f}\n"
    res_text += f"Precision: {res_precision:.3f}\n"
    res_text += f"Recall:    {res_recall:.3f}\n"
    res_text += f"F1-Score:  {res_f1_score:.3f}\n"
    res_text += "-" * LINE_LEN + "\n"

    # Feed the Interpretation to a text file
    res_text += "\n\n" + "-" * LINE_LEN + "\n"
    res_text += f"INTERPRETATION - {classifier_name}\n"
    res_text += "-" * LINE_LEN + "\n"
    res_text += f"  • Model correctly identifies {res_accuracy*100:.1f}% of cases\n"
    res_text += f"  • When predicting IOC, correct {res_precision*100:.1f}% of the time\n"
    res_text += f"  • Catches {res_recall*100:.1f}% of actual IOC events (sensitivity)\n"
    res_text += f"  • Overall balance: F1-Score = {res_f1_score:.3f}\n"
    res_text += "-" * LINE_LEN + "\n"

    with open(os.path.join(DIR_RESULTS, f"{classifier_acronym}-Metrics_Results_{v}.txt"), "w", encoding="utf-8") as f:
        print(res_text, file=f)

    if show_res:
        print(res_text)

    return {
        f'{classifier_acronym}_accuracy': res_accuracy,
        f'{classifier_acronym}_precision': res_precision,
        f'{classifier_acronym}_recall': res_recall,
        f'{classifier_acronym}_f1_score': res_f1_score
    }


def eval_feature_importance(classifier, X_columns, classifier_name, classifier_acronym, v, show_res=True):
    """

    Plot and return feature importance from a classifier.

    Parameters:
    - classifier_name: string to identify the classifier in the title
    - classifier_acronym: acronym to prefix metric names
    - classifier: trained classifier
    - X_columns: list of feature names
    - v: version or identifier for saving results
    - show_res: whether to print the results

    Returns: 
    - DataFrame with feature importance    
    """

    importances = classifier.feature_importances_
    feature_names = X_columns
    feature_importance_df = pd.DataFrame(
        {'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(
        by='Importance', ascending=False)

    # Plot Random Forest feature importance
    plt.close('all')
    fig = plt.figure(figsize=(10, 6))
    bp = sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    # Set axes labels with larger font sizes
    plt.xlabel('Importance', fontsize=20)
    plt.ylabel('Feature', fontsize=20)

    # bp.ax.tick_params(axis='both', which='major',
    #                   labelsize=20)  # tick label size
    # bp.ax.set_xlabel(bp.ax.get_xlabel(), fontsize=20)  # x-axis label size
    # bp.ax.set_ylabel(bp.ax.get_ylabel(), fontsize=20)  # y-axis label size
    plt.title(
        f'FEATURE IMPORTANCE - {classifier_name} - v{v}', fontsize=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig(
        # Append a date and time stamp to the filename
        os.path.join(DIR_RESULTS, f'{classifier_acronym}-Feature_Importance_{v}.png'), dpi=300)
    if show_res:
        plt.show()
    plt.close("all")

    # Save the feature importance to CSV
    feature_importance_df.to_csv(
        os.path.join(DIR_RESULTS, f'{classifier_acronym}-Feature_Importance_{v}.csv'), index=False)

    return feature_importance_df


def plot_mlp_training_history(mlp_classifier):
    """
    Docstring for plot_mlp_training_history

    :param mlp_classifier: MLP classifier object
    :param X_test_scaled: Scaled test features
    :param y_test: True labels for the test set
    :param X: Original feature set
    """

    _, ax = plt.subplots(figsize=(10, 6))
    ax.plot(mlp_classifier.loss_curve_, label='Training Loss', linewidth=2)
    ax.set_title('MLP Neural Network Training Loss Over Iterations')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Iteration')
    ax.legend()
    ax.grid()
    plt.tight_layout()
    plt.show()


def plot_mlp_feature_importance(mlp_classifier, X_test_scaled, y_test, X):
    """
    Docstring for plot_mlp_feature_importance

    :param mlp_classifier: MLP classifier object
    :param X_test_scaled: Scaled test features
    :param y_test: True labels for the test set
    :param X: Original feature set
    """

    perm_importance = permutation_importance(
        mlp_classifier, X_test_scaled, y_test, n_repeats=10, random_state=42)
    feature_names = X.columns
    perm_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': perm_importance.importances_mean,
    })

    perm_importance_df = perm_importance_df.sort_values(
        by='Importance', ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature',
                data=perm_importance_df, color='salmon')
    plt.title('Permutation Feature Importance from MLP Neural Network')
    plt.show()
