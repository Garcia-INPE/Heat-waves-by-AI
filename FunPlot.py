"""Plotting functions for regression analysis visualization."""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.inspection import permutation_importance


def plot_time_series_reduced_ticks(y_test, X_test_full, pred_values,
                                   date_labels, v_str, max_ticks=15):
    """
    Time series plot with reduced x-axis tick marks for better readability.

    Parameters:
    -----------
    y_test : array-like
        Observed values
    X_test : DataFrame
        Test features including Tx_Mod
    pred_values : dict
        Dictionary with model predictions {model_name: predictions}
    X_test_full : DataFrame
        Full test data including P90cl_Tx_Obs
    date_labels : array-like
        Labels for x-axis
    v_str : str
        Version string for title
    max_ticks : int
        Maximum number of x-axis ticks to display

    Returns:
    --------
    fig : matplotlib.figure.Figure
    """
    plt.close("all")
    fig = plt.figure(figsize=(14, 6))

    n_view = len(y_test)
    x_ax = np.arange(n_view)
    p90_tx_obs = X_test_full['P90cl_Tx_Obs'].values[:n_view]

    # Plot observed and baseline
    plt.plot(x_ax, y_test, 'o-', label='Observed (Truth)',
             color='black', linewidth=2, markersize=4)
    plt.plot(x_ax, X_test_full['Tx_Mod'].values, '--',
             label='Eta Model (Uncalibrated)', color='gray', linewidth=2, alpha=0.7)
    plt.plot(x_ax, p90_tx_obs, '-.', label='P90cl_Tx_Obs',
             color='orange', linewidth=2.2, alpha=0.8)

    # Highlight where observed >= P90
    highlight_mask = y_test >= p90_tx_obs
    highlight_idx = np.where(highlight_mask)[0]
    if len(highlight_idx) > 0:
        plt.scatter(highlight_idx, p90_tx_obs[highlight_idx],
                    color='orange', edgecolors='black', s=90, zorder=6,
                    label='P90cl_Tx_Obs (highlight)')

    # Plot calibrated models
    colors = ['green', 'blue', 'red']
    pred_values_plot = {k: v for k, v in pred_values.items() if k != 'Eta'}
    for (name, preds), color in zip(pred_values_plot.items(), colors):
        plt.plot(x_ax, preds[:n_view], label=f'{name}',
                 color=color, alpha=0.7, linewidth=2)

    # Set reduced tick marks
    tick_stride = max(1, n_view // max_ticks)
    plt.xticks(x_ax[::tick_stride],
               [date_labels.iloc[i] if hasattr(date_labels, 'iloc') else date_labels[i]
                for i in range(0, n_view, tick_stride)],
               rotation=45, ha='right', fontsize=9)

    plt.title(f'Model Calibration Comparison (Time Series) - v{v_str}',
              fontsize=14, fontweight='bold')
    plt.ylabel('Max Temperature (°C)', fontsize=12)
    plt.xlabel('Date (rod | prev)', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig


def plot_predicted_vs_actual(y_test, pred_values, v_str):
    """
    Scatter plots of predicted vs actual values for each model.

    Parameters:
    -----------
    y_test : array-like
        Observed values
    pred_values : dict
        Dictionary with model predictions {model_name: predictions}
    v_str : str
        Version string for title

    Returns:
    --------
    fig : matplotlib.figure.Figure
    """
    plt.close("all")

    # Exclude baseline (Eta) from scatter plots
    pred_values_plot = {k: v for k, v in pred_values.items() if k != 'Eta'}
    n_models = len(pred_values_plot)

    fig = plt.figure(figsize=(15, 5))

    for idx, (name, preds) in enumerate(pred_values_plot.items(), 1):
        ax = plt.subplot(1, n_models, idx)

        # Scatter plot
        ax.scatter(y_test, preds, alpha=0.6, s=60,
                   edgecolors='black', linewidth=0.5)

        # Perfect prediction line
        min_val = min(y_test.min(), min(preds))
        max_val = max(y_test.max(), max(preds))
        ax.plot([min_val, max_val], [min_val, max_val],
                'r--', linewidth=2, alpha=0.7, label='Perfect Prediction')

        # Calculate metrics
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        corr = np.corrcoef(y_test, preds)[0, 1]

        # Add metrics text box
        metrics_text = f'RMSE: {rmse:.4f}\nMAE: {mae:.4f}\nCORR: {corr:.4f}'
        ax.text(0.05, 0.95, metrics_text,
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        ax.set_xlabel('Observed (°C)', fontsize=11)
        ax.set_ylabel('Predicted (°C)', fontsize=11)
        ax.set_title(f'{name} Model', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(True, alpha=0.3)

        # Equal aspect ratio for easier interpretation
        ax.set_aspect('equal', adjustable='box')

    fig.suptitle(
        f'Predicted vs Actual - v{v_str}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    return fig


def plot_residuals(y_test, pred_values, X_test_full, date_labels, v_str, max_ticks=15):
    """
    Residual plots (prediction errors) over time for each model.

    Parameters:
    -----------
    y_test : array-like
        Observed values
    pred_values : dict
        Dictionary with model predictions {model_name: predictions}
    X_test_full : DataFrame
        Full test data
    date_labels : array-like
        Labels for x-axis
    v_str : str
        Version string for title
    max_ticks : int
        Maximum number of x-axis ticks to display

    Returns:
    --------
    fig : matplotlib.figure.Figure
    """
    plt.close("all")

    # Exclude baseline from residual analysis
    pred_values_plot = {k: v for k, v in pred_values.items() if k != 'Eta'}
    n_models = len(pred_values_plot)

    fig = plt.figure(figsize=(15, 4 * n_models))

    n_view = len(y_test)
    x_ax = np.arange(n_view)
    tick_stride = max(1, n_view // max_ticks)

    for idx, (name, preds) in enumerate(pred_values_plot.items(), 1):
        ax = plt.subplot(n_models, 1, idx)

        # Calculate residuals
        residuals = np.array(preds) - y_test

        # Plot residuals
        ax.plot(x_ax, residuals, 'o-', color='steelblue',
                alpha=0.7, linewidth=1.5, markersize=5)
        ax.axhline(y=0, color='red', linestyle='--',
                   linewidth=2, label='Zero Error')

        # Add mean residual line
        mean_res = residuals.mean()
        ax.axhline(y=mean_res, color='green', linestyle=':',
                   linewidth=2, label=f'Mean: {mean_res:.3f}')

        # Calculate metrics
        mae = np.abs(residuals).mean()
        rmse = np.sqrt((residuals ** 2).mean())

        # Metrics text
        metrics_text = f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nBias: {mean_res:.4f}'
        ax.text(0.02, 0.98, metrics_text,
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        # Set ticks
        ax.set_xticks(x_ax[::tick_stride])
        ax.set_xticklabels([date_labels.iloc[i] if hasattr(date_labels, 'iloc') else date_labels[i]
                           for i in range(0, n_view, tick_stride)],
                           rotation=45, ha='right', fontsize=9)

        ax.set_ylabel('Residual (°C)', fontsize=11)
        ax.set_title(f'{name} Model Residuals', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.xlabel('Date (rod | prev)', fontsize=12)
    fig.suptitle(f'Residual Analysis - v{v_str}',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    return fig


def plot_model_subpanels(y_test, X_test_full, pred_values,
                         date_labels, v_str, max_ticks=15):
    """
    Multi-panel plot with each model in its own subplot.

    Parameters:
    -----------
    y_test : array-like
        Observed values
    X_test : DataFrame
        Test features including Tx_Mod
    pred_values : dict
        Dictionary with model predictions {model_name: predictions}
    X_test_full : DataFrame
        Full test data including P90cl_Tx_Obs
    date_labels : array-like
        Labels for x-axis
    v_str : str
        Version string for title
    max_ticks : int
        Maximum number of x-axis ticks to display

    Returns:
    --------
    fig : matplotlib.figure.Figure
    """
    plt.close("all")

    pred_values_plot = {k: v for k, v in pred_values.items() if k != 'Eta'}
    n_models = len(pred_values_plot)

    fig = plt.figure(figsize=(15, 4 * n_models))

    n_view = len(y_test)
    x_ax = np.arange(n_view)
    p90_tx_obs = X_test_full['P90cl_Tx_Obs'].values[:n_view]
    tick_stride = max(1, n_view // max_ticks)

    for idx, (name, preds) in enumerate(pred_values_plot.items(), 1):
        ax = plt.subplot(n_models, 1, idx)

        # Plot observed, baseline, and current model
        ax.plot(x_ax, y_test, 'o-', label='Observed',
                color='black', linewidth=2, markersize=4)
        ax.plot(x_ax, X_test_full['Tx_Mod'].values, '--',
                label='Eta (baseline)', color='gray', linewidth=1.5, alpha=0.6)
        ax.plot(x_ax, preds, '-', label=f'{name} (calibrated)',
                color='blue', linewidth=2, alpha=0.8)
        ax.plot(x_ax, p90_tx_obs, '-.', label='P90cl_Tx_Obs',
                color='orange', linewidth=1.5, alpha=0.6)

        # Calculate metrics
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        corr = np.corrcoef(y_test, preds)[0, 1]

        # Metrics text
        metrics_text = f'RMSE: {rmse:.4f} | MAE: {mae:.4f} | CORR: {corr:.4f}'
        ax.text(0.02, 0.98, metrics_text,
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

        # Set ticks
        ax.set_xticks(x_ax[::tick_stride])
        ax.set_xticklabels([date_labels.iloc[i] if hasattr(date_labels, 'iloc') else date_labels[i]
                           for i in range(0, n_view, tick_stride)],
                           rotation=45, ha='right', fontsize=9)

        ax.set_ylabel('Temperature (°C)', fontsize=11)
        ax.set_title(f'{name} Model', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.xlabel('Date (rod | prev)', fontsize=12)
    fig.suptitle(f'Model Comparison (Individual Panels) - v{v_str}',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    return fig


def plot_comprehensive_dashboard(y_test, X_test, pred_values, X_test_full,
                                 date_labels, v_str, max_ticks=15):
    """
    Comprehensive dashboard combining time series and scatter plots.

    Parameters:
    -----------
    y_test : array-like
        Observed values
    X_test : DataFrame
        Test features including Tx_Mod
    pred_values : dict
        Dictionary with model predictions {model_name: predictions}
    X_test_full : DataFrame
        Full test data including P90cl_Tx_Obs
    date_labels : array-like
        Labels for x-axis
    v_str : str
        Version string for title
    max_ticks : int
        Maximum number of x-axis ticks to display

    Returns:
    --------
    fig : matplotlib.figure.Figure
    """
    plt.close("all")
    fig = plt.figure(figsize=(16, 10))

    # Prepare data
    n_view = len(y_test)
    x_ax = np.arange(n_view)
    p90_tx_obs = X_test_full['P90cl_Tx_Obs'].values[:n_view]
    tick_stride = max(1, n_view // max_ticks)

    # Panel 1: Time series with all models (top, spanning 3 columns)
    ax1 = plt.subplot(2, 3, (1, 3))
    ax1.plot(x_ax, y_test, 'o-', label='Observed',
             color='black', linewidth=2, markersize=4)
    ax1.plot(x_ax, X_test['Tx_Mod'].values, '--', label='Eta (baseline)',
             color='gray', alpha=0.7, linewidth=1.5)

    colors = {'RF': 'green', 'XGB': 'blue', 'NN': 'red'}
    for name, preds in pred_values.items():
        if name != 'Eta':
            ax1.plot(x_ax, preds, label=name, color=colors.get(name, 'purple'),
                     alpha=0.7, linewidth=2)

    ax1.axhline(p90_tx_obs.mean(), color='orange', linestyle='-.',
                alpha=0.5, linewidth=1.5, label='P90 mean')
    ax1.set_ylabel('Temperature (°C)', fontsize=11)
    ax1.set_title(f'Time Series - v{v_str}', fontsize=12, fontweight='bold')
    ax1.set_xticks(x_ax[::tick_stride])
    ax1.set_xticklabels([date_labels.iloc[i] if hasattr(date_labels, 'iloc') else date_labels[i]
                         for i in range(0, n_view, tick_stride)],
                        rotation=45, ha='right', fontsize=8)
    ax1.legend(loc='best', fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3)

    # Panels 2-4: Predicted vs Actual scatter plots (bottom row)
    panel_idx = 4
    for name, preds in pred_values.items():
        if name == 'Eta':
            continue

        ax = plt.subplot(2, 3, panel_idx)
        ax.scatter(y_test, preds, alpha=0.6, s=50, edgecolors='black', linewidth=0.5,
                   color=colors.get(name, 'purple'))

        # Perfect prediction line
        min_val = min(y_test.min(), min(preds))
        max_val = max(y_test.max(), max(preds))
        ax.plot([min_val, max_val], [min_val, max_val],
                'r--', linewidth=2, alpha=0.7, label='Perfect')

        # Metrics
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        corr = np.corrcoef(y_test, preds)[0, 1]

        metrics_text = f'RMSE: {rmse:.3f}\nMAE: {mae:.3f}\nCORR: {corr:.3f}'
        ax.text(0.05, 0.95, metrics_text,
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        ax.set_xlabel('Observed (°C)', fontsize=10)
        ax.set_ylabel('Predicted (°C)', fontsize=10)
        ax.set_title(f'{name} Model', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

        panel_idx += 1

    plt.tight_layout()

    return fig


def plot_xgb_candidate_time_series(df, date_cols=['data_rod', 'data_prev'], max_ticks=15):
    """
    Time series plot for XGB candidate comparison.

    Plots XGB predictions, baseline Eta model, P90 threshold, and highlights where
    XGB exceeds the P90 threshold.

    Parameters:
    -----------
    df : DataFrame
        Data with columns: XGB_Pred, Tx_Mod, P90cl_Tx_Obs, OC_Mod_Candid
    date_cols : list
        Column names to use for date labels [rod_col, prev_col]
    max_ticks : int
        Maximum number of x-axis ticks to display

    Returns:
    --------
    fig : matplotlib.figure.Figure
    """
    plt.close("all")

    df_sorted = df.sort_values(date_cols).reset_index(drop=True)
    n_view = len(df_sorted)
    x_ax = np.arange(n_view)

    # Create date labels from date columns
    date_labels = (df_sorted[date_cols[0]].astype(str) + '\n' +
                   df_sorted[date_cols[1]].astype(str)).values

    fig = plt.figure(figsize=(14, 6))

    # Plot observations and baseline
    plt.plot(x_ax, df_sorted['P90cl_Tx_Obs'], '-.',
             label='P90cl_Tx_Obs (Threshold)', color='orange', linewidth=2.2, alpha=0.8)
    plt.plot(x_ax, df_sorted['Tx_Obs'], 'o-', label='Observed (Truth)',
             color='black', linewidth=2, markersize=4)
    plt.plot(x_ax, df_sorted['Tx_Mod'], '--',
             label='Eta Model (Uncalibrated)', color='gray', linewidth=2, alpha=0.7)

    # Plot calibrated models
    plt.plot(x_ax, df_sorted['XGB_Pred'], 'o-',
             label='XGB_Pred (Calibrated)', color='blue', linewidth=2, alpha=0.8, markersize=4)

    # Highlight where XGB_Pred exceeds P90
    highlight_mask = df_sorted['XGB_Pred'] >= df_sorted['P90cl_Tx_Obs']
    highlight_idx = np.where(highlight_mask)[0]
    if len(highlight_idx) > 0:
        plt.scatter(highlight_idx, df_sorted['P90cl_Tx_Obs'].values[highlight_idx],
                    color='red', edgecolors='darkred', s=100, zorder=6, alpha=0.7,
                    marker='^', label='XGB Exceeds P90')

    # Set reduced tick marks for readability
    tick_stride = max(1, n_view // max_ticks)
    tick_positions = x_ax[::tick_stride]
    tick_labels = [date_labels[i] for i in range(0, n_view, tick_stride)]

    plt.xticks(tick_positions, tick_labels,
               rotation=45, ha='right', fontsize=9)

    plt.title('XGB Candidate Time Series Comparison',
              fontsize=14, fontweight='bold')
    plt.ylabel('Temperature (°C)', fontsize=12)
    plt.xlabel('Date (rod | prev)', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig


def _binary_metrics_from_cm(y_true, y_pred):
    """Compute binary classification metrics from confusion matrix values."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total = tn + fp + fn + tp

    accuracy = (tn + tp) / total if total else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1_score = (2 * precision * sensitivity) / (precision +
                                                sensitivity) if (precision + sensitivity) else 0.0

    return {
        'Accuracy': accuracy,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Precision': precision,
        'F1-Score': f1_score,
    }


def _plot_confusion_heatmap(ax, y_true, y_pred, title, cmap):
    """Plot a confusion matrix heatmap into the provided axes."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap=cmap,
        cbar=False,
        linewidths=0.8,
        linecolor='white',
        square=True,
        xticklabels=['Pred 0', 'Pred 1'],
        yticklabels=['True 0', 'True 1'],
        ax=ax
    )
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('')


def plot_composite_analysis(df_pred_classif, models, feature_names, X_test, X_test_scaled, y_test, v_str):
    """Create a 3x3 composite plot for baseline and calibrated model candidate analyses."""
    plt.close("all")
    n_features = max(1, len(feature_names))
    fig_height = max(17, 11 + 0.7 * n_features)
    fig, axes = plt.subplots(3, 3, figsize=(21, fig_height))

    y_true = df_pred_classif['OC_Obs_Candid'].astype(int).values
    baseline_pred = df_pred_classif['OC_Mod_Candid'].astype(int).values
    nn_pred = df_pred_classif['OC_NN_Candid'].astype(int).values
    rf_pred = df_pred_classif['OC_RF_Candid'].astype(int).values
    xgb_pred = df_pred_classif['OC_XGB_Candid'].astype(int).values

    model_preds = {
        'Baseline': baseline_pred,
        'NN': nn_pred,
        'RF': rf_pred,
        'XGB': xgb_pred,
    }

    metric_names = ['Accuracy', 'Sensitivity',
                    'Specificity', 'Precision', 'F1-Score']
    metrics = {name: _binary_metrics_from_cm(
        y_true, pred) for name, pred in model_preds.items()}

    # Row 1, Col 1: Baseline confusion matrix
    _plot_confusion_heatmap(
        axes[0, 0],
        y_true,
        baseline_pred,
        'Baseline Confusion Matrix\n(OC_Mod_Candid vs OC_Obs_Candid)',
        'Oranges'
    )

    # Row 1, Col 2: Vertical multi-bar metrics comparison
    x = np.arange(len(metric_names))
    width = 0.2
    model_order = ['Baseline', 'NN', 'RF', 'XGB']
    colors = ['#707B7C', '#5DADE2', '#58D68D', '#AF7AC5']

    for idx, model_name in enumerate(model_order):
        axes[0, 1].bar(
            x + (idx - 1.5) * width,
            [metrics[model_name][m] for m in metric_names],
            width=width,
            label=model_name,
            color=colors[idx],
            edgecolor='black',
            linewidth=0.5
        )

    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(metric_names, rotation=45, ha='right')
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_title('Performance Metrics Comparison',
                         fontsize=11, fontweight='bold')
    axes[0, 1].legend(fontsize=9)

    # Row 1, Col 3: Horizontal multi-bar improvements against baseline
    improvements = {
        model_name: [metrics[model_name][m] - metrics['Baseline'][m]
                     for m in metric_names]
        for model_name in ['NN', 'RF', 'XGB']
    }
    y_pos = np.arange(len(metric_names))
    h = 0.22
    for idx, model_name in enumerate(['NN', 'RF', 'XGB']):
        axes[0, 2].barh(
            y_pos + (idx - 1) * h,
            improvements[model_name],
            height=h,
            label=model_name,
            color=colors[idx + 1],
            edgecolor='black',
            linewidth=0.5
        )

    axes[0, 2].set_yticks(y_pos)
    axes[0, 2].set_yticklabels(metric_names)
    axes[0, 2].axvline(0, color='black', linewidth=1)
    axes[0, 2].set_xlabel('Delta (Model - Baseline)')
    axes[0, 2].set_title('Performance Improvement vs Baseline',
                         fontsize=11, fontweight='bold')
    axes[0, 2].legend(fontsize=9)

    # Row 2: Confusion matrix per model estimate
    _plot_confusion_heatmap(
        axes[1, 0], y_true, nn_pred, 'NN Confusion Matrix', 'Blues')
    _plot_confusion_heatmap(
        axes[1, 1], y_true, rf_pred, 'RF Confusion Matrix', 'Greens')
    _plot_confusion_heatmap(
        axes[1, 2], y_true, xgb_pred, 'XGB Confusion Matrix', 'Purples')

    # Row 3: Feature importance for NN, RF and XGB
    rf_importance = models['RF'].feature_importances_
    xgb_importance = models['XGB'].feature_importances_
    nn_perm = permutation_importance(
        models['NN'],
        X_test_scaled,
        y_test,
        n_repeats=10,
        random_state=42,
        scoring='neg_mean_squared_error'
    )
    nn_importance = nn_perm.importances_mean

    importance_data = {
        'NN': nn_importance,
        'RF': rf_importance,
        'XGB': xgb_importance,
    }
    importance_colors = {'NN': '#5DADE2', 'RF': '#58D68D', 'XGB': '#AF7AC5'}
    label_font_size = 9 if n_features <= 10 else (8 if n_features <= 16 else 7)

    for ax, model_name in zip(axes[2, :], ['NN', 'RF', 'XGB']):
        df_imp = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_data[model_name]
        }).sort_values(by='Importance', ascending=False)

        df_imp['Importance'] = df_imp['Importance'].fillna(0.0)
        df_imp['Feature'] = pd.Categorical(
            df_imp['Feature'],
            categories=df_imp['Feature'].tolist(),
            ordered=True
        )

        sns.barplot(data=df_imp, x='Importance', y='Feature',
                    color=importance_colors[model_name], ax=ax)
        ax.set_title(f'{model_name} Feature Importance',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Importance')
        ax.set_ylabel('')
        ax.tick_params(axis='y', labelsize=label_font_size)

    fig.suptitle(f'Composite Analysis - v{v_str}',
                 fontsize=16, fontweight='bold', y=0.995)
    fig.subplots_adjust(hspace=0.35)
    plt.tight_layout(rect=(0, 0, 1, 0.985))

    return fig
