from .model_wrappers import LinearModel
from .plot_templates import *
from .internal_validation import *
from .tools import connect_cursors

import mplcursors

def perform_regression_validation_and_plot(X, y, ax, highlights=None, labels=False, stats_loc='upper left'):
    model = LinearModel(X,y)
    model.fit()
    
    statistics = {}
    r2_train = r2_score(y, model.model.fittedvalues)
    statistics['R2 train'] = r2_train
    statistics['R2 adj train'] = model.model.rsquared_adj
    loo_results = calculate_leave_one_out(model.X_model, y)
    statistics['LOO train'] = loo_results['loo_r2']
    cv3_results = calculate_cross_validation(model.X_model, y, n_splits=3)
    statistics['3-cv train'] = cv3_results['cv mean']
    
    if len(X.shape) == 1 or X.shape[-1] == 1:
        plot_training_set(ax, x=model.X_model.iloc[:,-1], y=model.y, highlights=highlights)
        popup = mplcursors.cursor(ax, hover=True)
        connect_cursors(popup,X.index.values)
        add_regression_line(ax, model.X_model.iloc[:,-1], model.model.fittedvalues)
        add_confidence_interval(ax, model.X_model.iloc[:,-1], model.regressionline_summary['mean_ci_lower'], model.regressionline_summary['mean_ci_upper'])
        if labels:
            if isinstance(labels, list):
                add_labels(x=model.X_model.iloc[:,-1].loc[labels], y=model.y.loc[labels], labels=X.loc[labels].index.values, ax=ax)
            else:
                add_labels(x=model.X_model.iloc[:,-1], y=model.y, labels=X.index.values, ax=ax)
    else:
        ax.set_ylabel('Measured')
        ax.set_xlabel('Predicted')

        plot_training_set(ax, x=model.model.fittedvalues, y=model.y, highlights=highlights)
        popup = mplcursors.cursor(ax, hover=True)
        connect_cursors(popup,X.index.values)
        add_regression_line(ax, model.y, model.regressionline.fittedvalues)
        add_confidence_interval(ax, model.y, model.regressionline_summary['mean_ci_lower'], model.regressionline_summary['mean_ci_upper'])
        add_identity_line(ax)
        plot_leave_one_out(ax, x=loo_results['loo_predictions'], y=model.y)
        if labels:
            if isinstance(labels, list):
                add_labels(x=model.model.fittedvalues.loc[labels], y=model.y.loc[labels], labels=X.loc[labels].index.values, ax=ax)
            else:
                add_labels(x=model.model.fittedvalues, y=model.y, labels=X.index.values, ax=ax)
    
    if stats_loc is not None:
        add_model_statistics(ax, statistics=statistics, loc=stats_loc)

    add_plot_legend(ax)