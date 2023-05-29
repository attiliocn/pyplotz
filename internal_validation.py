from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut, RepeatedKFold, cross_val_score

def calculate_leave_one_out(x,y):
    loo_scores = cross_val_score(
        LinearRegression(),
        x,
        y,
        cv=LeaveOneOut(),
        scoring='neg_mean_absolute_error'
    )

    loo_predictions = (y - loo_scores)
    loo_results = {
        'loo_predictions': loo_predictions,
        'loo_mae': abs(loo_scores.mean()),
        'loo_r2': r2_score(y, loo_predictions)
    }
    return loo_results

def calculate_cross_validation(x,y, n_splits=3, n_repeats=100):
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=0)
    cv = cross_val_score(LinearRegression(), x, y, cv=rkf)
    cv_results = {
        'cv predictions': cv,
        'cv mean': cv.mean()
    }
    return cv_results