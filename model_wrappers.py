import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

class LinearModel():
    def __init__(self, X,y):
        self.dataset = pd.concat([X,y], axis=1)
        self.X = X
        self.y = y
        self.is_fitted = False
        self.model = None
        self.results = None

    def fit(self, formula=None, feature_columns_idx=None):
        if formula:
            self.model = smf.ols(formula=formula, data=self.dataset)
        elif feature_columns_idx:
            self.model = sm.OLS(self.y, sm.add_constant(self.dataset.iloc[:,feature_columns_idx]))
        else:
            self.model = sm.OLS(self.y, sm.add_constant(self.X))
        
        self.X = pd.DataFrame(self.model.exog, columns=self.model.exog_names, index=self.dataset.index)
        self.y = pd.Series(self.model.endog, name=self.model.endog_names, index=self.dataset.index)

        self.model = self.model.fit()
        self.model_summary = self.model.get_prediction().summary_frame(alpha=0.05).iloc[self.y.argsort()]
        print(self.model.summary())
        self.is_fitted = True
        
        self.regressionline = sm.OLS(self.model.fittedvalues, sm.add_constant(self.y))
        self.regressionline = self.regressionline.fit()
        self.regressionline_summary = self.regressionline.get_prediction().summary_frame(alpha=0.05).iloc[self.y.argsort()]