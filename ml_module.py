"""
ml_module.py
============
Optional machine-learning module for learning reaction-rate constants
from historical data and feeding predictions back into the simulator.

Supports:
- Linear / polynomial regression (sklearn)
- Random Forest regression
- Neural network (MLPRegressor)
- Feature importance analysis

Input features (environmental conditions) → Target (effective rate constants)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings


def _check_sklearn():
    try:
        import sklearn
        return True
    except ImportError:
        warnings.warn("scikit-learn not installed. ML module unavailable.")
        return False


# ================================================================
# DATA PREPARATION
# ================================================================
def prepare_training_data(historical_records: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert historical observation records into feature matrix X
    and target vector y.

    Each record is a dict with keys:
        'temperature', 'pH', 'DO', 'orp', 'spm', 'biomass',
        'observed_k_eff'  (effective degradation rate constant [1/day])

    Returns
    -------
    X : (n_samples, n_features) array
    y : (n_samples,) array of observed rate constants
    """
    feature_keys = ['temperature', 'pH', 'DO', 'orp', 'spm', 'biomass']
    X = np.array([[r[k] for k in feature_keys] for r in historical_records])
    y = np.array([r['observed_k_eff'] for r in historical_records])
    return X, y


def generate_synthetic_training_data(n_samples: int = 500,
                                     seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic training data based on known kinetic relationships.

    The 'true' rate constant follows:
        k_eff = k_ref · θ^(T-20) · (DO/(K_DO+DO)) · σ(ORP)

    with added noise, simulating historical measurements.
    """
    np.random.seed(seed)

    # Sample environmental conditions
    T = np.random.uniform(5, 30, n_samples)          # °C
    pH = np.random.uniform(6, 9, n_samples)
    DO = np.random.uniform(0.5, 12, n_samples)        # mg/L
    ORP = np.random.uniform(-200, 400, n_samples)      # mV
    SPM = np.random.uniform(5, 200, n_samples)         # mg/L
    biomass = np.random.uniform(1, 30, n_samples)      # mg/L

    # True kinetic model (known relationship)
    k_ref = 0.3
    theta = 1.047
    K_DO = 2.0
    ORP_thresh = 50.0
    beta = 0.02

    temp_factor = theta ** (T - 20.0)
    DO_factor = DO / (K_DO + DO)
    sigma_orp = 1.0 / (1.0 + np.exp(-beta * (ORP - ORP_thresh)))
    k_aer, k_anaer = 0.3, 0.04
    redox_factor = k_aer * sigma_orp + k_anaer * (1.0 - sigma_orp)

    # pH effect (empirical parabolic around pH 7.5)
    pH_factor = 1.0 - 0.1 * (pH - 7.5) ** 2

    k_true = k_ref * temp_factor * DO_factor * redox_factor * pH_factor
    # Biomass scaling
    K_X = 10.0
    k_true *= biomass / (K_X + biomass)

    # Add measurement noise (10% relative + small absolute)
    noise = 0.10 * k_true * np.random.randn(n_samples) + 0.01 * np.random.randn(n_samples)
    k_observed = np.maximum(k_true + noise, 0.001)

    X = np.column_stack([T, pH, DO, ORP, SPM, biomass])
    return X, k_observed


# ================================================================
# MODEL TRAINING
# ================================================================
class RateConstantPredictor:
    """
    Wrapper for ML models that predict reaction rate constants
    from environmental conditions.
    """

    FEATURE_NAMES = ['temperature', 'pH', 'DO', 'orp', 'spm', 'biomass']

    def __init__(self, model_type: str = 'random_forest'):
        """
        Parameters
        ----------
        model_type : 'linear', 'polynomial', 'random_forest', 'neural_network'
        """
        if not _check_sklearn():
            raise ImportError("scikit-learn required for ML module")

        self.model_type = model_type
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.is_fitted = False
        self.feature_importance = None

    def fit(self, X: np.ndarray, y: np.ndarray,
            test_fraction: float = 0.2, verbose: bool = True):
        """
        Train the ML model.

        Parameters
        ----------
        X : (n_samples, 6) feature matrix
        y : (n_samples,) target rate constants
        """
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import r2_score, mean_squared_error

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_fraction, random_state=42)

        # Standardise features
        self.scaler_X = StandardScaler()
        X_train_s = self.scaler_X.fit_transform(X_train)
        X_test_s = self.scaler_X.transform(X_test)

        # Standardise targets
        self.scaler_y = StandardScaler()
        y_train_s = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

        # Build model
        if self.model_type == 'linear':
            from sklearn.linear_model import Ridge
            self.model = Ridge(alpha=1.0)

        elif self.model_type == 'polynomial':
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.pipeline import Pipeline
            from sklearn.linear_model import Ridge
            self.model = Pipeline([
                ('poly', PolynomialFeatures(degree=2, include_bias=False)),
                ('ridge', Ridge(alpha=1.0))
            ])

        elif self.model_type == 'random_forest':
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(
                n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)

        elif self.model_type == 'neural_network':
            from sklearn.neural_network import MLPRegressor
            self.model = MLPRegressor(
                hidden_layer_sizes=(64, 32, 16), activation='relu',
                solver='adam', max_iter=1000, random_state=42,
                early_stopping=True, validation_fraction=0.15)

        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        # Fit
        self.model.fit(X_train_s, y_train_s)
        self.is_fitted = True

        # Evaluate
        y_pred_train = self.scaler_y.inverse_transform(
            self.model.predict(X_train_s).reshape(-1, 1)).ravel()
        y_pred_test = self.scaler_y.inverse_transform(
            self.model.predict(X_test_s).reshape(-1, 1)).ravel()

        metrics = {
            'R2_train': r2_score(y_train, y_pred_train),
            'R2_test': r2_score(y_test, y_pred_test),
            'RMSE_train': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'RMSE_test': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        }

        # Feature importance (if available)
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = dict(
                zip(self.FEATURE_NAMES, self.model.feature_importances_))
        elif self.model_type == 'linear':
            self.feature_importance = dict(
                zip(self.FEATURE_NAMES, np.abs(self.model.coef_)))

        if verbose:
            print(f"ML Model ({self.model_type}) trained:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")
            if self.feature_importance:
                print("  Feature importances:")
                for fname, imp in sorted(self.feature_importance.items(),
                                         key=lambda x: -x[1]):
                    print(f"    {fname}: {imp:.4f}")

        return metrics

    def predict(self, T, pH, DO, orp, spm, biomass):
        """
        Predict rate constant for given environmental conditions.

        Parameters can be scalars or arrays.

        Returns
        -------
        k_predicted : predicted rate constant(s) [1/day]
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        X = np.atleast_2d(np.column_stack([T, pH, DO, orp, spm, biomass]))
        X_s = self.scaler_X.transform(X)
        y_s = self.model.predict(X_s)
        k_pred = self.scaler_y.inverse_transform(y_s.reshape(-1, 1)).ravel()
        return np.maximum(k_pred, 0.0)

    def integrate_into_simulation(self, cfg):
        """
        Replace the fixed k_bio_max in cfg with a dynamic
        ML-predicted value based on current environmental conditions.

        Returns a callable:  predict_k(T, pH, DO, orp, spm, biomass) → k
        """
        return self.predict


def plot_ml_diagnostics(X, y, predictor, savefig=None):
    """
    Plot ML model diagnostics: observed vs predicted,
    feature importance, and residuals.
    """
    import matplotlib.pyplot as plt

    y_pred = predictor.predict(X[:, 0], X[:, 1], X[:, 2],
                               X[:, 3], X[:, 4], X[:, 5])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # (a) Observed vs Predicted
    ax = axes[0]
    ax.scatter(y, y_pred, s=8, alpha=0.5)
    lims = [min(y.min(), y_pred.min()), max(y.max(), y_pred.max())]
    ax.plot(lims, lims, 'r--', linewidth=1)
    ax.set_xlabel('Observed k [1/day]')
    ax.set_ylabel('Predicted k [1/day]')
    ax.set_title('Observed vs Predicted')
    ax.grid(True, alpha=0.3)

    # (b) Residuals
    ax = axes[1]
    residuals = y - y_pred
    ax.hist(residuals, bins=30, edgecolor='white', alpha=0.7)
    ax.axvline(0, color='r', linestyle='--')
    ax.set_xlabel('Residual [1/day]')
    ax.set_ylabel('Count')
    ax.set_title('Residual Distribution')

    # (c) Feature importance
    ax = axes[2]
    if predictor.feature_importance:
        names = list(predictor.feature_importance.keys())
        vals = list(predictor.feature_importance.values())
        order = np.argsort(vals)
        ax.barh([names[i] for i in order], [vals[i] for i in order],
                color='teal')
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importance')

    plt.tight_layout()
    if savefig:
        plt.savefig(savefig)
    plt.show()
