# ht_all_apths_v2.py
from dataclasses import dataclass
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Union, List
import networkx as nx
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
import warnings
from itertools import combinations

# ─────────────────── NEW HELPERS ────────────────────
import math


def z_to_stars(z):
    """Convert a z-score to a 1.0–5.0 numeric score."""
    bins = [0, 1, 2, 3, 4, 5]
    # count how many thresholds z meets or exceeds, then shift to 0–4
    idx = min(sum(z >= b for b in bins) - 1, 4)
    idx = max(idx, 0)
    return float(idx + 1)


def path_score_to_stars(score):
    """Convert a normalized path score (0–1) to a 1.0–5.0 numeric score."""
    bins = [0, 0.3, 0.5, 0.7, 0.9]
    # count how many thresholds score meets or exceeds, then shift to 0–4
    idx = min(sum(score >= b for b in bins) - 1, 4)
    idx = max(idx, 0)
    return float(idx + 1)


# --- END OF NEW FUNCTION ---


# ────────────────── END NEW HELPERS ──────────────────


from base import BaseConfig, BaseRCA, RCAResults


@dataclass
class HTConfig(BaseConfig):
    """
    Configuration class for the HT algorithm.

    Attributes:
        graph: A pandas DataFrame (adjacency matrix) or a filepath (CSV or pickle) representing the causal graph.
        aggregator: Function name for aggregating node scores ("max", "min", or "sum"). Default is "max".
        root_cause_top_k: Maximum number of root causes to return. Default is 3.
        model_type: The model to use for training nodes with parents. Options are:
                    "LinearRegression", "CatBoost", "Xgboost", "LightGBM", "RandomForest", "AutoML".
                    Default is "LinearRegression".
        auto_ml: If True, trains multiple models for each node and selects the best one.
                 Overrides model_type setting. Default is False.
        auto_ml_models: List of model types to test when auto_ml=True.
                       Default is ["LinearRegression", "RandomForest", "Xgboost", "LightGBM"].
    """

    graph: Union[pd.DataFrame, str]
    aggregator: str = "max"
    root_cause_top_k: int = 3
    model_type: str = "LinearRegression"
    auto_ml: bool = False
    auto_ml_models: List[str] = None

    def __post_init__(self):
        """Initialize default auto_ml_models if not provided"""
        if self.auto_ml_models is None:
            self.auto_ml_models = ["LinearRegression", "RandomForest", "Xgboost", "LightGBM"]

        # If model_type is "AutoML", enable auto_ml mode
        if self.model_type.lower() == "automl":
            self.auto_ml = True


class HT(BaseRCA):
    """
    Regression-based Hypothesis Testing (HT) method for Root Cause Analysis.

    This class replicates the HT algorithm from PyRCA in a lightweight, standalone package.
    Instead of using only LinearRegression by default, the model to be used can be specified
    via the configuration parameter `model_type`.
    """

    @staticmethod
    def generate_user_friendly_explanation(node_details):
        """
        Given a list of node_details (baseline vs. abnormal stats),
        return a readable text summary describing what happened.
        """
        explanation_lines = []
        for nd in node_details:
            node_name = nd["node"]
            baseline = nd["baseline_mean"]
            abnormal = nd["abnormal_mean"]
            difference = nd["difference"]
            pct_diff = nd["pct_difference"]

            if difference is None or pct_diff is None:
                line = f"- {node_name}: no baseline or abnormal stats available."
            elif difference > 0:
                line = (
                    f"- {node_name}: {abnormal:.2f} vs. baseline {baseline:.2f} "
                    f"(+{difference:.2f}, +{pct_diff:.1f}%)."
                )
            else:
                line = (
                    f"- {node_name}: {abnormal:.2f} vs. baseline {baseline:.2f} "
                    f"({difference:.2f}, {pct_diff:.1f}%)."
                )
            explanation_lines.append(line)
        return "\n".join(explanation_lines)

    config_class = HTConfig

    def __init__(self, config: HTConfig):
        self.config = config
        # Load the causal graph from file or use directly if DataFrame.
        if isinstance(config.graph, str):
            if config.graph.endswith(".csv"):
                graph = pd.read_csv(config.graph)
            elif config.graph.endswith(".pkl"):
                with open(config.graph, "rb") as f:
                    graph = pickle.load(f)
            else:
                raise RuntimeError("Unsupported graph file format. Use CSV or pickle.")
        else:
            graph = config.graph

        self.adjacency_mat = graph
        # Create a directed graph from the adjacency matrix.
        self.graph = nx.from_pandas_adjacency(graph, create_using=nx.DiGraph())
        # Dictionary to hold (regressor, scaler) for each node.
        self.regressors_dict: Dict[str, List] = {}

    def _get_aggregator(self, name):
        if name == "max":
            return max
        elif name == "min":
            return min
        elif name == "sum":
            return sum
        else:
            raise ValueError(f"Unknown aggregator {name}")

    def _create_model(self):
        """
        Create and return a model instance based on the configuration's model_type.
        Supported options: "LinearRegression", "CatBoost", "Xgboost", "LightGBM", "RandomForest".
        """
        model_name = self.config.model_type.lower()
        if model_name in ("linearregression", "linear"):
            return LinearRegression()
        elif model_name == "randomforest":
            from sklearn.ensemble import RandomForestRegressor

            return RandomForestRegressor()
        elif model_name == "xgboost":
            import xgboost as xgb

            return xgb.XGBRegressor(objective="reg:squarederror")
        elif model_name == "lightgbm":
            import lightgbm as lgb

            return lgb.LGBMRegressor()
        elif model_name == "catboost":
            from catboost import CatBoostRegressor

            return CatBoostRegressor(verbose=0)
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
        
    def _detect_categorical(self, series: pd.Series, threshold_unique: int = 10) -> bool:
        """
        Detect if a series represents a categorical variable.
        
        Args:
            series: Pandas series to check
            threshold_unique: Maximum unique values to consider as categorical
            
        Returns:
            True if categorical, False if continuous
        """
        # Explicit categorical or object types
        if pd.api.types.is_categorical_dtype(series) or pd.api.types.is_object_dtype(series):
            return True
        
        # Numeric but with few unique values (likely ordinal categorical)
        if pd.api.types.is_numeric_dtype(series):
            n_unique = series.nunique()
            n_samples = len(series)
            # Categorical if: few unique values AND less than 5% of total samples
            if n_unique <= threshold_unique and n_unique < n_samples * 0.05:
                return True
        
        return False

    def _create_classifier(self):
        """
        Create a classifier model based on the configuration's model_type.
        Maps regression models to their classification counterparts.
        """
        model_name = self.config.model_type.lower()
        
        if model_name in ("linearregression", "linear"):
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(max_iter=1000, random_state=42)
        
        elif model_name == "randomforest":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(n_estimators=100, random_state=42)
        
        elif model_name == "xgboost":
            import xgboost as xgb
            return xgb.XGBClassifier(
                objective="multi:softmax",
                random_state=42,
                eval_metric='mlogloss'
            )
        
        elif model_name == "lightgbm":
            import lightgbm as lgb
            return lgb.LGBMClassifier(random_state=42, verbose=-1)
        
        elif model_name == "catboost":
            from catboost import CatBoostClassifier
            return CatBoostClassifier(verbose=0, random_state=42)
        
        else:
            # Fallback to LogisticRegression
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(max_iter=1000, random_state=42)

    def _evaluate_regressor(self, model, X, y_true, y_pred, cv_scores=None):
        """
        Evaluate regression model performance.
        
        Returns:
            Dictionary with regression metrics
        """
        metrics = {
            'model_type': 'regression',
            'r2_score': round(r2_score(y_true, y_pred), 4),
            'rmse': round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
            'mae': round(mean_absolute_error(y_true, y_pred), 4),
            'n_samples': len(y_true),
        }
        
        # Add cross-validation score if available
        if cv_scores is not None and len(cv_scores) > 0:
            metrics['cv_r2_mean'] = round(np.mean(cv_scores), 4)
            metrics['cv_r2_std'] = round(np.std(cv_scores), 4)
        
        # Add feature importance if available
        if hasattr(model, 'feature_importances_'):
            metrics['has_feature_importance'] = True
        elif hasattr(model, 'coef_'):
            metrics['has_coefficients'] = True
        
        return metrics

    def _evaluate_classifier(self, model, X, y_true, y_pred, cv_scores=None):
        """
        Evaluate classification model performance.

        Returns:
            Dictionary with classification metrics
        """
        # Handle multiclass vs binary
        n_classes = len(np.unique(y_true))
        avg_method = 'binary' if n_classes == 2 else 'weighted'

        metrics = {
            'model_type': 'classification',
            'accuracy': round(accuracy_score(y_true, y_pred), 4),
            'f1_score': round(f1_score(y_true, y_pred, average=avg_method, zero_division=0), 4),
            'n_samples': len(y_true),
            'n_classes': n_classes,
        }

        # Add cross-validation score if available
        if cv_scores is not None and len(cv_scores) > 0:
            metrics['cv_accuracy_mean'] = round(np.mean(cv_scores), 4)
            metrics['cv_accuracy_std'] = round(np.std(cv_scores), 4)

        # Add feature importance if available
        if hasattr(model, 'feature_importances_'):
            metrics['has_feature_importance'] = True
        elif hasattr(model, 'coef_'):
            metrics['has_coefficients'] = True

        return metrics

    def _create_model_by_name(self, model_name: str, is_classifier: bool = False):
        """
        Create a model instance by name.

        Args:
            model_name: Name of the model type
            is_classifier: Whether to create a classifier (True) or regressor (False)

        Returns:
            Model instance
        """
        # Temporarily override config model_type
        original_model_type = self.config.model_type
        self.config.model_type = model_name

        try:
            if is_classifier:
                model = self._create_classifier()
            else:
                model = self._create_model()
        finally:
            # Restore original model_type
            self.config.model_type = original_model_type

        return model

    def _train_and_select_best_model(
        self,
        X,
        y,
        is_categorical: bool,
        perform_cv: bool = True,
        cv_folds: int = 5
    ):
        """
        Train multiple models and select the best one based on performance.

        Args:
            X: Feature matrix
            y: Target variable
            is_categorical: Whether target is categorical (classification) or continuous (regression)
            perform_cv: Whether to perform cross-validation
            cv_folds: Number of CV folds

        Returns:
            Tuple of (best_model, best_metrics, best_model_name)
        """
        model_names = self.config.auto_ml_models
        best_model = None
        best_metrics = None
        best_model_name = None
        best_score = -np.inf

        results = []

        for model_name in model_names:
            try:
                # Create model
                model = self._create_model_by_name(model_name, is_classifier=is_categorical)

                # Train model
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X, y)

                # Make predictions
                y_pred = model.predict(X)

                # Perform cross-validation if requested
                cv_scores = None
                if perform_cv and X.shape[0] >= cv_folds * 2:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            if is_categorical:
                                cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')
                            else:
                                cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='r2')
                    except Exception:
                        pass

                # Evaluate model
                if is_categorical:
                    metrics = self._evaluate_classifier(model, X, y, y_pred, cv_scores)
                    # Use accuracy for selection
                    score = metrics['accuracy']
                    metric_str = f"Acc: {metrics['accuracy']:.3f}, F1: {metrics['f1_score']:.3f}"
                else:
                    metrics = self._evaluate_regressor(model, X, y, y_pred, cv_scores)
                    # Use R² for selection
                    score = metrics['r2_score']
                    metric_str = f"R²: {metrics['r2_score']:.3f}, RMSE: {metrics['rmse']:.3f}"

                results.append({
                    'model_name': model_name,
                    'model': model,
                    'metrics': metrics,
                    'score': score,
                    'metric_str': metric_str
                })

                # Track best model
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_metrics = metrics
                    best_model_name = model_name

            except Exception as e:
                # Skip models that fail to train
                results.append({
                    'model_name': model_name,
                    'model': None,
                    'metrics': None,
                    'score': -np.inf,
                    'error': str(e)[:100]
                })
                continue

        # Store auto ML results for inspection
        if not hasattr(self, 'auto_ml_results'):
            self.auto_ml_results = {}

        return best_model, best_metrics, best_model_name, results

    def train(self, normal_df: pd.DataFrame, perform_cv: bool = True, cv_folds: int = 5, **kwargs):
        """
        Train regression/classification models for each node based on its parents.
        NOW WITH: Automatic categorical detection, appropriate model selection, and quality metrics.
        
        Args:
            normal_df: Training data (normal/baseline period)
            perform_cv: Whether to perform cross-validation for quality assessment
            cv_folds: Number of folds for cross-validation
        """
        if self.graph is None:
            raise ValueError("Graph is not set.")
        
        # Initialize storage for model quality metrics
        self.node_types = {}  # Store whether each node is categorical or continuous
        self.label_encoders = {}  # Store label encoders for categorical variables
        self.model_metrics = {}  # Store performance metrics for each node
        
        print("=" * 60)
        print("🎓 TRAINING MODELS WITH QUALITY ASSESSMENT")
        print("=" * 60)
        
        # Step 1: Detect variable types and prepare encoders
        print("\n📊 Detecting variable types...")
        for node in list(self.graph):
            if node not in normal_df.columns:
                print(f"Warning: Node '{node}' not found in training data")
                continue
            
            is_categorical = self._detect_categorical(normal_df[node])
            self.node_types[node] = 'categorical' if is_categorical else 'continuous'
            
            # Create label encoder for categorical variables
            if is_categorical:
                le = LabelEncoder()
                le.fit(normal_df[node].dropna())
                self.label_encoders[node] = le
                print(f"   ✓ {node}: CATEGORICAL ({len(le.classes_)} classes: {list(le.classes_)[:5]}...)")
            else:
                print(f"   ✓ {node}: CONTINUOUS")
        
        # Step 2: Train models for each node
        if self.config.auto_ml:
            print(f"\n🤖 Training models (AUTO-ML mode: {len(self.config.auto_ml_models)} models per node)...")
        else:
            print(f"\n🔧 Training models (model_type: {self.config.model_type})...")
        
        for node in list(self.graph):
            if node not in normal_df.columns:
                continue
            
            parents = list(self.graph.predecessors(node))
            is_node_categorical = self.node_types.get(node) == 'categorical'
            
            # Prepare target variable
            if is_node_categorical:
                # Encode categorical target
                y = self.label_encoders[node].transform(normal_df[node].dropna())
                # Get indices of non-null values
                valid_idx = normal_df[node].notna()
            else:
                y = normal_df[node].values
                valid_idx = np.ones(len(y), dtype=bool)
            
            if parents and len(parents) > 0:
                # Node has parents - fit supervised model
                
                # Prepare parent features (encode categorical parents)
                X_list = []
                for parent in parents:
                    if self.node_types.get(parent) == 'categorical':
                        # Encode categorical parent
                        encoded = self.label_encoders[parent].transform(normal_df[parent].dropna())
                        # Handle potential misalignment due to NaN
                        if len(encoded) != len(normal_df):
                            parent_series = normal_df[parent].copy()
                            parent_series_encoded = np.zeros(len(parent_series))
                            parent_series_encoded[parent_series.notna()] = encoded
                            X_list.append(parent_series_encoded)
                        else:
                            X_list.append(encoded)
                    else:
                        X_list.append(normal_df[parent].values)
                
                X = np.column_stack(X_list)
                
                # Filter for valid samples (no NaN in target)
                X = X[valid_idx]
                
                if X.shape[0] < 10:
                    print(f"   ⚠️  {node}: Insufficient samples ({X.shape[0]}), skipping model training")
                    # Still create a scaler for consistency
                    scaler = StandardScaler().fit(y.reshape(-1, 1))
                    self.regressors_dict[node] = [None, scaler]
                    continue
                
                # Determine if we should use Auto ML or single model
                use_auto_ml = self.config.auto_ml

                # Train model(s)
                try:
                    if use_auto_ml:
                        # AUTO ML MODE: Train multiple models and select the best
                        best_model, best_metrics, best_model_name, all_results = self._train_and_select_best_model(
                            X, y, is_node_categorical, perform_cv, cv_folds
                        )

                        # Store results for this node
                        if not hasattr(self, 'auto_ml_results'):
                            self.auto_ml_results = {}
                        self.auto_ml_results[node] = all_results

                        if best_model is None:
                            print(f"   ❌ {node}: All AutoML models failed")
                            # Fallback to simple scaler
                            if is_node_categorical:
                                self.regressors_dict[node] = [None, None]
                            else:
                                scaler = StandardScaler().fit(y.reshape(-1, 1))
                                self.regressors_dict[node] = [None, scaler]
                            continue

                        model = best_model
                        metrics = best_metrics
                        model_type_str = "classifier" if is_node_categorical else "regressor"

                        # Print AutoML results
                        if is_node_categorical:
                            print(f"   ✓ {node}: AUTO-ML → {best_model_name} | Accuracy: {metrics['accuracy']:.3f} | F1: {metrics['f1_score']:.3f}")
                        else:
                            print(f"   ✓ {node}: AUTO-ML → {best_model_name} | R²: {metrics['r2_score']:.3f} | RMSE: {metrics['rmse']:.3f}")

                        # Show other models tested (optional, verbose mode)
                        if kwargs.get('verbose_automl', False):
                            print(f"      Tested models:")
                            for res in all_results:
                                if res.get('model') is not None:
                                    print(f"        - {res['model_name']}: {res['metric_str']}")
                                elif 'error' in res:
                                    print(f"        - {res['model_name']}: Failed ({res['error'][:50]})")

                        y_pred = model.predict(X)

                    else:
                        # SINGLE MODEL MODE: Use specified model_type
                        if is_node_categorical:
                            model = self._create_classifier()
                            model_type_str = "classifier"
                        else:
                            model = self._create_model()
                            model_type_str = "regressor"

                        model.fit(X, y)
                        y_pred = model.predict(X)

                        # Perform cross-validation if requested
                        cv_scores = None
                        if perform_cv and X.shape[0] >= cv_folds * 2:
                            try:
                                with warnings.catch_warnings():
                                    warnings.simplefilter("ignore")
                                    if is_node_categorical:
                                        cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')
                                    else:
                                        cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='r2')
                            except Exception as e:
                                print(f"      ⚠️  CV failed: {str(e)[:50]}")

                        # Evaluate model
                        if is_node_categorical:
                            metrics = self._evaluate_classifier(model, X, y, y_pred, cv_scores)
                            print(f"   ✓ {node}: {model_type_str} trained | Accuracy: {metrics['accuracy']:.3f} | F1: {metrics['f1_score']:.3f}")
                        else:
                            metrics = self._evaluate_regressor(model, X, y, y_pred, cv_scores)
                            print(f"   ✓ {node}: {model_type_str} trained | R²: {metrics['r2_score']:.3f} | RMSE: {metrics['rmse']:.3f}")

                    # Common: Store metrics
                    metrics['node'] = node
                    metrics['parents'] = parents
                    metrics['n_parents'] = len(parents)
                    if use_auto_ml:
                        metrics['selected_model'] = best_model_name
                    self.model_metrics[node] = metrics

                    # Store model appropriately
                    if is_node_categorical:
                        # For categorical, store model directly (no residual scaler needed)
                        self.regressors_dict[node] = [model, None]
                    else:
                        # For continuous, store model + residual scaler
                        residuals = y - y_pred
                        scaler = StandardScaler().fit(residuals.reshape(-1, 1))
                        self.regressors_dict[node] = [model, scaler]
                    
                except Exception as e:
                    print(f"   ❌ {node}: Training failed - {str(e)[:100]}")
                    # Fallback to simple scaler
                    if is_node_categorical:
                        self.regressors_dict[node] = [None, None]
                    else:
                        scaler = StandardScaler().fit(y.reshape(-1, 1))
                        self.regressors_dict[node] = [None, scaler]
                    continue
            
            else:
                # Node has no parents - just fit a scaler (root nodes)
                print(f"   ✓ {node}: Root node (no parents) - baseline scaling only")
                
                if not is_node_categorical:
                    scaler = StandardScaler().fit(y.reshape(-1, 1))
                    self.regressors_dict[node] = [None, scaler]
                else:
                    # Categorical root node - no model needed
                    self.regressors_dict[node] = [None, None]
                
                # Store minimal metrics
                self.model_metrics[node] = {
                    'node': node,
                    'model_type': 'root_node',
                    'parents': [],
                    'n_parents': 0,
                    'n_samples': len(y)
                }
        
        # Step 3: Store baseline statistics (existing logic)
        print("\n📈 Computing baseline statistics...")
        self.baseline_stats = {}
        for node in list(self.graph):
            if node not in normal_df.columns:
                continue
            
            series = normal_df[node]
            
            if self.node_types.get(node) == 'categorical':
                # For categorical, store mode and distribution
                self.baseline_stats[node] = {
                    "type": "categorical",
                    "mode": series.mode()[0] if len(series.mode()) > 0 else None,
                    "distribution": series.value_counts(normalize=True).to_dict(),
                    "count": len(series),
                }
            else:
                # For continuous, store standard stats
                self.baseline_stats[node] = {
                    "type": "continuous",
                    "mean": float(series.mean()),
                    "std": float(series.std()),
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "count": len(series),
                }
        
        # Step 4: Estimate edge elasticities (existing logic, enhanced)
        print("\n🔗 Estimating edge elasticities...")
        self.edge_elasticities = {}
        
        for node_v in list(self.graph):
            parents = list(self.graph.predecessors(node_v))
            if not parents or node_v not in self.regressors_dict:
                continue
            
            model, _ = self.regressors_dict[node_v]
            if model is None:
                continue
            
            # Only calculate elasticities for continuous variables
            if self.node_types.get(node_v) != 'continuous':
                continue
            
            # Extract coefficients/importances
            if isinstance(model, LinearRegression) and hasattr(model, "coef_"):
                # Linear regression: use actual coefficients
                if node_v in self.baseline_stats and all(p in self.baseline_stats for p in parents):
                    mean_v = self.baseline_stats[node_v].get("mean")
                    if mean_v is not None and abs(mean_v) > 1e-9:
                        for i, node_u in enumerate(parents):
                            if i >= len(model.coef_):
                                continue
                            # Skip if parent is categorical (elasticity doesn't make sense)
                            if self.node_types.get(node_u) == 'categorical':
                                continue
                            mean_u = self.baseline_stats[node_u].get("mean")
                            if mean_u is not None:
                                coef_u = model.coef_[i]
                                elasticity = coef_u * (mean_u / mean_v)
                                self.edge_elasticities[(node_u, node_v)] = float(elasticity)
            
            elif hasattr(model, "feature_importances_"):
                # Tree-based models: use feature importances as proxy
                importances = model.feature_importances_
                total_importance = importances.sum()
                
                if total_importance > 0 and node_v in self.baseline_stats:
                    mean_v = self.baseline_stats[node_v].get("mean")
                    if mean_v is not None and abs(mean_v) > 1e-9:
                        for i, node_u in enumerate(parents):
                            if i >= len(importances):
                                continue
                            # Skip if parent is categorical
                            if self.node_types.get(node_u) == 'categorical':
                                continue
                            weight = importances[i] / total_importance
                            mean_u = self.baseline_stats[node_u].get("mean", 1.0)
                            elasticity = weight * (mean_u / mean_v)
                            self.edge_elasticities[(node_u, node_v)] = float(elasticity)
        
        print(f"   ✓ Found {len(self.edge_elasticities)} edge elasticities")
        
        print("\n" + "=" * 60)
        print("✅ TRAINING COMPLETE")
        print("=" * 60)
        print(f"Total nodes: {len(self.graph.nodes())}")
        print(f"Categorical nodes: {sum(1 for t in self.node_types.values() if t == 'categorical')}")
        print(f"Continuous nodes: {sum(1 for t in self.node_types.values() if t == 'continuous')}")
        print(f"Models trained: {len([m for m in self.model_metrics.values() if m.get('model_type') not in ['root_node']])}")
        print("\n💡 Use get_model_quality_report() to see detailed metrics")
        print("=" * 60 + "\n")

    def get_model_quality_report(self, detailed: bool = False) -> Dict:
        """
        Generate a comprehensive quality report for the trained models.
        
        Args:
            detailed: If True, include per-node detailed metrics
            
        Returns:
            Dictionary with quality metrics and trust indicators
        """
        if not hasattr(self, 'model_metrics'):
            return {"error": "No model metrics available. Train the model first."}
        
        report = {
            "overall_summary": {},
            "trust_indicators": {},
            "model_breakdown": {},
        }
        
        # Separate regression and classification metrics
        regression_metrics = [m for m in self.model_metrics.values() if m.get('model_type') == 'regression']
        classification_metrics = [m for m in self.model_metrics.values() if m.get('model_type') == 'classification']
        
        # Overall Summary
        report["overall_summary"] = {
            "total_nodes": len(self.graph.nodes()),
            "nodes_with_models": len(regression_metrics) + len(classification_metrics),
            "regression_nodes": len(regression_metrics),
            "classification_nodes": len(classification_metrics),
            "root_nodes": sum(1 for m in self.model_metrics.values() if m.get('model_type') == 'root_node'),
        }
        
        # Regression Performance
        if regression_metrics:
            r2_scores = [m['r2_score'] for m in regression_metrics if 'r2_score' in m]
            rmse_scores = [m['rmse'] for m in regression_metrics if 'rmse' in m]
            
            report["overall_summary"]["regression_performance"] = {
                "mean_r2": round(np.mean(r2_scores), 4) if r2_scores else None,
                "median_r2": round(np.median(r2_scores), 4) if r2_scores else None,
                "min_r2": round(np.min(r2_scores), 4) if r2_scores else None,
                "max_r2": round(np.max(r2_scores), 4) if r2_scores else None,
                "mean_rmse": round(np.mean(rmse_scores), 4) if rmse_scores else None,
            }
        
        # Classification Performance
        if classification_metrics:
            accuracy_scores = [m['accuracy'] for m in classification_metrics if 'accuracy' in m]
            f1_scores = [m['f1_score'] for m in classification_metrics if 'f1_score' in m]
            
            report["overall_summary"]["classification_performance"] = {
                "mean_accuracy": round(np.mean(accuracy_scores), 4) if accuracy_scores else None,
                "median_accuracy": round(np.median(accuracy_scores), 4) if accuracy_scores else None,
                "min_accuracy": round(np.min(accuracy_scores), 4) if accuracy_scores else None,
                "max_accuracy": round(np.max(accuracy_scores), 4) if accuracy_scores else None,
                "mean_f1": round(np.mean(f1_scores), 4) if f1_scores else None,
            }
        
        # Trust Indicators
        trust_indicators = {
            "graph_coverage": round(
                (len(regression_metrics) + len(classification_metrics)) / len(self.graph.nodes()) * 100, 1
            ),
            "quality_grade": "Not computed",
            "recommendations": [],
        }
        
        # Calculate quality grade
        if regression_metrics:
            avg_r2 = np.mean([m['r2_score'] for m in regression_metrics if 'r2_score' in m])
            if avg_r2 >= 0.8:
                trust_indicators["quality_grade"] = "A (Excellent)"
            elif avg_r2 >= 0.6:
                trust_indicators["quality_grade"] = "B (Good)"
            elif avg_r2 >= 0.4:
                trust_indicators["quality_grade"] = "C (Fair)"
            else:
                trust_indicators["quality_grade"] = "D (Poor)"
                trust_indicators["recommendations"].append("Consider collecting more training data or feature engineering")
        
        if classification_metrics:
            avg_acc = np.mean([m['accuracy'] for m in classification_metrics if 'accuracy' in m])
            if avg_acc >= 0.9:
                trust_indicators["quality_grade_classification"] = "A (Excellent)"
            elif avg_acc >= 0.75:
                trust_indicators["quality_grade_classification"] = "B (Good)"
            elif avg_acc >= 0.6:
                trust_indicators["quality_grade_classification"] = "C (Fair)"
            else:
                trust_indicators["quality_grade_classification"] = "D (Poor)"
        
        # Add recommendations
        if len(self.edge_elasticities) < len(self.graph.edges()) * 0.5:
            trust_indicators["recommendations"].append("Many edges lack elasticity estimates - consider using LinearRegression for better interpretability")
        
        report["trust_indicators"] = trust_indicators
        
        # Model Breakdown (detailed per-node metrics)
        if detailed:
            report["detailed_metrics"] = {}
            for node, metrics in self.model_metrics.items():
                report["detailed_metrics"][node] = metrics
        else:
            # Just include summary by node type
            report["model_breakdown"] = {
                "regression_nodes": [m['node'] for m in regression_metrics],
                "classification_nodes": [m['node'] for m in classification_metrics],
                "root_nodes": [m['node'] for m in self.model_metrics.values() if m.get('model_type') == 'root_node'],
            }
        
        return report

    def print_model_quality_report(self):
        """
        Print a human-readable model quality report to console.
        """
        report = self.get_model_quality_report(detailed=False)
        
        if "error" in report:
            print(report["error"])
            return
        
        print("\n" + "=" * 70)
        print("📊 MODEL QUALITY REPORT")
        print("=" * 70)
        
        # Overall Summary
        summary = report["overall_summary"]
        print(f"\n🎯 OVERALL SUMMARY")
        print(f"   Total Nodes: {summary['total_nodes']}")
        print(f"   Nodes with Models: {summary['nodes_with_models']}")
        print(f"   ├─ Regression: {summary['regression_nodes']}")
        print(f"   ├─ Classification: {summary['classification_nodes']}")
        print(f"   └─ Root Nodes: {summary['root_nodes']}")
        
        # Regression Performance
        if "regression_performance" in summary:
            perf = summary["regression_performance"]
            print(f"\n📈 REGRESSION PERFORMANCE")
            print(f"   Mean R²: {perf['mean_r2']:.4f} (Range: {perf['min_r2']:.4f} - {perf['max_r2']:.4f})")
            print(f"   Median R²: {perf['median_r2']:.4f}")
            print(f"   Mean RMSE: {perf['mean_rmse']:.4f}")
        
        # Classification Performance
        if "classification_performance" in summary:
            perf = summary["classification_performance"]
            print(f"\n🎯 CLASSIFICATION PERFORMANCE")
            print(f"   Mean Accuracy: {perf['mean_accuracy']:.4f} (Range: {perf['min_accuracy']:.4f} - {perf['max_accuracy']:.4f})")
            print(f"   Mean F1-Score: {perf['mean_f1']:.4f}")
        
        # Trust Indicators
        trust = report["trust_indicators"]
        print(f"\n✅ TRUST INDICATORS")
        print(f"   Graph Coverage: {trust['graph_coverage']}%")
        print(f"   Quality Grade: {trust['quality_grade']}")
        if "quality_grade_classification" in trust:
            print(f"   Classification Grade: {trust['quality_grade_classification']}")
        
        if trust["recommendations"]:
            print(f"\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(trust["recommendations"], 1):
                print(f"   {i}. {rec}")
        
        print("\n" + "=" * 70)
        print("✅ Model is ready for root cause analysis and simulation!")
        print("=" * 70 + "\n")

    # -----> PASTE THE COPIED BLOCK STARTING HERE <-----
    def find_root_causes(
        self,
        abnormal_df: pd.DataFrame,
        anomalous_metrics: str = None,
        adjustment: bool = False,
        # return_paths=True, # Argument removed as paths are implicitly handled
        **kwargs,
    ) -> RCAResults:
        """
        Identify root causes from abnormal data using enhanced scoring and pruning.

        For each node, compute a score based on the prediction error on abnormal data.
        Optionally, perform descendant adjustment. If an anomalous metric is provided,
        paths from candidates to the anomaly are found, scored based on causal
        consistency, normalized, and pruned for similarity.
        """
        # --- Initial Checks ---
        if self.graph is None or not self.regressors_dict:
            raise ValueError("Model not trained. Run train() first.")
        if anomalous_metrics and anomalous_metrics not in self.graph:
            raise ValueError(
                f"Anomalous metric '{anomalous_metrics}' not found in graph."
            )
        if anomalous_metrics and anomalous_metrics not in abnormal_df.columns:
            raise ValueError(
                f"Anomalous metric '{anomalous_metrics}' not found in abnormal_df."
            )
        if not hasattr(self, "baseline_stats"):
            raise ValueError(
                "Baseline stats not found. Ensure train() was run successfully."
            )
        # Ensure edge elasticities are initialized, even if empty
        if not hasattr(self, "edge_elasticities"):
            print(
                "Warning: Edge elasticities not calculated during train(). Path scoring quality may be reduced."
            )
            self.edge_elasticities = {}  # Initialize empty if missing

        # === 1. Calculate Node Percentage Changes and Overall Outcome Change ===
        node_pct_changes = {}
        overall_outcome_pct_change = None

        # Calculate for the specified anomalous metric (target node) first
        if anomalous_metrics:
            target_node = anomalous_metrics
            if (
                target_node in self.baseline_stats
                and target_node in abnormal_df.columns
            ):
                baseline_mean = self.baseline_stats[target_node].get("mean")
                # Use mean over the abnormal period for the current value
                abnormal_mean = abnormal_df[target_node].mean()
                if baseline_mean is not None and abs(baseline_mean) > 1e-9:
                    overall_outcome_pct_change = (
                        (abnormal_mean - baseline_mean) / baseline_mean
                    ) * 100
                # Store the outcome change also in the node_pct_changes dict
                node_pct_changes[target_node] = overall_outcome_pct_change
            # else: print(f"Warning: Could not calculate outcome % change for {target_node}")

        # Calculate for all other nodes in the graph
        for node in self.graph.nodes():
            # Avoid recalculating if it's the target node and already done
            if node == anomalous_metrics and node in node_pct_changes:
                continue
            if node in self.baseline_stats and node in abnormal_df.columns:
                baseline_mean = self.baseline_stats[node].get("mean")
                abnormal_mean = abnormal_df[node].mean()
                if baseline_mean is not None and abs(baseline_mean) > 1e-9:
                    pct_change = ((abnormal_mean - baseline_mean) / baseline_mean) * 100
                    node_pct_changes[node] = pct_change
                else:
                    node_pct_changes[node] = (
                        None  # Indicate unreliable change (e.g., baseline is zero)
                    )
            else:
                node_pct_changes[node] = None  # Indicate missing data for this node

        # === 2. Calculate Original Deviation Scores (Z-Scores) ===
        # Keep track of original node scores (deviations) before adjustment
        original_node_scores = {}  # Stores score for each node

        # This part calculates the initial anomaly score for each node based on model residuals
        temp_node_scores_for_adjustment = (
            {}
        )  # Use a temporary dict for adjustment step if needed
        for node in list(self.graph):
            parents = list(self.graph.predecessors(node))
            node_values_abnormal = abnormal_df[node].values
            scores = None  # Initialize scores

            if parents:
                abnormal_x = abnormal_df[parents].values
                if abnormal_x.shape[1] > 0 and node in self.regressors_dict:
                    regressor, scaler = self.regressors_dict[node]
                    if regressor:  # If a model exists
                        abnormal_err = node_values_abnormal - regressor.predict(
                            abnormal_x
                        )
                        if scaler:  # If a scaler exists for residuals
                            scores = scaler.transform(abnormal_err.reshape(-1, 1))[:, 0]
                    elif (
                        scaler
                    ):  # Node has parents but maybe no regressor (e.g., error during training?), use scaler directly on values
                        scores = scaler.transform(node_values_abnormal.reshape(-1, 1))[
                            :, 0
                        ]
                elif (
                    node in self.regressors_dict
                ):  # Node has parents listed, but no data? Fallback to scaling node value directly
                    _, scaler = self.regressors_dict[node]
                    if scaler:
                        scores = scaler.transform(node_values_abnormal.reshape(-1, 1))[
                            :, 0
                        ]

            # Handle nodes with no parents or if residual calculation failed
            if scores is None and node in self.regressors_dict:
                _, scaler = self.regressors_dict[node]
                if scaler:
                    scores = scaler.transform(node_values_abnormal.reshape(-1, 1))[:, 0]

            # If scores were calculated, aggregate them; otherwise, assign a default score (e.g., 0)
            if scores is not None:
                agg_func = self._get_aggregator(self.config.aggregator)
                # Use absolute scores for aggregation as per original logic apparent intent
                score = agg_func(np.abs(scores))
            else:
                score = 0.0  # Default score if calculation failed
                print(f"Warning: Could not calculate score for node {node}")

            # Store the calculated score
            original_node_scores[node] = score
            temp_node_scores_for_adjustment[node] = [
                score,
                0.0,
            ]  # Keep format for adjustment code

        # === 3. Optional: Descendant Adjustment ===
        # This adjusts scores based on scores of downstream nodes
        if adjustment:
            H = self.graph.reverse(copy=True)
            topological_sort = list(nx.topological_sort(H))
            child_nodes = {
                node: list(self.graph.successors(node)) for node in self.graph
            }  # More direct way to get children

            # The adjustment logic might need refinement - original logic had thresholds (e.g., score < 3)
            # This simplified version propagates max child score, adapt if needed
            adjusted_scores = temp_node_scores_for_adjustment.copy()  # Work on a copy
            for node in topological_sort:  # Process nodes in reverse topological order
                node_score = adjusted_scores[node][0]
                # Find max score among children (original logic might be more complex)
                max_child_score = 0
                if node in child_nodes:
                    children = child_nodes[node]
                    if children:
                        max_child_score = max(
                            adjusted_scores[child][0]
                            for child in children
                            if child in adjusted_scores
                        )

                # Apply adjustment (e.g., add max child score - adapt this logic if needed)
                adjusted_scores[node][0] = (
                    node_score + max_child_score
                )  # Example adjustment

            # Update original_node_scores with adjusted scores
            for node in adjusted_scores:
                original_node_scores[node] = adjusted_scores[node][0]

        # === 4. Select Top-K Candidate Root Causes ===
        # Based on the potentially adjusted deviation scores
        root_cause_nodes_candidates = [
            (key, original_node_scores.get(key, 0.0)) for key in original_node_scores
        ]
        # Sort by absolute score magnitude, descending
        root_cause_nodes_candidates = sorted(
            root_cause_nodes_candidates, key=lambda r: abs(r[1]), reverse=True
        )[: self.config.root_cause_top_k]
        candidate_roots = [node for node, score in root_cause_nodes_candidates]

        # === 5. Find Paths from Candidates to Target ===
        # Find all simple paths for the top candidates
        # Note: This can be computationally expensive for dense graphs or long paths!
        root_cause_paths = {}
        if anomalous_metrics is not None:
            print(
                f"Finding paths from {len(candidate_roots)} candidates to {anomalous_metrics}..."
            )
            for root in candidate_roots:
                try:
                    # Consider adding a cutoff if performance is an issue
                    # all_paths = list(nx.all_simple_paths(self.graph, source=root, target=anomalous_metrics, cutoff=10))
                    all_paths = list(
                        nx.all_simple_paths(
                            self.graph, source=root, target=anomalous_metrics
                        )
                    )
                except nx.exception.NetworkXNoPath:
                    all_paths = []  # No path exists
                except nx.exception.NodeNotFound:
                    print(
                        f"Warning: Node '{root}' or '{anomalous_metrics}' not found in graph during path search."
                    )
                    all_paths = []
                # Only add the root if at least one path was found
                if all_paths:
                    root_cause_paths[root] = all_paths
            print(f"Found paths for {len(root_cause_paths)} roots.")

        # === 6. Initial Path Filtering (Optional, keep if desired) ===
        # Your original code included filtering for subsequences/redundancy *before* scoring.
        # This can be kept, but be aware it might remove variations before scoring.
        # Keeping the placeholder for your original logic here.
        # Define is_subsequence if you use it:
        def is_subsequence(short, long):
            # Keep your original implementation here
            # Example simple check (contiguous subsequence):
            if len(short) >= len(long):
                return False
            for i in range(len(long) - len(short) + 1):
                if long[i : i + len(short)] == short:
                    return True
            return False

        # Apply your original filtering logic if needed (operates on root_cause_paths)
        # Example:
        # filtered_paths = {}
        # for root, paths in root_cause_paths.items():
        #     # ... your filtering logic using is_subsequence ...
        #     filtered_paths[root] = # result of filtering
        # root_cause_paths = filtered_paths # Update with filtered results

        # print(f"Paths remaining after initial filtering: {sum(len(p) for p in root_cause_paths.values())}")

        # === 7. Calculate Enhanced Path Scores and Prepare for Pruning ===
        path_details_scored = {}  # Stores { root: [ { path: [], raw_score: float } ] }
        all_raw_scores = []  # Collect all raw scores for normalization range

        # Use the roots that currently have paths associated with them
        active_roots = list(root_cause_paths.keys())

        print("Calculating enhanced scores for paths...")
        for root in active_roots:
            if root not in root_cause_paths:
                continue  # Safety check

            path_details_scored[root] = []
            # Root strength: Use the absolute deviation score (potentially adjusted)
            root_strength = abs(
                original_node_scores.get(root, 1.0)
            )  # Default to 1 if score missing

            for path in root_cause_paths[root]:
                if not path or len(path) < 2:
                    continue  # Path needs at least one edge

                # Calculate Path Plausibility Score (PPS)
                path_plausibility = 1.0
                accumulated_influence_sign = (
                    1.0  # Tracks sign changes along the path based on elasticity
                )

                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    pct_change_u = node_pct_changes.get(u)
                    pct_change_v = node_pct_changes.get(v)
                    # Fetch elasticity, default to 0 if missing (implies neutral influence)
                    elasticity = self.edge_elasticities.get((u, v), 0.0)

                    # Check for sign consistency only if data is available
                    if pct_change_u is not None and pct_change_v is not None:
                        # Check only if elasticity and cause change are non-negligible
                        if abs(elasticity) > 1e-6 and abs(pct_change_u) > 1e-6:
                            expected_sign_v = np.sign(elasticity * pct_change_u)
                            observed_sign_v = np.sign(pct_change_v)
                            # Penalize if signs are defined and opposite
                            if (
                                expected_sign_v != 0
                                and observed_sign_v != 0
                                and expected_sign_v != observed_sign_v
                            ):
                                path_plausibility *= 0.1  # Heavy penalty
                        # Update accumulated sign based on elasticity sign
                        accumulated_influence_sign *= (
                            np.sign(elasticity) if abs(elasticity) > 1e-6 else 1.0
                        )
                    else:
                        path_plausibility *= (
                            0.8  # Penalize slightly for missing data along path
                        )

                # Final Check: Compare overall path direction with outcome direction
                pct_change_root = node_pct_changes.get(path[0])
                if (
                    overall_outcome_pct_change is not None
                    and pct_change_root is not None
                    and abs(pct_change_root) > 1e-6
                ):
                    expected_overall_sign = np.sign(
                        pct_change_root * accumulated_influence_sign
                    )
                    observed_overall_sign = np.sign(overall_outcome_pct_change)
                    if (
                        expected_overall_sign != 0
                        and observed_overall_sign != 0
                        and expected_overall_sign != observed_overall_sign
                    ):
                        path_plausibility *= 0.5  # Penalize inconsistency with outcome

                # Path Length Penalty (gentle)
                path_length_penalty = np.exp(-0.05 * (len(path) - 1))

                # Combine scores: Root Strength * Plausibility * Length Penalty
                # Ensure plausibility doesn't go negative due to penalties
                final_path_score_raw = (
                    root_strength * max(0.0, path_plausibility) * path_length_penalty
                )

                path_details_scored[root].append(
                    {
                        "path": path,
                        "raw_score": final_path_score_raw,
                        # normalized_score will be added next
                    }
                )
                # Collect scores for normalization range finding
                if (
                    final_path_score_raw > 1e-9
                ):  # Avoid near-zero scores affecting range badly
                    all_raw_scores.append(final_path_score_raw)

        # === 8. Normalize Path Scores (Min-Max Scaling to 0-1) ===
        min_score = min(all_raw_scores) if all_raw_scores else 0
        max_score = max(all_raw_scores) if all_raw_scores else 0
        score_range = max_score - min_score
        print(f"Normalizing scores (Range: {min_score:.4f} - {max_score:.4f})")

        for root in path_details_scored:
            for path_info in path_details_scored[root]:
                raw_score = path_info["raw_score"]
                if score_range > 1e-9:  # Avoid division by zero if all scores are same
                    normalized_score = (raw_score - min_score) / score_range
                elif (
                    max_score > 1e-9
                ):  # Handle case where all scores are the same positive value
                    normalized_score = 1.0  # Max score is the score, maps to 1.0
                else:  # Handle case where all scores are zero or negative
                    normalized_score = 0.0
                # Clip score to ensure it's strictly within [0, 1]
                path_info["normalized_score"] = max(
                    0.0, min(1.0, round(normalized_score, 4))
                )

        # === 9. Prune Similar Paths (Based on Jaccard Similarity) ===

        final_pruned_paths_details = (
            {}
        )  # Stores { root: [ {path, raw_score, normalized_score}, ... ] }
        jaccard_threshold = 0.8  # Threshold for similarity (e.g., 80% node overlap)

        # Define Jaccard helper function (can be moved to class scope if preferred)
        def calculate_jaccard(set1, set2):
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            return intersection / union if union > 0 else 0

        print(f"Pruning paths with Jaccard threshold > {jaccard_threshold}...")
        for root, scored_paths_list in path_details_scored.items():
            # Sort by normalized_score descending to prioritize keeping better paths
            sorted_paths_info = sorted(
                scored_paths_list, key=lambda x: x["normalized_score"], reverse=True
            )
            # --- The Jaccard filtering below will now operate on 'paths_after_subpath_filter' ---

            selected_paths_for_root = (
                []
            )  # Stores the info dictionaries of selected paths
            selected_path_sets = (
                []
            )  # Stores sets of nodes for faster Jaccard calculation

            for path_info in sorted_paths_info:
                current_path = path_info["path"]
                current_path_set = set(
                    current_path
                )  # Convert path to set for comparison
                is_redundant = False
                for existing_path_set in selected_path_sets:
                    # Check similarity with already selected paths for THIS root
                    if (
                        calculate_jaccard(current_path_set, existing_path_set)
                        > jaccard_threshold
                    ):
                        is_redundant = True
                        break  # Found a similar path already selected, discard current one

                if not is_redundant:
                    selected_paths_for_root.append(
                        path_info
                    )  # Keep this path's details
                    selected_path_sets.append(
                        current_path_set
                    )  # Add its node set for future comparison

            # Only add the root to the final results if it still has paths after pruning
            if selected_paths_for_root:
                final_pruned_paths_details[root] = selected_paths_for_root

        print(
            f"Paths remaining after pruning: {sum(len(p) for p in final_pruned_paths_details.values())}"
        )

        # === 10. Final Output Formatting (Matching RCAResults Structure) ===

        # 1. Filter root cause nodes list based on roots surviving pruning
        surviving_roots = set(final_pruned_paths_details.keys())
        # Get the original candidates again and filter
        final_root_cause_nodes_tuples = [
            (node, score)
            for node, score in root_cause_nodes_candidates  # Use the original top-k list
            if node in surviving_roots  # Keep only if they still have paths
        ]
        # Ensure we don't exceed top-k AFTER filtering
        final_root_cause_nodes_tuples = final_root_cause_nodes_tuples[
            : self.config.root_cause_top_k
        ]

        # Format for RCAResults internal list: [{"root_cause": name, "score": original_deviation_score}, ...]
        final_root_cause_nodes_list = [
            {
                "root_cause": node,
                "score": round(score, 2),
                "severity": z_to_stars(score),
            }  # Use the original deviation score here
            for node, score in final_root_cause_nodes_tuples
        ]

        # 2. Format root_cause_paths for RCAResults (for the final JSON)
        # Format: { root: [ {"path": [...], "score": normalized_score}, ... ] }
        final_formatted_paths = {}
        for root, path_infos_list in final_pruned_paths_details.items():
            # Ensure this root is actually in the final list of root causes
            if root in [rc["root_cause"] for rc in final_root_cause_nodes_list]:
                final_formatted_paths[root] = [
                    {
                        "path": p_info["path"],
                        "score": round(p_info["normalized_score"], 2),
                        "path_severity": path_score_to_stars(
                            p_info["normalized_score"]
                        ),
                    }  # Use normalized score
                    for p_info in path_infos_list
                ]

        # 3. Prepare path_explanations (Internal structure for RCAResults)
        # Format: { root: [ {"path": ..., "path_score": normalized_score, "summary": ...}, ...] }
        final_path_explanations = {}
        # Generate summary data once for efficiency
        # Need to call the helper method generate_node_abnormal_summary
        # Ensure it's available in the class (Add it in Step 3 below if not already done)
        temp_abnormal_summary = self.generate_node_abnormal_summary(abnormal_df)

        for root, path_infos_list in final_pruned_paths_details.items():
            if root in final_formatted_paths:  # Only generate for roots we are keeping
                explanations_for_root = []
                for p_info in path_infos_list:
                    path = p_info["path"]
                    node_details = []
                    for node in path:
                        # Build details needed for explanation string from the summary
                        node_summary = temp_abnormal_summary.get(node, {})
                        node_details.append(
                            {
                                "node": node,
                                "baseline_mean": node_summary.get("baseline_mean"),
                                "abnormal_mean": node_summary.get("abnormal_mean"),
                                "difference": node_summary.get("difference"),
                                "pct_difference": node_summary.get("pct_difference"),
                            }
                        )
                    # Use the static method from HT class to generate the text
                    summary_text = HT.generate_user_friendly_explanation(node_details)
                    explanations_for_root.append(
                        {
                            "path": path,
                            "path_score": p_info[
                                "normalized_score"
                            ],  # Use normalized score
                            "summary": summary_text,
                        }
                    )
                final_path_explanations[root] = explanations_for_root

        # 4. Get Node Abnormal Summary for the final output
        # This uses the same helper method called above
        node_abnormal_summary = temp_abnormal_summary  # Reuse the calculated summary

        # 5. Return the RCAResults object
        print("RCA finished.")
        return RCAResults(
            root_cause_nodes=final_root_cause_nodes_list,  # List of {'root_cause': name, 'score': deviation_score}
            root_cause_paths=final_formatted_paths,  # Dict of {root: [{'path': path, 'score': normalized_score}]}
            path_explanations=final_path_explanations,  # Internal dict with summaries and normalized scores
            outcome_pct_change=overall_outcome_pct_change,  # Overall % change of the target
            node_abnormal_summary=node_abnormal_summary,  # Dict of {node: {stats}}
        )

    # === NEW: Helper method to generate node summary (add this to HT class) ===
    def generate_node_abnormal_summary(self, abnormal_df):
        """Generates baseline vs abnormal stats for all nodes."""
        summary = {}
        if not hasattr(self, "baseline_stats"):
            print("Warning: Cannot generate abnormal summary, baseline_stats missing.")
            return summary  # Return empty if no baseline data

        # Ensure baseline_stats is a dictionary
        if not isinstance(self.baseline_stats, dict):
            print("Warning: baseline_stats is not a dictionary.")
            return summary

        # Iterate through nodes known in the graph
        for node in self.graph.nodes():
            # Check if baseline stats exist for this node AND data exists in abnormal_df
            if node in self.baseline_stats and node in abnormal_df.columns:
                # Safely get baseline mean using .get() with a default of None
                baseline_mean = self.baseline_stats[node].get("mean")

                # Calculate mean for the abnormal period passed to find_root_causes
                # Check if abnormal data for the node is not empty or all NaN
                if not abnormal_df[node].isnull().all():
                    abnormal_mean = abnormal_df[node].mean()
                else:
                    # Handle cases where abnormal data is missing/NaN for this node
                    abnormal_mean = None
                    print(
                        f"Warning: Abnormal data for node '{node}' is missing or all NaN."
                    )
                    continue  # Skip calculation for this node if no abnormal mean

                diff = None
                pct_diff = None

                # Calculate diff and pct_diff only if both means are valid numbers
                if baseline_mean is not None and abnormal_mean is not None:
                    # Check they are numbers (though .mean() should return numbers or NaN)
                    if isinstance(baseline_mean, (int, float)) and isinstance(
                        abnormal_mean, (int, float)
                    ):
                        diff = abnormal_mean - baseline_mean
                        # Calculate percentage difference only if baseline_mean is not near zero
                        if abs(baseline_mean) > 1e-9:
                            pct_diff = (diff / baseline_mean) * 100
                        else:
                            pct_diff = None  # Avoid division by zero
                    else:
                        diff = None  # One of the means wasn't a number
                        pct_diff = None

                # Add results to summary, rounding where appropriate
                summary[node] = {
                    "baseline_mean": (
                        round(baseline_mean, 3) if baseline_mean is not None else None
                    ),
                    "abnormal_mean": (
                        round(abnormal_mean, 3) if abnormal_mean is not None else None
                    ),
                    "difference": round(diff, 3) if diff is not None else None,
                    "pct_difference": (
                        round(pct_diff, 1) if pct_diff is not None else None
                    ),
                }
            # else: Node missing baseline or abnormal data, skip summary for it
        return summary

class CausalSimulator:
    """
    Performs causal simulations and what-if analysis on a trained HT model.
    
    This class allows you to intervene on variables and see how changes
    propagate through the causal graph to affect outcome variables.
    """
    
    def __init__(self, trained_ht_model: HT):
        """
        Initialize the simulator with a trained HT model.
        
        Args:
            trained_ht_model: A trained HT model instance
        """
        if not trained_ht_model.regressors_dict:
            raise ValueError("The HT model must be trained before simulation!")
        
        self.ht_model = trained_ht_model
        self.graph = trained_ht_model.graph
        self.regressors_dict = trained_ht_model.regressors_dict
        self.baseline_stats = trained_ht_model.baseline_stats
    
    def _propagate_through_graph(self, initial_values: Dict[str, float], intervened_nodes: set = None) -> Dict[str, float]:
        """
        Propagate values through the causal graph.
        
        Args:
            initial_values: Starting values for all nodes
            intervened_nodes: Set of nodes that are intervened on (their values are fixed)
            
        Returns:
            Dictionary of predicted values for all nodes
        """
        if intervened_nodes is None:
            intervened_nodes = set()
        
        # Start with initial values
        predicted_values = initial_values.copy()
        
        # Get topological order to process nodes correctly
        try:
            topological_order = list(nx.topological_sort(self.graph))
        except nx.NetworkXError:
            raise ValueError("Graph contains cycles! Cannot perform causal simulation on cyclic graphs.")
        
        # Process each node in topological order
        for node in topological_order:
            # Skip intervened nodes - their values are fixed
            if node in intervened_nodes:
                continue
            
            parents = list(self.graph.predecessors(node))
            
            if not parents:
                # No parents, keep initial value
                continue
            
            # Get the regression model for this node
            if node not in self.regressors_dict:
                continue
                
            regressor, scaler = self.regressors_dict[node]
            
            if regressor is None:
                # Node has no trained model, keep initial value
                continue
            
            # Get parent values (after propagation)
            parent_values = np.array([[predicted_values[p] for p in parents]])
            
            # Predict the new value for this node
            try:
                # Check if node is categorical
                is_categorical = self.ht_model.node_types.get(node) == 'categorical'
                
                if is_categorical:
                    # For categorical: predict class and decode
                    predicted_class = regressor.predict(parent_values)[0]
                    # Use the numeric encoded value for propagation
                    predicted_value = float(predicted_class)
                else:
                    # For continuous: direct prediction
                    predicted_value = regressor.predict(parent_values)[0]
                
                predicted_values[node] = predicted_value
                
            except Exception as e:
                print(f"Warning: Could not predict value for node {node}: {e}")
                continue
        
        return predicted_values
    
    def simulate_intervention(
        self,
        interventions: Union[Dict[str, float], Dict[str, Dict[str, float]]],
        outcome_nodes: Union[str, List[str]] = None,
        baseline_df: pd.DataFrame = None
    ) -> Dict:
        """
        Simulate what happens when you intervene on one or more nodes.
        
        This properly implements causal inference by:
        1. Simulating the "natural" world (no intervention)
        2. Simulating the world with intervention(s)
        3. Computing the difference (true causal effect)
        
        Args:
            interventions: Dictionary of interventions. Can be:
                - Simple: {'node_name': new_value}
                - Detailed: {'node_name': {'value': X} or {'pct_change': Y}}
            outcome_nodes: Node(s) to track. Can be single string or list. 
                          If None, tracks all nodes.
            baseline_df: Baseline data to use (optional, uses training baseline if not provided)
            
        Returns:
            Dictionary with simulation results including:
            - interventions: Dict of what was changed
            - world_without_intervention: Predicted values with no intervention
            - world_with_intervention: Predicted values with intervention
            - causal_effects: The difference (intervention - no_intervention)
            - outcome_effects: Detailed effects on specified outcome nodes
            
        Examples:
            # Single intervention, percentage change
            result = simulator.simulate_intervention(
                interventions={'CPU_usage': {'pct_change': 20}},
                outcome_nodes='response_time'
            )
            
            # Single intervention, absolute value
            result = simulator.simulate_intervention(
                interventions={'CPU_usage': {'value': 75}},
                outcome_nodes=['response_time', 'error_rate']
            )
            
            # Multiple interventions
            result = simulator.simulate_intervention(
                interventions={
                    'CPU_usage': {'pct_change': 20},
                    'Memory_usage': {'value': 80}
                },
                outcome_nodes=['response_time', 'throughput']
            )
            
            # Simple format (just values)
            result = simulator.simulate_intervention(
                interventions={'CPU_usage': 75, 'Memory_usage': 80},
                outcome_nodes='response_time'
            )
        """
        # === Step 1: Parse and Validate Interventions ===
        parsed_interventions = {}
        
        for node, intervention in interventions.items():
            if node not in self.graph.nodes():
                raise ValueError(f"Intervention node '{node}' not found in graph!")
            
            # Handle different intervention formats
            if isinstance(intervention, (int, float)):
                # Simple format: just a value
                parsed_interventions[node] = {'value': float(intervention)}
            elif isinstance(intervention, dict):
                # Detailed format
                if 'value' in intervention and 'pct_change' in intervention:
                    raise ValueError(f"For node '{node}': specify either 'value' or 'pct_change', not both!")
                parsed_interventions[node] = intervention
            else:
                raise ValueError(f"Invalid intervention format for node '{node}'")
        
        # === Step 2: Validate Outcome Nodes ===
        if outcome_nodes is None:
            # Track all nodes
            outcome_nodes_list = list(self.graph.nodes())
        elif isinstance(outcome_nodes, str):
            # Single outcome node
            if outcome_nodes not in self.graph.nodes():
                raise ValueError(f"Outcome node '{outcome_nodes}' not found in graph!")
            outcome_nodes_list = [outcome_nodes]
        else:
            # Multiple outcome nodes
            outcome_nodes_list = list(outcome_nodes)
            for node in outcome_nodes_list:
                if node not in self.graph.nodes():
                    raise ValueError(f"Outcome node '{node}' not found in graph!")
        
        # === Step 3: Get Baseline Values ===
        if baseline_df is not None:
            baseline_values = {
                node: baseline_df[node].mean() 
                for node in self.graph.nodes() 
                if node in baseline_df.columns
            }
        else:
            baseline_values = {
                node: stats.get('mean', 0) 
                for node, stats in self.baseline_stats.items()
            }
        
        # === Step 4: Calculate Intervention Values ===
        intervention_values = {}
        intervention_details = {}
        
        for node, intervention_spec in parsed_interventions.items():
            baseline_value = baseline_values.get(node, 0)
            
            if 'value' in intervention_spec:
                # Absolute value specified
                new_value = intervention_spec['value']
            elif 'pct_change' in intervention_spec:
                # Percentage change specified
                pct_change = intervention_spec['pct_change']
                new_value = baseline_value * (1 + pct_change / 100)
            else:
                raise ValueError(f"Must specify either 'value' or 'pct_change' for node '{node}'")
            
            intervention_values[node] = new_value
            
            # Store details
            absolute_change = new_value - baseline_value
            pct_change_actual = None
            if abs(baseline_value) > 1e-9:
                pct_change_actual = (absolute_change / baseline_value) * 100
            
            intervention_details[node] = {
                'baseline': round(baseline_value, 4),
                'intervention_value': round(new_value, 4),
                'absolute_change': round(absolute_change, 4),
                'pct_change': round(pct_change_actual, 2) if pct_change_actual is not None else None
            }
        
        # === Step 5: Simulate World WITHOUT Intervention (Counterfactual) ===
        world_no_intervention = self._propagate_through_graph(
            initial_values=baseline_values,
            intervened_nodes=set()  # No intervention
        )
        
        # === Step 6: Simulate World WITH Intervention ===
        # Start with baseline values
        initial_with_intervention = baseline_values.copy()
        # Override with intervention values
        for node, value in intervention_values.items():
            initial_with_intervention[node] = value
        
        world_with_intervention = self._propagate_through_graph(
            initial_values=initial_with_intervention,
            intervened_nodes=set(intervention_values.keys())  # Fix these nodes
        )
        
        # === Step 7: Calculate Causal Effects (Difference) ===
        causal_effects = {}
        
        for node in self.graph.nodes():
            no_intervention_value = world_no_intervention.get(node, 0)
            with_intervention_value = world_with_intervention.get(node, 0)
            
            absolute_effect = with_intervention_value - no_intervention_value
            
            pct_effect = None
            if abs(no_intervention_value) > 1e-9:
                pct_effect = (absolute_effect / no_intervention_value) * 100
            
            causal_effects[node] = {
                'no_intervention': round(no_intervention_value, 4),
                'with_intervention': round(with_intervention_value, 4),
                'absolute_effect': round(absolute_effect, 4),
                'pct_effect': round(pct_effect, 2) if pct_effect is not None else None
            }
        
        # === Step 8: Extract Outcome-Specific Effects ===
        outcome_effects = {}
        
        for outcome_node in outcome_nodes_list:
            outcome_effects[outcome_node] = causal_effects[outcome_node]
        
        # === Step 9: Calculate Elasticities (for single outcome) ===
        elasticities = {}
        
        if len(outcome_nodes_list) == 1:
            outcome_node = outcome_nodes_list[0]
            outcome_pct_effect = causal_effects[outcome_node].get('pct_effect')
            
            if outcome_pct_effect is not None:
                for interv_node, interv_info in intervention_details.items():
                    interv_pct_change = interv_info.get('pct_change')
                    if interv_pct_change is not None and abs(interv_pct_change) > 1e-9:
                        elasticity = outcome_pct_effect / interv_pct_change
                        elasticities[f"{interv_node}_to_{outcome_node}"] = round(elasticity, 4)
        
        # === Step 10: Build Result ===
        result = {
            'interventions': intervention_details,
            'world_without_intervention': {k: round(v, 4) for k, v in world_no_intervention.items()},
            'world_with_intervention': {k: round(v, 4) for k, v in world_with_intervention.items()},
            'causal_effects': causal_effects,
            'outcome_effects': outcome_effects,
        }
        
        if elasticities:
            result['elasticities'] = elasticities
        
        return result
    
    def compare_scenarios(
        self,
        scenarios: List[Dict[str, Union[float, Dict]]],
        outcome_nodes: Union[str, List[str]] = None,
        scenario_names: List[str] = None
    ) -> pd.DataFrame:
        """
        Compare multiple intervention scenarios.
        
        Args:
            scenarios: List of intervention dictionaries (same format as simulate_intervention)
            outcome_nodes: Node(s) to focus on in comparison
            scenario_names: Optional names for scenarios
            
        Returns:
            DataFrame comparing all scenarios
            
        Example:
            scenarios = [
                {'CPU_usage': {'pct_change': 10}},
                {'CPU_usage': {'pct_change': 20}},
                {'CPU_usage': {'pct_change': 30}},
                {'CPU_usage': {'pct_change': 20}, 'Memory_usage': {'pct_change': 15}},
            ]
            comparison = simulator.compare_scenarios(
                scenarios, 
                outcome_nodes=['response_time', 'throughput']
            )
        """
        if scenario_names is None:
            scenario_names = [f"Scenario_{i+1}" for i in range(len(scenarios))]
        
        if len(scenario_names) != len(scenarios):
            raise ValueError("Number of scenario_names must match number of scenarios!")
        
        results = []
        
        for i, (scenario, name) in enumerate(zip(scenarios, scenario_names)):
            try:
                result = self.simulate_intervention(
                    interventions=scenario,
                    outcome_nodes=outcome_nodes
                )
                
                # Build row for this scenario
                row = {'scenario': name}
                
                # Add intervention details
                for node, details in result['interventions'].items():
                    row[f'{node}_intervention'] = details['pct_change']
                    row[f'{node}_new_value'] = details['intervention_value']
                
                # Add outcome effects
                for node, effects in result['outcome_effects'].items():
                    row[f'{node}_effect'] = effects['absolute_effect']
                    row[f'{node}_pct_effect'] = effects['pct_effect']
                
                # Add elasticities if available
                if 'elasticities' in result:
                    for key, value in result['elasticities'].items():
                        row[f'elasticity_{key}'] = value
                
                results.append(row)
                
            except Exception as e:
                print(f"Error in {name}: {e}")
                continue
        
        return pd.DataFrame(results)
    
    def _estimate_prediction_interval(self, intervention_dict, outcome_node, confidence_level=0.90):
        """
        Estimate prediction interval using model uncertainty.
        
        Args:
            intervention_dict: Dictionary of interventions
            outcome_node: Target outcome node
            confidence_level: Confidence level (0.90 or 0.95)
            
        Returns:
            Tuple of (lower_margin, upper_margin) or (None, None) if unavailable
        """
        if not hasattr(self.ht_model, 'model_metrics'):
            return None, None
        
        # Use model RMSE as uncertainty proxy
        node_metrics = self.ht_model.model_metrics.get(outcome_node)
        if not node_metrics or node_metrics.get('model_type') != 'regression':
            return None, None
        
        rmse = node_metrics.get('rmse', 0)
        if rmse == 0:
            return None, None
        
        # Use z-score based on confidence level
        # 90% CI: ±1.645*RMSE, 95% CI: ±1.96*RMSE
        z_score = 1.645 if confidence_level == 0.90 else 1.96
        margin = z_score * rmse
        
        return margin, margin  # Symmetric bounds


    def find_best_intervention(
        self,
        outcome_node: str,
        target_change: float,
        candidate_nodes: List[str] = None,
        max_intervention_pct: float = 50,
        allow_combinations: bool = False,
        tolerance: float = 2.0,
        tolerance_relative: float = None,
        max_search_iterations: int = 15,
        min_model_quality: float = 0.5,
        enable_uncertainty: bool = True,
    ) -> Dict:
        """
        Grade-A Production Method for finding interventions with enhanced robustness.
        
        Fortune 100 Ready: Simple, Robust, Validated.
        
        Key Features:
        - Iterative search finds actual working interventions
        - Adaptive refinement handles non-linear relationships
        - Model quality gates fail fast if models aren't reliable
        - Zero-effect filtering removes spurious results
        - Uncertainty quantification provides confidence intervals
        - Direct validation tests what you'll actually implement
        
        Args:
            outcome_node: Outcome variable to change
            target_change: Desired % change (e.g., -10 for 10% reduction)
            candidate_nodes: Nodes to test (None = all ancestors)
            max_intervention_pct: Maximum allowed change per variable
            allow_combinations: Test 2-node combinations
            tolerance: Acceptable error in ABSOLUTE percentage points (default: ±2.0%)
                Example: target=-10%, tolerance=2.0 → accept -8% to -12%
            tolerance_relative: Alternative relative tolerance (% of target)
                Example: target=-10%, tolerance_relative=20 → accept -8% to -12%
                If specified, overrides `tolerance`
            max_search_iterations: Maximum refinement iterations
            min_model_quality: Minimum model R² to attempt intervention
            enable_uncertainty: Calculate prediction intervals (requires model metrics)
            
        Returns:
            Dict with validated, business-ready recommendations including:
            - best_intervention: Top-ranked option with details
            - all_options: Sorted list of all feasible interventions
            - summary: Aggregate statistics
            - recommendation: Clear guidance for implementation
            
        Example:
            >>> best = simulator.find_best_intervention(
            ...     outcome_node='total_delam_waste',
            ...     target_change=-10,
            ...     tolerance=2.0,  # Accept -8% to -12%
            ...     max_intervention_pct=30,
            ...     allow_combinations=True
            ... )
        """
        
        # ============================================================
        # STEP 0: PRE-FLIGHT CHECKS
        # ============================================================
        
        print(f"\n{'='*70}")
        print(f"🎯 GRADE-A INTERVENTION SEARCH (v2.0)")
        print(f"{'='*70}")
        print(f"Target: {target_change:+.1f}% change in {outcome_node}")
        
        if outcome_node not in self.graph.nodes():
            raise ValueError(f"Outcome node '{outcome_node}' not found!")
        
        # Get candidates
        if candidate_nodes is None:
            candidate_nodes = list(nx.ancestors(self.graph, outcome_node))
            if not candidate_nodes:
                raise ValueError(f"No ancestor nodes found for '{outcome_node}'!")
        
        # Calculate acceptable error with clearer semantics
        if tolerance_relative is not None:
            # Use relative tolerance (% of target)
            acceptable_error = abs(target_change) * (tolerance_relative / 100.0)
            print(f"Tolerance: ±{tolerance_relative}% of target (±{acceptable_error:.1f}% points)")
        else:
            # Use absolute tolerance (percentage points) - more intuitive default
            acceptable_error = tolerance
            print(f"Tolerance: ±{tolerance}% points")
        
        print(f"Max intervention: ±{max_intervention_pct}%")
        print(f"Acceptable range: {target_change - acceptable_error:+.1f}% to {target_change + acceptable_error:+.1f}%")
        
        # Check model quality FIRST
        print(f"\n📊 Pre-flight: Checking model quality...")
        
        quality_issues = []
        if hasattr(self.ht_model, 'model_metrics'):
            for node in candidate_nodes:
                if node in self.ht_model.model_metrics:
                    metrics = self.ht_model.model_metrics[node]
                    if metrics.get('model_type') == 'regression':
                        r2 = metrics.get('r2_score', 0)
                        if r2 < min_model_quality:
                            quality_issues.append(f"{node}: R²={r2:.2f}")
            
            if quality_issues:
                print(f"   ⚠️  Low quality models detected:")
                for issue in quality_issues[:5]:
                    print(f"      - {issue}")
                if len(quality_issues) > 5:
                    print(f"      ... and {len(quality_issues) - 5} more")
                print(f"   ⚠️  Results may be unreliable")
        
        # ============================================================
        # STEP 1: DIRECT SEARCH WITH VALIDATION
        # ============================================================
        
        print(f"\n🔍 Testing {len(candidate_nodes)} candidates with DIRECT search...")
        
        results = []
        
        for node in candidate_nodes:
            if node == outcome_node:
                continue
            
            print(f"\n   Testing: {node}")
            
            # ====== CHECK FOR CAUSAL PATH ======
            has_path = nx.has_path(self.graph, node, outcome_node)
            if not has_path:
                print(f"      ✗ No causal path to outcome - skipping")
                continue
            
            # ====== ITERATIVE SEARCH ======
            # Start with reasonable guesses based on direction
            if target_change < 0:
                test_interventions = [-5, -10, -15, -20, -25, -30]
            else:
                test_interventions = [5, 10, 15, 20, 25, 30]
            
            # Limit to max_intervention_pct
            test_interventions = [t for t in test_interventions if abs(t) <= max_intervention_pct]
            
            best_found = None
            best_error = float('inf')
            
            # Try each test intervention
            for test_pct in test_interventions:
                try:
                    result = self.simulate_intervention(
                        interventions={node: {'pct_change': test_pct}},
                        outcome_nodes=outcome_node
                    )
                    
                    actual_effect = result['outcome_effects'][outcome_node].get('pct_effect')
                    
                    if actual_effect is None:
                        continue
                    
                    # Calculate error from target
                    error = abs(actual_effect - target_change)
                    
                    # Track best so far
                    if error < best_error:
                        best_error = error
                        best_found = {
                            'intervention_pct': test_pct,
                            'actual_effect': actual_effect,
                            'error': error
                        }
                    
                    # If within tolerance, we found a good solution!
                    if error <= acceptable_error:
                        print(f"      ✓ Found: {test_pct:+.1f}% → {actual_effect:+.1f}% (error: {error:.1f}%)")
                        break
                        
                except Exception as e:
                    continue
            
            if best_found is None:
                print(f"      ✗ No valid interventions found")
                continue
            
            # ====== CHECK FOR ZERO-EFFECT ISSUE ======
            if abs(best_found['actual_effect']) < 0.01:
                print(f"      ⚠️  Effect ≈ 0 (model may be unreliable)")
                # Significantly lower confidence for near-zero effects
                zero_effect_penalty = 0.5
            else:
                zero_effect_penalty = 1.0
            
            # ====== ADAPTIVE REFINEMENT ======
            refinement_iterations = 0
            if best_found['error'] > acceptable_error:
                print(f"      ⚙️  Refining (current error: {best_found['error']:.1f}%)...")
                
                step_size = 1.0  # Start with full adjustment
                
                for iteration in range(max_search_iterations):
                    refinement_iterations = iteration + 1
                    current_pct = best_found['intervention_pct']
                    current_effect = best_found['actual_effect']
                    
                    # Check if we can refine (need non-zero effect)
                    if abs(current_effect) < 1e-9:
                        print(f"      ✗ Cannot refine from zero effect")
                        break
                    
                    # Calculate adjustment with adaptive step size
                    raw_adjustment_factor = target_change / current_effect
                    adjusted_pct = current_pct * (1 + (raw_adjustment_factor - 1) * step_size)
                    
                    # Clamp to max
                    adjusted_pct = max(-max_intervention_pct, min(max_intervention_pct, adjusted_pct))
                    
                    # Test refined intervention
                    try:
                        result = self.simulate_intervention(
                            interventions={node: {'pct_change': adjusted_pct}},
                            outcome_nodes=outcome_node
                        )
                        
                        actual_effect = result['outcome_effects'][outcome_node].get('pct_effect')
                        
                        if actual_effect is None:
                            step_size *= 0.5
                            if step_size < 0.01:
                                break
                            continue
                        
                        error = abs(actual_effect - target_change)
                        
                        if error < best_found['error']:
                            # Improvement! Increase confidence in step size
                            best_found = {
                                'intervention_pct': adjusted_pct,
                                'actual_effect': actual_effect,
                                'error': error
                            }
                            step_size = min(1.0, step_size * 1.1)
                            
                            if error <= acceptable_error:
                                print(f"      ✓ Converged: {adjusted_pct:+.1f}% → {actual_effect:+.1f}% (error: {error:.1f}%)")
                                break
                        else:
                            # No improvement, reduce step size
                            step_size *= 0.5
                            if step_size < 0.01:
                                break
                            
                    except Exception as e:
                        step_size *= 0.5
                        if step_size < 0.01:
                            break
            
            # ====== FINALIZE RESULT ======
            intervention_pct = best_found['intervention_pct']
            actual_effect = best_found['actual_effect']
            error = best_found['error']
            
            # Check if within tolerance
            within_tolerance = error <= acceptable_error
            
            # Calculate confidence based on multiple factors
            confidence_factors = []
            
            # Factor 1: Error magnitude
            if within_tolerance:
                confidence_factors.append(0.9 - (error / acceptable_error) * 0.2)
            else:
                confidence_factors.append(max(0.3, 0.7 - (error / abs(target_change))))
            
            # Factor 2: Model quality
            if hasattr(self.ht_model, 'model_metrics') and node in self.ht_model.model_metrics:
                metrics = self.ht_model.model_metrics[node]
                if metrics.get('model_type') == 'regression':
                    r2 = metrics.get('r2_score', 0.5)
                    confidence_factors.append(r2)
            
            # Factor 3: Zero-effect penalty
            confidence_factors.append(zero_effect_penalty)
            
            # Average confidence
            confidence = sum(confidence_factors) / len(confidence_factors)
            
            # Calculate leverage
            leverage = actual_effect / intervention_pct if abs(intervention_pct) > 1e-9 else 0
            
            result_entry = {
                'intervention_type': 'single',
                'nodes': [node],
                'required_pct_changes': {node: round(intervention_pct, 2)},
                'actual_effect': round(actual_effect, 2),
                'error_from_target': round(error, 2),
                'error_pct': round((error / abs(target_change)) * 100, 1) if abs(target_change) > 1e-9 else 0,
                'within_tolerance': within_tolerance,
                'leverage': round(leverage, 2),
                'confidence': round(confidence, 2),
                'feasible': abs(intervention_pct) <= max_intervention_pct,
                'search_iterations': refinement_iterations + 1
            }
            
            # ====== ADD UNCERTAINTY BOUNDS ======
            if enable_uncertainty:
                try:
                    margin_lower, margin_upper = self._estimate_prediction_interval(
                        {node: {'pct_change': intervention_pct}},
                        outcome_node,
                        confidence_level=0.90
                    )
                    
                    if margin_lower is not None:
                        result_entry['prediction_interval_lower'] = round(actual_effect - margin_lower, 2)
                        result_entry['prediction_interval_upper'] = round(actual_effect + margin_upper, 2)
                        print(f"      📊 90% CI: [{result_entry['prediction_interval_lower']:+.1f}%, {result_entry['prediction_interval_upper']:+.1f}%]")
                except Exception as e:
                    pass  # Uncertainty estimation failed, skip
            
            results.append(result_entry)
            
            if within_tolerance:
                print(f"      ✅ VALIDATED within tolerance")
            else:
                print(f"      ⚠️  Error {error:.1f}% exceeds tolerance ({acceptable_error:.1f}%)")
        
        # ============================================================
        # STEP 2: SMART COMBINATION SEARCH
        # ============================================================
        
        if allow_combinations and len(candidate_nodes) >= 2:
            print(f"\n🔗 Testing optimized two-node combinations...")
            
            # Only test combinations of nodes that worked reasonably well
            good_nodes = [r['nodes'][0] for r in results if r['error_from_target'] < abs(target_change) * 2]
            
            if len(good_nodes) < 2:
                good_nodes = candidate_nodes[:10]
            
            combo_count = 0
            for node1, node2 in list(combinations(good_nodes[:5], 2))[:10]:
                combo_count += 1
                
                # Grid search for optimal combination
                best_combo = None
                best_combo_error = float('inf')
                
                # Test 3x3 grid of intervention combinations
                test_range = [-20, -10, -5] if target_change < 0 else [5, 10, 20]
                
                for pct1 in test_range:
                    for pct2 in test_range:
                        if abs(pct1) > max_intervention_pct or abs(pct2) > max_intervention_pct:
                            continue
                        
                        try:
                            result = self.simulate_intervention(
                                interventions={
                                    node1: {'pct_change': pct1},
                                    node2: {'pct_change': pct2}
                                },
                                outcome_nodes=outcome_node
                            )
                            
                            actual_effect = result['outcome_effects'][outcome_node].get('pct_effect')
                            
                            if actual_effect is None:
                                continue
                            
                            error = abs(actual_effect - target_change)
                            
                            if error < best_combo_error:
                                best_combo_error = error
                                best_combo = {
                                    'pct1': pct1,
                                    'pct2': pct2,
                                    'effect': actual_effect
                                }
                        
                        except Exception as e:
                            continue
                
                # Add best combo if better than individual nodes
                if best_combo and best_combo_error <= acceptable_error * 1.5:
                    within_tolerance = best_combo_error <= acceptable_error
                    
                    # Calculate combined leverage
                    combined_pct = (abs(best_combo['pct1']) + abs(best_combo['pct2'])) / 2
                    leverage = best_combo['effect'] / combined_pct if combined_pct > 0 else 0
                    
                    # Confidence slightly lower for combinations
                    confidence = 0.75 if within_tolerance else 0.5
                    
                    result_entry = {
                        'intervention_type': 'combination',
                        'nodes': [node1, node2],
                        'required_pct_changes': {
                            node1: round(best_combo['pct1'], 2),
                            node2: round(best_combo['pct2'], 2)
                        },
                        'actual_effect': round(best_combo['effect'], 2),
                        'error_from_target': round(best_combo_error, 2),
                        'error_pct': round((best_combo_error / abs(target_change)) * 100, 1) if abs(target_change) > 1e-9 else 0,
                        'within_tolerance': within_tolerance,
                        'leverage': round(leverage, 2),
                        'confidence': round(confidence, 2),
                        'feasible': abs(best_combo['pct1']) <= max_intervention_pct and abs(best_combo['pct2']) <= max_intervention_pct,
                        'search_iterations': 9  # Grid search iterations
                    }
                    
                    results.append(result_entry)
                    
                    print(f"      ✓ Combo {combo_count}: {node1} ({best_combo['pct1']:+.1f}%) + {node2} ({best_combo['pct2']:+.1f}%) → {best_combo['effect']:+.1f}%")
        
        # ============================================================
        # STEP 3: RANK AND SELECT BEST
        # ============================================================
        
        if not results:
            return {
                'error': 'No feasible interventions found',
                'reason': 'Could not find any interventions that achieve the target',
                'recommendation': [
                    'Consider: 1) Relaxing tolerance constraints',
                    '2) Improving model quality (collect more training data)',
                    '3) Checking if target is realistic given the system dynamics'
                ]
            }
        
        # Sort by: within_tolerance > confidence > error
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(
            ['within_tolerance', 'confidence', 'error_from_target'],
            ascending=[False, False, True]
        )
        
        best = results_df.iloc[0].to_dict()
        
        # ============================================================
        # STEP 4: GENERATE BUSINESS-FRIENDLY REPORT
        # ============================================================
        
        print(f"\n{'='*70}")
        print(f"✅ SEARCH COMPLETE")
        print(f"{'='*70}")
        
        validated_count = sum(1 for r in results if r['within_tolerance'])
        high_conf_count = sum(1 for r in results if r['confidence'] >= 0.7)
        
        print(f"\nResults:")
        print(f"   Total options tested: {len(results)}")
        print(f"   Within tolerance: {validated_count}")
        print(f"   High confidence (≥70%): {high_conf_count}")
        
        print(f"\nBest Option:")
        print(f"   Type: {best['intervention_type']}")
        print(f"   Nodes: {', '.join(best['nodes'])}")
        for node, pct in best['required_pct_changes'].items():
            print(f"   └─ {node}: {pct:+.2f}%")
        print(f"   Predicted outcome: {best['actual_effect']:+.1f}% (target: {target_change:+.1f}%)")
        print(f"   Error: {best['error_from_target']:.1f}% ({best['error_pct']:.0f}% of target)")
        
        if 'prediction_interval_lower' in best:
            print(f"   90% Confidence Interval: [{best['prediction_interval_lower']:+.1f}%, {best['prediction_interval_upper']:+.1f}%]")
        
        print(f"   Leverage: {best['leverage']:.2f}x")
        print(f"   Confidence: {best['confidence']:.0%}")
        print(f"   Status: {'✅ VALIDATED' if best['within_tolerance'] else '⚠️  OUTSIDE TOLERANCE'}")
        
        # Generate clear recommendations
        recommendation_text = []
        
        if best['within_tolerance']:
            recommendation_text.append("✅ RECOMMENDATION: APPROVED FOR IMPLEMENTATION")
            recommendation_text.append(f"Expected outcome: {best['actual_effect']:+.1f}% ± {acceptable_error:.1f}%")
            recommendation_text.append(f"Confidence level: {best['confidence']:.0%}")
            
            if 'prediction_interval_lower' in best:
                recommendation_text.append(f"90% prediction range: [{best['prediction_interval_lower']:+.1f}%, {best['prediction_interval_upper']:+.1f}%]")
        else:
            recommendation_text.append("⚠️  RECOMMENDATION: CAUTION REQUIRED")
            recommendation_text.append(f"Prediction error ({best['error_pct']:.0f}%) exceeds tolerance")
            
            if best['confidence'] >= 0.6:
                recommendation_text.append("Consider: pilot test with monitoring before full deployment")
            else:
                recommendation_text.append("Consider: relaxed constraints, improved models, or alternative approach")
        
        print(f"\n{chr(10).join(recommendation_text)}")
        print(f"{'='*70}\n")
        
        return {
            'target_outcome': outcome_node,
            'target_change': target_change,
            'tolerance': acceptable_error,
            'best_intervention': best,
            'all_options': results_df.to_dict('records'),
            'summary': {
                'total_tested': len(results),
                'within_tolerance': validated_count,
                'high_confidence': high_conf_count,
                'best_error_pct': best['error_pct'],
                'approved_for_implementation': best['within_tolerance']
            },
            'recommendation': recommendation_text
        }


    def generate_recommendations(
        self,
        intervention_result: Dict,
        rca_results: 'RCAResults' = None,
        business_context: str = "",
        variable_definitions: Dict[str, str] = None,
        domain_context: str = "",
        api_key: str = None,
        max_recommendations: int = 3
    ) -> List[Dict]:
        """
        Generate executive-ready, actionable recommendations from intervention analysis.
        
        GENERIC and DOMAIN-AGNOSTIC: Works across manufacturing, healthcare, logistics, 
        finance, and any other domain.
        
        Args:
            intervention_result: Output from find_best_intervention()
            rca_results: RCAResults object from find_root_causes() (optional but recommended)
            business_context: Description of the business problem and goals
            variable_definitions: Dictionary mapping variable names to business-friendly descriptions
                Example: {
                    'moisture_2_sigma_cd': 'Cross-directional moisture variation',
                    'cpu_usage': 'Server CPU utilization percentage',
                    'patient_wait_time': 'Average time patients wait for service'
                }
            domain_context: Additional domain-specific context (industry terms, constraints, priorities)
            api_key: Gemini API key (uses GOOGLE_API_KEY env var if not provided)
            max_recommendations: Maximum number of recommendations (default: 3)
            
        Returns:
            List of executive-ready recommendation dictionaries
            
        Example Usage:
            # Manufacturing domain
            recommendations = simulator.generate_recommendations(
                intervention_result=best,
                rca_results=rca_results,
                business_context="Reduce delamination waste by 10% in corrugated packaging production",
                variable_definitions={
                    'moisture_2_sigma_cd': 'Moisture consistency across paper width',
                    'formation': 'Paper fiber uniformity and quality',
                    'corrugator_run_speed': 'Production line speed'
                },
                domain_context="Manufacturing facility produces 24/7. Downtime is costly. Focus on interventions implementable during production."
            )
            
            # Healthcare domain
            recommendations = simulator.generate_recommendations(
                intervention_result=best,
                business_context="Reduce patient wait times by 15% in emergency department",
                variable_definitions={
                    'triage_time': 'Time to complete initial patient assessment',
                    'nurse_staffing': 'Number of nurses on duty per shift',
                    'bed_availability': 'Percentage of beds available'
                },
                domain_context="Urban hospital, 250 beds, serving diverse population. Patient satisfaction is key metric."
            )
        """
        from google import genai
        import os
        import json
        
        # Validate input
        if 'error' in intervention_result:
            raise ValueError(f"Intervention result contains error: {intervention_result['error']}")
        
        # === STEP 1: Extract Key Information ===
        print("📊 Extracting intervention results...")
        
        target_outcome = intervention_result['target_outcome']
        target_change = intervention_result['target_change']
        all_options = intervention_result['all_options']
        best_option = intervention_result['best_intervention']
        
        # Determine direction
        direction = "reduce" if target_change < 0 else "increase"
        
        # === STEP 2: Select Top Options ===
        # Prioritize: validated + feasible > feasible > high confidence
        feasible_options = [opt for opt in all_options if opt.get('feasible', False) and opt.get('within_tolerance', False)]
        
        if len(feasible_options) >= max_recommendations:
            selected_options = feasible_options[:max_recommendations]
        elif len(feasible_options) > 0:
            remaining_feasible = [opt for opt in all_options if opt.get('feasible', False) and not opt.get('within_tolerance', False)]
            selected_options = feasible_options + remaining_feasible[:max_recommendations - len(feasible_options)]
        else:
            # Take top options by confidence and leverage
            selected_options = sorted(all_options, key=lambda x: (x.get('confidence', 0), abs(x.get('leverage', 0))), reverse=True)[:max_recommendations]
        
        # === STEP 3: Build Variable Context for LLM ===
        print("🔗 Building variable context...")
        
        # Collect all variables mentioned in selected options
        all_variables = set()
        for option in selected_options:
            all_variables.update(option['nodes'])
        all_variables.add(target_outcome)
        
        # Build variable context string for LLM
        variable_context = ""
        if variable_definitions:
            variable_context = "\n\nVARIABLE DEFINITIONS:\n"
            for var in all_variables:
                if var in variable_definitions:
                    variable_context += f"- {var}: {variable_definitions[var]}\n"
                else:
                    # Variable not defined, let LLM infer from name
                    variable_context += f"- {var}: [Infer meaning from variable name]\n"
        else:
            variable_context = "\n\nNOTE: No variable definitions provided. Translate technical variable names to clear business language based on context and common sense.\n"
        
        # === STEP 4: Build Structured Intervention Context ===
        print("🔗 Building intervention context for LLM...")
        
        intervention_summaries = []
        
        for idx, option in enumerate(selected_options, 1):
            nodes = option['nodes']
            required_changes = option['required_pct_changes']
            
            # Format changes
            intervention_details = []
            for node in nodes:
                change_pct = required_changes[node]
                action = "Reduce" if change_pct < 0 else "Increase"
                intervention_details.append(f"{action} {node} by {abs(change_pct):.1f}%")
            
            # Get causal path if available
            causal_mechanism = "Direct effect"
            if rca_results and hasattr(rca_results, 'root_cause_paths'):
                for node in nodes:
                    if node in rca_results.root_cause_paths:
                        paths = rca_results.root_cause_paths[node]
                        if paths:
                            raw_path = paths[0]['path']
                            causal_mechanism = ' → '.join(raw_path)
                            break
            
            summary = {
                'option_number': idx,
                'intervention_required': ' AND '.join(intervention_details),
                'expected_outcome': f"{abs(option['actual_effect']):.1f}% {direction} in {target_outcome}",
                'validated': option.get('within_tolerance', False),
                'feasible': option.get('feasible', False),
                'causal_mechanism': causal_mechanism,
                'priority': option.get('priority', 'Medium')
            }
            
            intervention_summaries.append(summary)
        
        # === STEP 5: Generate Executive-Ready LLM Prompt ===
        prompt = f"""You are a senior consultant preparing recommendations for C-level executives.

    BUSINESS OBJECTIVE:
    {direction.capitalize()} {target_outcome} by {abs(target_change)}%

    BUSINESS CONTEXT:
    {business_context}

    {f"DOMAIN CONTEXT: {domain_context}" if domain_context else ""}

    {variable_context}

    VALIDATED INTERVENTIONS FROM ANALYSIS:
    Our causal analysis identified {len(selected_options)} actionable interventions:

    """
        
        for intervention in intervention_summaries:
            status = "✓ VALIDATED" if intervention['validated'] else "⚠ REQUIRES MONITORING"
            prompt += f"""
    Option {intervention['option_number']}: {status}
    • Required Action: {intervention['intervention_required']}
    • Expected Outcome: {intervention['expected_outcome']}
    • Causal Mechanism: {intervention['causal_mechanism']}
    • Implementation Priority: {intervention['priority']}
    • Feasible: {'Yes - can be implemented immediately' if intervention['feasible'] else 'Requires further assessment'}

    """
        
        prompt += f"""
    YOUR TASK:
    Transform these analytical findings into {max_recommendations} executive-ready recommendations suitable for C-suite decision makers.

    CRITICAL REQUIREMENTS:

    1. TRANSLATE TECHNICAL TO BUSINESS LANGUAGE:
    - Convert ALL technical variable names to clear business terminology
    - Use the variable definitions provided when available
    - If no definition provided, infer the business meaning from context
    - Avoid acronyms unless they're standard in the domain
    - Think: "What would a CEO or CFO call this variable?"

    2. CLARITY & CONCISENESS:
    - Summary: ONE sentence, max 12 words, pure action statement
    - Detail: 2-3 sentences explaining WHAT to do and WHY it matters to the business
    - NO statistical jargon, NO model terminology, NO technical metrics
    - Write for someone with domain knowledge but no analytics background

    3. ACTION-ORIENTED & SPECIFIC:
    - Start with strong action verbs: Implement, Optimize, Standardize, Monitor, Adjust, Establish, Reduce, Increase
    - Include the SPECIFIC percentage changes from the analysis
    - Focus on WHAT to do operationally, not the analytical process
    - Make it clear WHO should implement (operations, procurement, engineering, etc.)

    4. BUSINESS IMPACT:
    - Clearly connect action to business outcome
    - Explain WHY this matters: cost savings, quality, customer satisfaction, efficiency, risk reduction
    - Include the expected quantitative impact from the analysis
    - Reference the causal mechanism in simple terms

    5. APPROPRIATE CONFIDENCE FRAMING:
    - If VALIDATED and FEASIBLE: Frame with confidence ("This intervention has been validated and can be implemented")
    - If FEASIBLE but not validated: Frame with caution ("Implement with monitoring to track actual impact")
    - If neither: Frame as requiring pilot testing ("Recommend pilot test before full deployment")

    6. DISTINCTIVENESS:
    - Each recommendation must address a DIFFERENT operational lever
    - No overlap between recommendations
    - Cover diverse approaches (don't just recommend 3 variations of the same thing)

    CATEGORY SELECTION (choose most appropriate):
    - Quality Control: Product quality, defect reduction, consistency, inspection
    - Process Optimization: Efficiency, workflow, cycle time, throughput
    - Material Management: Raw materials, inventory, specifications, sourcing
    - Equipment Utilization: Machine settings, maintenance, capacity, calibration
    - Logistics Flow: Material handling, scheduling, routing, sequencing
    - Resource Allocation: Staffing, capacity planning, workload distribution
    - Customer Service: Wait times, service quality, satisfaction, response time
    - Cost Management: Waste reduction, resource efficiency, cost per unit
    - Risk Management: Safety, compliance, reliability, error prevention

    RESPONSE FORMAT (JSON array):
    [
    {{
        "summary": "Action verb + specific business-friendly intervention + expected outcome",
        "recommendationDetail": "Explain what to do operationally, how it works causally (in simple terms), and the business impact. Use specific percentages from analysis. Frame confidence appropriately based on validation status. Make it actionable for operations teams.",
        "category": "Most relevant category from list above",
        "subcategory": "Specific focus area within category"
    }}
    ]

    QUALITY CHECKLIST:
    ✓ Every variable name translated to business language
    ✓ Specific percentages from analysis included
    ✓ Expected outcome quantified
    ✓ Causal mechanism explained simply
    ✓ Validation status reflected in tone
    ✓ Actionable by operations (not just aspirational)
    ✓ Each recommendation is distinct
    ✓ No technical jargon or statistical terms

    EXAMPLES OF EXCELLENT EXECUTIVE RECOMMENDATIONS:

    Good Summary (Generic):
    ✓ "Optimize [BUSINESS TERM] to reduce [OUTCOME] by X%"
    ✓ "Standardize [BUSINESS PROCESS] to improve [METRIC]"
    ✓ "Adjust [OPERATIONAL LEVER] to achieve [TARGET]"

    Good Detail (Generic):
    ✓ "Implement tighter controls on [BUSINESS TERM] to maintain consistency. Analysis shows [VARIABLE A] influences [OUTCOME] through [SIMPLE CAUSAL MECHANISM]. This intervention will reduce [OUTCOME] by X%, directly improving [BUSINESS METRIC]. Validated and ready for immediate implementation."

    ✓ "Establish new standards for [PROCESS/MATERIAL] to ensure [QUALITY ATTRIBUTE]. This addresses a root cause of [PROBLEM], with projected X% improvement in [METRIC]. Requires coordination with [DEPARTMENT/SUPPLIERS] but offers high confidence results with minimal risk."

    Bad Examples (DO NOT DO):
    ✗ "Reduce moisture_2_sigma_cd by 25%" (technical variable name not translated)
    ✗ "The regression analysis shows high leverage" (analytical jargon)
    ✗ "Implement statistical process control" (too vague, not specific)
    ✗ "This option has 94% confidence and 0.43 leverage" (technical metrics)

    Generate {max_recommendations} recommendations in JSON format. Return ONLY the JSON array, no other text.
    """
        
        # === STEP 6: Call Gemini LLM ===
        print("🤖 Generating executive recommendations with Gemini...")
        
        try:
            # Initialize Gemini client
            if api_key:
                os.environ['GOOGLE_API_KEY'] = api_key
            
            client = genai.Client()
            
            # Call LLM with increased temperature for better language quality
            response = client.models.generate_content(
                model="gemini-flash-latest",
                contents=prompt
            )
            
            # Parse response
            llm_text = response.text.strip()
            
            # Extract JSON from response (handle markdown code blocks)
            if '```json' in llm_text:
                llm_text = llm_text.split('```json')[1].split('```')[0].strip()
            elif '```' in llm_text:
                llm_text = llm_text.split('```')[1].split('```')[0].strip()
            
            llm_recommendations = json.loads(llm_text)
            
            print(f"✅ Generated {len(llm_recommendations)} executive recommendations")
            
        except Exception as e:
            print(f"⚠️  LLM generation failed: {e}")
            print("⚠️  Using fallback template generation...")
            
            # Fallback: Generate simple recommendations without LLM
            llm_recommendations = []
            for intervention in intervention_summaries[:max_recommendations]:
                summary = f"{intervention['intervention_required']} to achieve {direction} in {target_outcome}"
                detail = f"Implement {intervention['intervention_required'].lower()}. Causal mechanism: {intervention['causal_mechanism']}. Expected outcome: {intervention['expected_outcome']}. {'Validated and ready for implementation.' if intervention['validated'] else 'Requires monitoring during implementation.'}"
                
                llm_recommendations.append({
                    'summary': summary,
                    'recommendationDetail': detail,
                    'category': 'Process Optimization',
                    'subcategory': 'Operational Improvement'
                })
        
        # === STEP 7: Combine with Quantitative Metrics ===
        print("🔗 Finalizing recommendations with validation data...")
        
        final_recommendations = []
        
        for idx, (llm_rec, option) in enumerate(zip(llm_recommendations, selected_options[:max_recommendations]), 1):
            
            # Build causal path details (keep raw technical names here for traceability)
            causal_details = []
            if rca_results and hasattr(rca_results, 'root_cause_paths'):
                for node in option['nodes']:
                    if node in rca_results.root_cause_paths:
                        paths = rca_results.root_cause_paths[node]
                        for path_info in paths[:2]:  # Max 2 paths
                            raw_path = path_info['path']
                            causal_details.append({'causal_path': ' → '.join(raw_path)})
            
            # Fallback if no RCA paths
            if not causal_details:
                simple_path = ' → '.join(option['nodes'] + [target_outcome])
                causal_details.append({'causal_path': simple_path})
            
            # Determine type: action (implementable) vs signal (informational)
            is_validated = option.get('within_tolerance', False)
            is_feasible = option.get('feasible', False)
            rec_type = 'action' if (is_validated and is_feasible) else 'signal'
            
            # Priority mapping
            priority = option.get('priority', 'Medium')
            
            # Validity period (days) based on validation status
            if is_validated and is_feasible:
                validity = 30  # High confidence
            elif is_feasible:
                validity = 21  # Medium confidence
            else:
                validity = 14  # Lower confidence
            
            # Confidence level (simplified for executives)
            if is_validated and is_feasible:
                confidence = 'High'
            elif is_feasible or is_validated:
                confidence = 'Medium'
            else:
                confidence = 'Low'
            
            recommendation = {
                'id': idx,
                'cat': llm_rec.get('category', 'Process Optimization'),
                'type': rec_type,
                'subcat': llm_rec.get('subcategory', 'Operational Improvement'),
                'details': causal_details,
                'summary': llm_rec['summary'],
                'priority': priority,
                'validity': validity,
                'confidence': confidence,
                'opportunityValue': None,  # Can be calculated if cost data available
                'recommendationDetail': llm_rec['recommendationDetail']
            }
            
            final_recommendations.append(recommendation)
        
        # === STEP 8: Final Validation ===
        print("✅ Validating recommendations...")
        
        # Check for distinctiveness
        summaries = [r['summary'] for r in final_recommendations]
        if len(summaries) != len(set(summaries)):
            print("⚠️  Warning: Some recommendations may be too similar")
        
        # Check summary length
        for rec in final_recommendations:
            word_count = len(rec['summary'].split())
            if word_count > 15:
                print(f"⚠️  Warning: Recommendation {rec['id']} summary is long ({word_count} words)")
        
        print(f"✅ Generated {len(final_recommendations)} executive-ready recommendations\n")
        
        return final_recommendations