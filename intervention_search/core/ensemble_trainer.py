"""
Ensemble Model Trainer for HT

Trains multiple models per node with varying complexity and selects the best one.
This addresses the need for more robust model selection and better uncertainty estimation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, f1_score
import warnings

warnings.filterwarnings('ignore')


class ModelCandidate:
    """Represents a single model candidate with its configuration"""

    def __init__(self, name: str, model_type: str, complexity: str):
        """
        Args:
            name: Display name
            model_type: Type identifier (for creating model)
            complexity: 'low', 'medium', 'high'
        """
        self.name = name
        self.model_type = model_type
        self.complexity = complexity

    def create_regressor(self):
        """Create regression model instance"""
        if self.model_type == 'linear':
            from sklearn.linear_model import LinearRegression
            return LinearRegression()

        elif self.model_type == 'ridge':
            from sklearn.linear_model import Ridge
            return Ridge(alpha=1.0, random_state=42)

        elif self.model_type == 'lasso':
            from sklearn.linear_model import Lasso
            return Lasso(alpha=0.1, random_state=42, max_iter=2000)

        elif self.model_type == 'elasticnet':
            from sklearn.linear_model import ElasticNet
            return ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=2000)

        elif self.model_type == 'randomforest':
            from sklearn.ensemble import RandomForestRegressor
            n_estimators = 50 if self.complexity == 'low' else 100 if self.complexity == 'medium' else 200
            max_depth = 5 if self.complexity == 'low' else 10 if self.complexity == 'medium' else None
            return RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1
            )

        elif self.model_type == 'xgboost':
            try:
                import xgboost as xgb
                n_estimators = 50 if self.complexity == 'low' else 100 if self.complexity == 'medium' else 200
                max_depth = 3 if self.complexity == 'low' else 5 if self.complexity == 'medium' else 7
                return xgb.XGBRegressor(
                    objective='reg:squarederror',
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0
                )
            except ImportError:
                return None

        elif self.model_type == 'lightgbm':
            try:
                import lightgbm as lgb
                n_estimators = 50 if self.complexity == 'low' else 100 if self.complexity == 'medium' else 200
                max_depth = 3 if self.complexity == 'low' else 5 if self.complexity == 'medium' else 7
                return lgb.LGBMRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                )
            except ImportError:
                return None

        elif self.model_type == 'catboost':
            try:
                from catboost import CatBoostRegressor
                iterations = 50 if self.complexity == 'low' else 100 if self.complexity == 'medium' else 200
                depth = 4 if self.complexity == 'low' else 6 if self.complexity == 'medium' else 8
                return CatBoostRegressor(
                    iterations=iterations,
                    depth=depth,
                    learning_rate=0.1,
                    random_state=42,
                    verbose=0
                )
            except ImportError:
                return None

        return None

    def create_classifier(self):
        """Create classification model instance"""
        if self.model_type == 'linear':
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(max_iter=1000, random_state=42)

        elif self.model_type in ['ridge', 'lasso', 'elasticnet']:
            # For classification, use LogisticRegression with appropriate penalty
            from sklearn.linear_model import LogisticRegression
            penalty = 'l2' if self.model_type == 'ridge' else 'l1' if self.model_type == 'lasso' else 'elasticnet'
            return LogisticRegression(
                penalty=penalty,
                max_iter=1000,
                random_state=42,
                solver='saga' if penalty == 'elasticnet' else 'liblinear'
            )

        elif self.model_type == 'randomforest':
            from sklearn.ensemble import RandomForestClassifier
            n_estimators = 50 if self.complexity == 'low' else 100 if self.complexity == 'medium' else 200
            max_depth = 5 if self.complexity == 'low' else 10 if self.complexity == 'medium' else None
            return RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1
            )

        elif self.model_type == 'xgboost':
            try:
                import xgboost as xgb
                n_estimators = 50 if self.complexity == 'low' else 100 if self.complexity == 'medium' else 200
                max_depth = 3 if self.complexity == 'low' else 5 if self.complexity == 'medium' else 7
                return xgb.XGBClassifier(
                    objective='multi:softmax',
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0,
                    eval_metric='mlogloss'
                )
            except ImportError:
                return None

        elif self.model_type == 'lightgbm':
            try:
                import lightgbm as lgb
                n_estimators = 50 if self.complexity == 'low' else 100 if self.complexity == 'medium' else 200
                max_depth = 3 if self.complexity == 'low' else 5 if self.complexity == 'medium' else 7
                return lgb.LGBMClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                )
            except ImportError:
                return None

        elif self.model_type == 'catboost':
            try:
                from catboost import CatBoostClassifier
                iterations = 50 if self.complexity == 'low' else 100 if self.complexity == 'medium' else 200
                depth = 4 if self.complexity == 'low' else 6 if self.complexity == 'medium' else 8
                return CatBoostClassifier(
                    iterations=iterations,
                    depth=depth,
                    learning_rate=0.1,
                    random_state=42,
                    verbose=0
                )
            except ImportError:
                return None

        return None


# Define model candidates with varying complexity
REGRESSION_CANDIDATES = [
    # Linear models (low complexity)
    ModelCandidate("Linear Regression", "linear", "low"),
    ModelCandidate("Ridge Regression", "ridge", "low"),
    ModelCandidate("Lasso Regression", "lasso", "low"),

    # Medium complexity
    ModelCandidate("Random Forest (Simple)", "randomforest", "low"),
    ModelCandidate("Random Forest (Medium)", "randomforest", "medium"),

    # High complexity (ensemble methods)
    ModelCandidate("XGBoost (Simple)", "xgboost", "low"),
    ModelCandidate("XGBoost (Medium)", "xgboost", "medium"),
    ModelCandidate("LightGBM (Simple)", "lightgbm", "low"),
    ModelCandidate("LightGBM (Medium)", "lightgbm", "medium"),
    ModelCandidate("CatBoost (Simple)", "catboost", "low"),
]

CLASSIFICATION_CANDIDATES = [
    # Linear models
    ModelCandidate("Logistic Regression", "linear", "low"),
    ModelCandidate("Ridge Classifier", "ridge", "low"),

    # Medium complexity
    ModelCandidate("Random Forest (Simple)", "randomforest", "low"),
    ModelCandidate("Random Forest (Medium)", "randomforest", "medium"),

    # High complexity
    ModelCandidate("XGBoost (Simple)", "xgboost", "low"),
    ModelCandidate("XGBoost (Medium)", "xgboost", "medium"),
    ModelCandidate("LightGBM (Simple)", "lightgbm", "low"),
    ModelCandidate("CatBoost (Simple)", "catboost", "low"),
]


class EnsembleTrainer:
    """
    Trains multiple models per node and selects the best one.

    This provides:
    1. Better model selection (not limited to single model type)
    2. More robust uncertainty estimates
    3. Automatic complexity tuning
    """

    def __init__(
        self,
        cv_folds: int = 5,
        scoring_metric_regression: str = 'r2',
        scoring_metric_classification: str = 'accuracy',
        prefer_simpler_models: bool = True,
        complexity_penalty: float = 0.02
    ):
        """
        Initialize ensemble trainer.

        Args:
            cv_folds: Number of cross-validation folds
            scoring_metric_regression: Metric for regression ('r2', 'neg_mean_squared_error')
            scoring_metric_classification: Metric for classification ('accuracy', 'f1_weighted')
            prefer_simpler_models: If True, prefer simpler models when performance is similar
            complexity_penalty: Penalty for each complexity level increase (0-1)
        """
        self.cv_folds = cv_folds
        self.scoring_metric_regression = scoring_metric_regression
        self.scoring_metric_classification = scoring_metric_classification
        self.prefer_simpler_models = prefer_simpler_models
        self.complexity_penalty = complexity_penalty

    def train_best_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        is_categorical: bool,
        node_name: str = "unknown"
    ) -> Tuple[Any, Dict]:
        """
        Train multiple models and select the best one.

        Args:
            X: Feature matrix
            y: Target variable
            is_categorical: Whether target is categorical
            node_name: Name of node (for logging)

        Returns:
            Tuple of (best_model, metrics_dict)
        """
        candidates = CLASSIFICATION_CANDIDATES if is_categorical else REGRESSION_CANDIDATES

        results = []

        for candidate in candidates:
            try:
                # Create model
                if is_categorical:
                    model = candidate.create_classifier()
                else:
                    model = candidate.create_regressor()

                if model is None:
                    continue  # Skip if dependencies not available

                # Train model
                model.fit(X, y)
                y_pred = model.predict(X)

                # Evaluate with cross-validation
                if X.shape[0] >= self.cv_folds * 2:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            if is_categorical:
                                cv_scores = cross_val_score(
                                    model, X, y,
                                    cv=self.cv_folds,
                                    scoring=self.scoring_metric_classification
                                )
                            else:
                                cv_scores = cross_val_score(
                                    model, X, y,
                                    cv=self.cv_folds,
                                    scoring=self.scoring_metric_regression
                                )
                            cv_mean = np.mean(cv_scores)
                            cv_std = np.std(cv_scores)
                    except Exception:
                        cv_mean = None
                        cv_std = None
                else:
                    cv_mean = None
                    cv_std = None

                # Compute metrics
                if is_categorical:
                    train_score = accuracy_score(y, y_pred)
                    f1 = f1_score(y, y_pred, average='weighted', zero_division=0)

                    metrics = {
                        'model_name': candidate.name,
                        'model_type': 'classification',
                        'complexity': candidate.complexity,
                        'train_accuracy': round(train_score, 4),
                        'train_f1': round(f1, 4),
                        'cv_mean': round(cv_mean, 4) if cv_mean is not None else None,
                        'cv_std': round(cv_std, 4) if cv_std is not None else None,
                        'score': cv_mean if cv_mean is not None else train_score
                    }
                else:
                    train_score = r2_score(y, y_pred)
                    rmse = np.sqrt(mean_squared_error(y, y_pred))
                    mae = mean_absolute_error(y, y_pred)

                    metrics = {
                        'model_name': candidate.name,
                        'model_type': 'regression',
                        'complexity': candidate.complexity,
                        'train_r2': round(train_score, 4),
                        'train_rmse': round(rmse, 4),
                        'train_mae': round(mae, 4),
                        'cv_mean': round(cv_mean, 4) if cv_mean is not None else None,
                        'cv_std': round(cv_std, 4) if cv_std is not None else None,
                        'score': cv_mean if cv_mean is not None else train_score
                    }

                results.append({
                    'model': model,
                    'candidate': candidate,
                    'metrics': metrics
                })

            except Exception as e:
                # Skip models that fail
                continue

        if not results:
            raise ValueError(f"No models could be trained for {node_name}")

        # Select best model
        best_result = self._select_best_model(results)

        # Add comparison info
        best_result['metrics']['n_models_tested'] = len(results)
        best_result['metrics']['all_models'] = [r['metrics'] for r in results]

        return best_result['model'], best_result['metrics']

    def _select_best_model(self, results: List[Dict]) -> Dict:
        """
        Select best model from candidates.

        If prefer_simpler_models is True, applies complexity penalty.
        """
        if not self.prefer_simpler_models:
            # Simply select model with highest score
            return max(results, key=lambda r: r['metrics']['score'] or -1e9)

        # Apply complexity penalty
        complexity_map = {'low': 0, 'medium': 1, 'high': 2}

        best_result = None
        best_adjusted_score = -1e9

        for result in results:
            score = result['metrics']['score']
            if score is None:
                continue

            complexity_level = complexity_map.get(result['candidate'].complexity, 1)
            penalty = complexity_level * self.complexity_penalty
            adjusted_score = score - penalty

            if adjusted_score > best_adjusted_score:
                best_adjusted_score = adjusted_score
                best_result = result

        if best_result is None:
            # Fallback to first result
            best_result = results[0]

        return best_result


def integrate_ensemble_training_into_ht(ht_instance, normal_df: pd.DataFrame, **kwargs):
    """
    Replace the standard training in HT with ensemble training.

    This is a helper function to make it easy to use ensemble training
    with an existing HT instance.

    Args:
        ht_instance: HT instance
        normal_df: Training data
        **kwargs: Additional arguments for ensemble trainer

    Returns:
        Modified HT instance
    """
    ensemble_trainer = EnsembleTrainer(**kwargs)

    # Store original train method
    original_train = ht_instance.train

    # Wrapper that uses ensemble training
    def train_with_ensemble(normal_df: pd.DataFrame, use_ensemble: bool = True, **train_kwargs):
        if not use_ensemble:
            # Use original training
            return original_train(normal_df, **train_kwargs)

        # Initialize storage
        ht_instance.node_types = {}
        ht_instance.label_encoders = {}
        ht_instance.model_metrics = {}
        ht_instance.ensemble_selection_details = {}  # Store which models were selected

        print("=" * 70)
        print("🎓 ENSEMBLE TRAINING - TESTING MULTIPLE MODELS PER NODE")
        print("=" * 70)

        # Step 1: Detect variable types
        print("\n📊 Detecting variable types...")
        for node in list(ht_instance.graph):
            if node not in normal_df.columns:
                continue

            is_categorical = ht_instance._detect_categorical(normal_df[node])
            ht_instance.node_types[node] = 'categorical' if is_categorical else 'continuous'

            if is_categorical:
                le = LabelEncoder()
                le.fit(normal_df[node].dropna())
                ht_instance.label_encoders[node] = le
                print(f"   ✓ {node}: CATEGORICAL ({len(le.classes_)} classes)")
            else:
                print(f"   ✓ {node}: CONTINUOUS")

        # Step 2: Train ensemble models
        print(f"\n🔧 Training ensemble models...")

        for node in list(ht_instance.graph):
            if node not in normal_df.columns:
                continue

            parents = list(ht_instance.graph.predecessors(node))
            is_node_categorical = ht_instance.node_types.get(node) == 'categorical'

            # Prepare target
            if is_node_categorical:
                y = ht_instance.label_encoders[node].transform(normal_df[node].dropna())
                valid_idx = normal_df[node].notna()
            else:
                y = normal_df[node].values
                valid_idx = np.ones(len(y), dtype=bool)

            if parents and len(parents) > 0:
                # Prepare features
                X_list = []
                for parent in parents:
                    if ht_instance.node_types.get(parent) == 'categorical':
                        encoded = ht_instance.label_encoders[parent].transform(normal_df[parent].dropna())
                        if len(encoded) != len(normal_df):
                            parent_series_encoded = np.zeros(len(normal_df))
                            parent_series_encoded[normal_df[parent].notna()] = encoded
                            X_list.append(parent_series_encoded)
                        else:
                            X_list.append(encoded)
                    else:
                        X_list.append(normal_df[parent].values)

                X = np.column_stack(X_list)
                X = X[valid_idx]

                if X.shape[0] < 10:
                    print(f"   ⚠️  {node}: Insufficient samples, skipping")
                    scaler = StandardScaler().fit(y.reshape(-1, 1))
                    ht_instance.regressors_dict[node] = [None, scaler]
                    continue

                # Train ensemble
                try:
                    print(f"   🔍 {node}: Testing multiple models...")
                    best_model, metrics = ensemble_trainer.train_best_model(
                        X, y, is_node_categorical, node
                    )

                    # Store best model
                    if is_node_categorical:
                        ht_instance.regressors_dict[node] = [best_model, None]
                    else:
                        y_pred = best_model.predict(X)
                        residuals = y - y_pred
                        scaler = StandardScaler().fit(residuals.reshape(-1, 1))
                        ht_instance.regressors_dict[node] = [best_model, scaler]

                    # Store metrics
                    metrics['node'] = node
                    metrics['parents'] = parents
                    metrics['n_parents'] = len(parents)
                    ht_instance.model_metrics[node] = metrics

                    # Store selection details
                    ht_instance.ensemble_selection_details[node] = metrics

                    # Print result
                    if is_node_categorical:
                        print(f"      ✓ BEST: {metrics['model_name']} | Accuracy: {metrics.get('train_accuracy', 0):.3f} | CV: {metrics.get('cv_mean', 0):.3f}")
                    else:
                        print(f"      ✓ BEST: {metrics['model_name']} | R²: {metrics.get('train_r2', 0):.3f} | CV: {metrics.get('cv_mean', 0):.3f}")

                except Exception as e:
                    print(f"   ❌ {node}: Ensemble training failed - {str(e)[:100]}")
                    if is_node_categorical:
                        ht_instance.regressors_dict[node] = [None, None]
                    else:
                        scaler = StandardScaler().fit(y.reshape(-1, 1))
                        ht_instance.regressors_dict[node] = [None, scaler]
                    continue

            else:
                # Root node
                print(f"   ✓ {node}: Root node (no parents)")
                if not is_node_categorical:
                    scaler = StandardScaler().fit(y.reshape(-1, 1))
                    ht_instance.regressors_dict[node] = [None, scaler]
                else:
                    ht_instance.regressors_dict[node] = [None, None]

                ht_instance.model_metrics[node] = {
                    'node': node,
                    'model_type': 'root_node',
                    'parents': [],
                    'n_parents': 0,
                    'n_samples': len(y)
                }

        # Step 3: Compute baseline statistics (same as before)
        print("\n📈 Computing baseline statistics...")
        ht_instance.baseline_stats = {}
        for node in list(ht_instance.graph):
            if node not in normal_df.columns:
                continue

            series = normal_df[node]

            if ht_instance.node_types.get(node) == 'categorical':
                ht_instance.baseline_stats[node] = {
                    "type": "categorical",
                    "mode": series.mode()[0] if len(series.mode()) > 0 else None,
                    "distribution": series.value_counts(normalize=True).to_dict(),
                    "count": len(series),
                }
            else:
                ht_instance.baseline_stats[node] = {
                    "type": "continuous",
                    "mean": float(series.mean()),
                    "std": float(series.std()),
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "count": len(series),
                }

        # Step 4: Estimate edge elasticities (same as before but skip categorical)
        print("\n🔗 Estimating edge elasticities...")
        ht_instance.edge_elasticities = {}

        for node_v in list(ht_instance.graph):
            parents = list(ht_instance.graph.predecessors(node_v))
            if not parents or node_v not in ht_instance.regressors_dict:
                continue

            model, _ = ht_instance.regressors_dict[node_v]
            if model is None:
                continue

            if ht_instance.node_types.get(node_v) != 'continuous':
                continue

            # Extract coefficients/importances
            from sklearn.linear_model import LinearRegression

            if isinstance(model, LinearRegression) and hasattr(model, "coef_"):
                if node_v in ht_instance.baseline_stats and all(p in ht_instance.baseline_stats for p in parents):
                    mean_v = ht_instance.baseline_stats[node_v].get("mean")
                    if mean_v is not None and abs(mean_v) > 1e-9:
                        for i, node_u in enumerate(parents):
                            if i >= len(model.coef_):
                                continue
                            if ht_instance.node_types.get(node_u) == 'categorical':
                                continue
                            mean_u = ht_instance.baseline_stats[node_u].get("mean")
                            if mean_u is not None:
                                coef_u = model.coef_[i]
                                elasticity = coef_u * (mean_u / mean_v)
                                ht_instance.edge_elasticities[(node_u, node_v)] = float(elasticity)

            elif hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                total_importance = importances.sum()

                if total_importance > 0 and node_v in ht_instance.baseline_stats:
                    mean_v = ht_instance.baseline_stats[node_v].get("mean")
                    if mean_v is not None and abs(mean_v) > 1e-9:
                        for i, node_u in enumerate(parents):
                            if i >= len(importances):
                                continue
                            if ht_instance.node_types.get(node_u) == 'categorical':
                                continue
                            weight = importances[i] / total_importance
                            mean_u = ht_instance.baseline_stats[node_u].get("mean", 1.0)
                            elasticity = weight * (mean_u / mean_v)
                            ht_instance.edge_elasticities[(node_u, node_v)] = float(elasticity)

        print(f"   ✓ Found {len(ht_instance.edge_elasticities)} edge elasticities")

        print("\n" + "=" * 70)
        print("✅ ENSEMBLE TRAINING COMPLETE")
        print("=" * 70)
        print(f"Total nodes: {len(ht_instance.graph.nodes())}")
        print(f"Models trained: {len([m for m in ht_instance.model_metrics.values() if m.get('model_type') not in ['root_node']])}")
        print(f"\nModel Selection Summary:")

        # Summarize model selections
        model_counts = {}
        for node, details in ht_instance.ensemble_selection_details.items():
            model_name = details.get('model_name', 'Unknown')
            model_counts[model_name] = model_counts.get(model_name, 0) + 1

        for model_name, count in sorted(model_counts.items(), key=lambda x: -x[1]):
            print(f"   {model_name}: {count} nodes")

        print("=" * 70 + "\n")

    # Replace train method
    ht_instance.train = train_with_ensemble

    return ht_instance
