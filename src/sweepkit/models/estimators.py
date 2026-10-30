"""Supervised models from scikit-learn-style libraries, all through one adapter: each registered name builds the
library object for the task (classification, multi-label or regression) with every setting passed explicitly, plus
`extra` for any library parameter the settings class does not list. The runner (sweepkit.families.estimator) fits it
on the training rows and reads predict and predict_proba (or decision_function).

GPU paths (device cuda): cuML for logistic regression, SVMs, k-nearest neighbours, random forests and the linear
regressions; XGBoost's own CUDA path. Everything has a CPU path.

Adding a model: one settings class, one build function, one register_estimator line, an example sweep, a smoke entry
and a test (docs/EXTENDING.md; `scaffold --kind estimator` writes them).
"""
import dataclasses
from typing import Literal

from sweepkit.models.base import merged_kwargs
from sweepkit.registry import register_model
from sweepkit.settings import Scalar

CLASSIFY = ("classification", "multilabel")
REGRESS = ("regression",)
ALL = ("classification", "multilabel", "regression")


def register_estimator(name: str, settings: type, tasks: tuple, gpu: str, profile: str, packages: tuple,
                       description: str, example: str):
    def decorator(build):
        class Adapter:
            make = staticmethod(build)

            @staticmethod
            def check_job(job) -> None:
                hook = getattr(build, "check", None)
                if hook is not None:
                    hook(job)

            @staticmethod
            def gpu_unsupported(settings):
                """Why the GPU path cannot take these settings (the estimator's own gpu_unsupported), or None."""
                hook = getattr(build, "gpu_unsupported", None)
                return hook(settings) if hook is not None else None

        Adapter.__name__ = f"Estimator_{name}"
        Adapter.__module__ = build.__module__       # where the estimator was written (template or project)
        register_model(name, settings=settings, family="estimator", tasks=tasks, kinds=("tabular",), gpu=gpu,
                       profile=profile, packages=packages, description=description, example=example)(Adapter)
        return build
    return decorator


def _multilabel(model, task: str, native: bool):
    if task != "multilabel" or native:
        return model
    from sklearn.multioutput import MultiOutputClassifier

    return MultiOutputClassifier(model)


def _weight(value: str):
    return None if value == "none" else value


@dataclasses.dataclass(frozen=True)
class LogisticSettings:
    c: float | None                 # inverse regularization strength; None: no penalty
    l1_ratio: float                 # 0: L2 penalty, 1: L1, between: elastic net (scikit-learn 1.8 and later)
    class_weight: Literal["none", "balanced"]
    max_iter: int
    extra: dict[str, Scalar]        # any other LogisticRegression parameter, for example {"tol": 1e-5}


@register_estimator("logistic_regression", LogisticSettings, CLASSIFY, "cuml", "core", ("sklearn",),
                    "logistic regression (scikit-learn; cuML on the GPU)", "classify_logistic")
def logistic(s: LogisticSettings, task: str, seed: int, device: str):
    if device == "cuda":
        from cuml.linear_model import LogisticRegression

        if s.c is None:
            penalty = "none"
        else:
            penalty = "l2" if s.l1_ratio == 0 else ("l1" if s.l1_ratio == 1 else "elasticnet")
        kwargs = {"penalty": penalty, "C": 1.0 if s.c is None else s.c, "max_iter": s.max_iter,
                  "class_weight": _weight(s.class_weight)}
        if penalty == "elasticnet":
            kwargs["l1_ratio"] = s.l1_ratio
        return _multilabel(LogisticRegression(**merged_kwargs(kwargs, s.extra, "logistic_regression")), task, False)
    import math

    from sklearn.linear_model import LogisticRegression

    kwargs = {"C": math.inf if s.c is None else s.c, "l1_ratio": s.l1_ratio, "class_weight": _weight(s.class_weight),
              "max_iter": s.max_iter, "random_state": seed,
              "solver": "lbfgs" if (s.c is None or s.l1_ratio == 0) else "saga"}
    return _multilabel(LogisticRegression(**merged_kwargs(kwargs, s.extra, "logistic_regression")), task, False)


def _check_logistic(job):
    if not 0 <= job.model.l1_ratio <= 1:
        raise ValueError("logistic_regression: l1_ratio in [0, 1]")
    if job.model.c is not None and not job.model.c > 0:
        raise ValueError("logistic_regression: c must be positive (None for no penalty)")


logistic.check = _check_logistic


@dataclasses.dataclass(frozen=True)
class LinearSVMSettings:
    c: float
    class_weight: Literal["none", "balanced"]
    max_iter: int
    extra: dict[str, Scalar]


@register_estimator("linear_svm", LinearSVMSettings, CLASSIFY, "cuml", "core", ("sklearn",),
                    "linear support vector machine (decision values as scores)", "classify_svm")
def linear_svm(s: LinearSVMSettings, task: str, seed: int, device: str):
    if device == "cuda":
        from cuml.svm import LinearSVC

        model = LinearSVC(C=s.c, max_iter=s.max_iter, class_weight=_weight(s.class_weight), **s.extra)
    else:
        from sklearn.svm import LinearSVC

        model = LinearSVC(**merged_kwargs({"C": s.c, "class_weight": _weight(s.class_weight), "max_iter": s.max_iter,
                                           "random_state": seed}, s.extra, "linear_svm"))
    return _multilabel(model, task, False)


@dataclasses.dataclass(frozen=True)
class SVMSettings:
    kernel: Literal["rbf", "poly", "sigmoid", "linear"]
    c: float
    gamma: float | None             # None: scikit-learn's "scale", 1 / (columns x variance)
    degree: int                     # poly only (ignored otherwise; give 3)
    class_weight: Literal["none", "balanced"]
    probability: bool               # Platt-scaled probabilities (slower); False: decision values as scores
    extra: dict[str, Scalar]


@register_estimator("svm", SVMSettings, ALL, "cuml", "core", ("sklearn",),
                    "kernel support vector machine (SVC, SVR)", "classify_svm")
def svm(s: SVMSettings, task: str, seed: int, device: str):
    gamma = "scale" if s.gamma is None else s.gamma
    if task == "regression":
        if device == "cuda":
            from cuml.svm import SVR
        else:
            from sklearn.svm import SVR
        return SVR(**merged_kwargs({"kernel": s.kernel, "C": s.c, "gamma": gamma, "degree": s.degree}, s.extra, "svm"))
    if device == "cuda":
        from cuml.svm import SVC

        reason = _svm_gpu_unsupported(s)
        if reason is not None:
            raise ValueError(reason)
        model = SVC(**merged_kwargs({"kernel": s.kernel, "C": s.c, "gamma": gamma, "degree": s.degree,
                                     "class_weight": _weight(s.class_weight), "random_state": seed}, s.extra, "svm"))
    else:
        from sklearn.svm import SVC

        kwargs = {"kernel": s.kernel, "C": s.c, "gamma": gamma, "degree": s.degree,
                  "class_weight": _weight(s.class_weight), "random_state": seed}
        if s.probability:
            # scikit-learn 1.9 deprecates SVC's probability parameter and warns whenever it is given (1.11 removes
            # it; CalibratedClassifierCV(SVC(), ensemble=False) replaces it). Without it SVC gives no probabilities,
            # so it is passed only when they are wanted.
            kwargs["probability"] = True
        model = SVC(**merged_kwargs(kwargs, s.extra, "svm"))
    return _multilabel(model, task, False)


def _svm_gpu_unsupported(s) -> str | None:
    # cuML 26.10's SVC has no probability parameter and no predict_proba (its documentation lists decision_function,
    # fit and predict); the first node run, 2026-10-27, failed with "unexpected keyword argument 'probability'"
    if s.probability:
        return "cuML's SVC gives no probabilities (probability true)"
    return None


svm.gpu_unsupported = _svm_gpu_unsupported


def _check_svm(job):
    if job.task_name == "regression" and job.model.class_weight != "none":
        raise ValueError("svm: class_weight is none for regression")


svm.check = _check_svm


@dataclasses.dataclass(frozen=True)
class KNNSettings:
    n_neighbors: int
    weights: Literal["uniform", "distance"]
    metric: Literal["euclidean", "manhattan", "chebyshev", "cosine", "minkowski"]
    extra: dict[str, Scalar]


@register_estimator("knn", KNNSettings, ALL, "cuml", "core", ("sklearn",),
                    "k-nearest neighbours (classifier or regressor)", "classify_knn")
def knn(s: KNNSettings, task: str, seed: int, device: str):
    if device == "cuda":
        from cuml.neighbors import KNeighborsClassifier, KNeighborsRegressor
    else:
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    cls = KNeighborsRegressor if task == "regression" else KNeighborsClassifier
    return cls(**merged_kwargs({"n_neighbors": s.n_neighbors, "weights": s.weights, "metric": s.metric}, s.extra,
                               "knn"))


@dataclasses.dataclass(frozen=True)
class NaiveBayesSettings:
    variant: Literal["gaussian", "multinomial", "bernoulli", "complement"]
    smoothing: float                # gaussian: var_smoothing; the others: alpha
    extra: dict[str, Scalar]


@register_estimator("naive_bayes", NaiveBayesSettings, CLASSIFY, "none", "core", ("sklearn",),
                    "naive Bayes (Gaussian, multinomial, Bernoulli, complement)", "classify_bayes")
def naive_bayes(s: NaiveBayesSettings, task: str, seed: int, device: str):
    from sklearn import naive_bayes as nb

    if s.variant == "gaussian":
        model = nb.GaussianNB(**merged_kwargs({"var_smoothing": s.smoothing}, s.extra, "naive_bayes"))
    else:
        cls = {"multinomial": nb.MultinomialNB, "bernoulli": nb.BernoulliNB, "complement": nb.ComplementNB}[s.variant]
        model = cls(**merged_kwargs({"alpha": s.smoothing}, s.extra, "naive_bayes"))
    return _multilabel(model, task, False)


@dataclasses.dataclass(frozen=True)
class TreeSettings:
    max_depth: int | None
    min_samples_leaf: int
    max_features: float | None      # share of columns tried per split; None: all
    class_weight: Literal["none", "balanced"]
    extra: dict[str, Scalar]


@register_estimator("decision_tree", TreeSettings, ALL, "none", "core", ("sklearn",), "one decision tree",
                    "classify_trees")
def decision_tree(s: TreeSettings, task: str, seed: int, device: str):
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    kwargs = {"max_depth": s.max_depth, "min_samples_leaf": s.min_samples_leaf, "max_features": s.max_features,
              "random_state": seed}
    if task == "regression":
        return DecisionTreeRegressor(**merged_kwargs(kwargs, s.extra, "decision_tree"))
    return DecisionTreeClassifier(**merged_kwargs({**kwargs, "class_weight": _weight(s.class_weight)}, s.extra,
                                                  "decision_tree"))


@dataclasses.dataclass(frozen=True)
class ForestSettings:
    n_estimators: int
    max_depth: int | None
    min_samples_leaf: int
    max_features: float | None      # share of columns tried per split; None: all
    bootstrap: bool
    class_weight: Literal["none", "balanced"]
    n_jobs: int                     # CPU processes (-1: all)
    extra: dict[str, Scalar]


def _forest(kind: str):
    def build(s: ForestSettings, task: str, seed: int, device: str):
        if device == "cuda" and kind == "random_forest":
            from cuml.ensemble import RandomForestClassifier, RandomForestRegressor

            cls = RandomForestRegressor if task == "regression" else RandomForestClassifier
            return cls(n_estimators=s.n_estimators, max_depth=s.max_depth if s.max_depth is not None else 16,
                       min_samples_leaf=s.min_samples_leaf,
                       max_features=s.max_features if s.max_features is not None else 1.0,
                       bootstrap=s.bootstrap, random_state=seed, **s.extra)
        from sklearn import ensemble

        names = {"random_forest": ("RandomForestClassifier", "RandomForestRegressor"),
                 "extra_trees": ("ExtraTreesClassifier", "ExtraTreesRegressor")}[kind]
        cls = getattr(ensemble, names[1] if task == "regression" else names[0])
        kwargs = {"n_estimators": s.n_estimators, "max_depth": s.max_depth, "min_samples_leaf": s.min_samples_leaf,
                  "max_features": s.max_features, "bootstrap": s.bootstrap, "n_jobs": s.n_jobs, "random_state": seed}
        if task != "regression":
            kwargs["class_weight"] = _weight(s.class_weight)
        return cls(**merged_kwargs(kwargs, s.extra, kind))

    def check(job):
        if job.task_name == "regression" and job.model.class_weight != "none":
            raise ValueError(f"{kind}: class_weight is none for regression")
        if job.fit.device == "cuda" and kind == "random_forest" and job.model.max_depth is None:
            raise ValueError("random_forest on cuML needs a max_depth (cuML has no unlimited depth)")
    build.check = check
    return build


register_estimator("random_forest", ForestSettings, ALL, "cuml", "core", ("sklearn",),
                   "random forest (scikit-learn; cuML on the GPU)", "classify_trees")(_forest("random_forest"))
register_estimator("extra_trees", ForestSettings, ALL, "none", "core", ("sklearn",),
                   "extremely randomized trees", "classify_trees")(_forest("extra_trees"))


@dataclasses.dataclass(frozen=True)
class HistGBSettings:
    learning_rate: float
    max_iter: int                   # boosting rounds
    max_leaf_nodes: int | None
    max_depth: int | None
    min_samples_leaf: int
    l2_regularization: float
    early_stopping: bool            # scikit-learn's own split of the training rows (not our validation rows)
    class_weight: Literal["none", "balanced"]
    extra: dict[str, Scalar]


@register_estimator("hist_gradient_boosting", HistGBSettings, ("classification", "regression"), "none", "core",
                    ("sklearn",), "histogram gradient boosting (scikit-learn)", "classify_boosting")
def hist_gb(s: HistGBSettings, task: str, seed: int, device: str):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    kwargs = {"learning_rate": s.learning_rate, "max_iter": s.max_iter, "max_leaf_nodes": s.max_leaf_nodes,
              "max_depth": s.max_depth, "min_samples_leaf": s.min_samples_leaf,
              "l2_regularization": s.l2_regularization, "early_stopping": s.early_stopping, "random_state": seed}
    if task == "regression":
        return HistGradientBoostingRegressor(**merged_kwargs(kwargs, s.extra, "hist_gradient_boosting"))
    return HistGradientBoostingClassifier(**merged_kwargs({**kwargs, "class_weight": _weight(s.class_weight)},
                                                          s.extra, "hist_gradient_boosting"))


@dataclasses.dataclass(frozen=True)
class BoostSettings:
    n_estimators: int
    learning_rate: float
    max_depth: int | None           # XGBoost: tree depth; LightGBM: -1 is unlimited, so None means -1
    num_leaves: int | None          # LightGBM only (None for XGBoost)
    subsample: float
    colsample_bytree: float
    reg_lambda: float
    n_jobs: int
    extra: dict[str, Scalar]


@register_estimator("xgboost", BoostSettings, ("classification", "regression"), "xgboost", "boosting", ("xgboost",),
                    "XGBoost gradient boosting (CPU, or CUDA)", "classify_boosting")
def xgboost_model(s: BoostSettings, task: str, seed: int, device: str):
    import xgboost

    cls = xgboost.XGBRegressor if task == "regression" else xgboost.XGBClassifier
    return cls(**merged_kwargs({"n_estimators": s.n_estimators, "learning_rate": s.learning_rate,
                                "max_depth": s.max_depth, "subsample": s.subsample,
                                "colsample_bytree": s.colsample_bytree, "reg_lambda": s.reg_lambda,
                                "n_jobs": s.n_jobs, "random_state": seed, "device": device,
                                "tree_method": "hist"}, s.extra, "xgboost"))


def _check_xgb(job):
    if job.model.num_leaves is not None:
        raise ValueError("xgboost: num_leaves is LightGBM's; give None")


xgboost_model.check = _check_xgb


@register_estimator("lightgbm", BoostSettings, ("classification", "regression"), "none", "boosting", ("lightgbm",),
                    "LightGBM gradient boosting (CPU)", "classify_boosting")
def lightgbm_model(s: BoostSettings, task: str, seed: int, device: str):
    import lightgbm

    cls = lightgbm.LGBMRegressor if task == "regression" else lightgbm.LGBMClassifier
    return cls(**merged_kwargs({"n_estimators": s.n_estimators, "learning_rate": s.learning_rate,
                                "max_depth": -1 if s.max_depth is None else s.max_depth,
                                "num_leaves": s.num_leaves, "subsample": s.subsample, "subsample_freq": 1,
                                "colsample_bytree": s.colsample_bytree, "reg_lambda": s.reg_lambda,
                                "n_jobs": s.n_jobs, "random_state": seed, "verbose": -1}, s.extra, "lightgbm"))


def _check_lgbm(job):
    if job.model.num_leaves is None:
        raise ValueError("lightgbm: give num_leaves (LightGBM's main size setting)")


lightgbm_model.check = _check_lgbm


@dataclasses.dataclass(frozen=True)
class LinearRegressionSettings:
    penalty: Literal["none", "ridge", "lasso", "elastic_net"]
    alpha: float                    # regularization strength (0 with penalty none)
    l1_ratio: float | None          # elastic_net only
    fit_intercept: bool
    max_iter: int
    extra: dict[str, Scalar]


@register_estimator("linear", LinearRegressionSettings, REGRESS, "cuml", "core", ("sklearn",),
                    "linear regression: ordinary, ridge, lasso or elastic net", "regress_linear")
def linear(s: LinearRegressionSettings, task: str, seed: int, device: str):
    if device == "cuda":
        from cuml import linear_model as lm
    else:
        from sklearn import linear_model as lm
    if s.penalty == "none":
        return lm.LinearRegression(fit_intercept=s.fit_intercept, **s.extra)
    if s.penalty == "ridge":
        return lm.Ridge(alpha=s.alpha, fit_intercept=s.fit_intercept, **s.extra)
    if s.penalty == "lasso":
        return lm.Lasso(alpha=s.alpha, fit_intercept=s.fit_intercept, max_iter=s.max_iter, **s.extra)
    return lm.ElasticNet(alpha=s.alpha, l1_ratio=s.l1_ratio, fit_intercept=s.fit_intercept, max_iter=s.max_iter,
                         **s.extra)


def _check_linear(job):
    s = job.model
    if (s.penalty == "elastic_net") != (s.l1_ratio is not None):
        raise ValueError("linear: l1_ratio is set with penalty elastic_net only")
    if s.penalty == "none" and s.alpha != 0:
        raise ValueError("linear: alpha is 0 with penalty none")


linear.check = _check_linear


@dataclasses.dataclass(frozen=True)
class TPOTSettings:
    generations: int
    population_size: int
    max_time_mins: float | None     # stop the search after this many minutes; None: every generation
    max_eval_time_mins: float       # give up on one pipeline after this many minutes
    n_jobs: int
    extra: dict[str, Scalar]


@register_estimator("tpot", TPOTSettings, ("classification", "regression"), "none", "automl", ("tpot",),
                    "TPOT: genetic programming over scikit-learn pipelines (its own environment)", "classify_tpot")
def tpot_model(s: TPOTSettings, task: str, seed: int, device: str):
    import tpot

    cls = tpot.TPOTRegressor if task == "regression" else tpot.TPOTClassifier
    return cls(**merged_kwargs({"generations": s.generations, "population_size": s.population_size,
                                "max_time_mins": s.max_time_mins, "max_eval_time_mins": s.max_eval_time_mins,
                                "n_jobs": s.n_jobs, "random_state": seed}, s.extra, "tpot"))
