import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


def _install_numpy_pickle_shims() -> None:
    """Provide backward/forward compatibility for pickles created with different NumPy versions.

    Some artifacts reference internal modules like `numpy._core.multiarray` (NumPy 2.x layout).
    In NumPy 1.x these live under `numpy.core.*`. We register aliases in `sys.modules`
    so unpickling can resolve them.
    """

    import types

    # Ensure `numpy._core` is treated as a *package* so submodule imports work.
    core_pkg = sys.modules.get("numpy._core")
    if core_pkg is None or not hasattr(core_pkg, "__path__"):
        core_pkg = types.ModuleType("numpy._core")
        core_pkg.__path__ = []  # mark as package
        sys.modules["numpy._core"] = core_pkg

    # Common internal modules referenced by pickles.
    try:
        import numpy.core.multiarray as _multiarray

        sys.modules.setdefault("numpy._core.multiarray", _multiarray)
    except Exception:
        pass


def _install_sklearn_pickle_shims() -> None:
    """Provide compatibility for pickles created with different scikit-learn versions.

    scikit-learn sometimes pickles private helper classes (e.g. ColumnTransformer internals).
    When those names move/remove between versions, unpickling raises AttributeError.
    We add minimal stand-ins to allow older artifacts to load.
    """

    try:
        import sklearn.compose._column_transformer as _ct

        if not hasattr(_ct, "_RemainderColsList"):
            class _RemainderColsList(list):
                pass

            _ct._RemainderColsList = _RemainderColsList
    except Exception:
        # If sklearn isn't available or internals changed, let unpickling raise the original error.
        pass

    try:
        import numpy.core._multiarray_umath as _multiarray_umath

        sys.modules.setdefault("numpy._core._multiarray_umath", _multiarray_umath)
    except Exception:
        pass

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        _install_numpy_pickle_shims()
        _install_sklearn_pickle_shims()

        with open(file_path, "rb") as file_obj:
            try:
                obj = pickle.load(file_obj)
            except Exception:
                # Some artifacts are saved with dill; retry from start.
                file_obj.seek(0)
                obj = dill.load(file_obj)

        # Patch up sklearn objects for version compatibility.
        try:
            from sklearn.compose import ColumnTransformer

            if isinstance(obj, ColumnTransformer):
                # Newer sklearn versions may expect this attribute to exist.
                if not hasattr(obj, "_name_to_fitted_passthrough"):
                    obj._name_to_fitted_passthrough = {}
        except Exception:
            pass

        return obj

    except Exception as e:
        raise CustomException(e, sys)
