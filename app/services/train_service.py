import os
import pandas as pd
from sqlalchemy.orm import Session
from ..db_models import Model
from ..database import SessionLocal
import tempfile
import json


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def save_model_to_db(
        db: Session,
        model_name: str,
        model_type: str,
        model_file: bytes,
        metrics: dict,
        owner_id: int,
        target_column: str = None,
        hyperparameters: str = None,
        training_data_path: str = None,
        ):
    try:
        metrics_json = json.dumps(metrics) if isinstance(metrics, dict) else metrics

        db_model = Model(
            model_name=model_name,
            model_type=model_type,
            model_file=model_file,
            metrics=metrics_json,
            target_column=target_column,
            training_data_path=training_data_path,
            hyperparameters=hyperparameters,
            status="trained",
            owner_id=owner_id,
        )
        db.add(db_model)
        db.commit()
        db.refresh(db_model)

        print(f"Model ID after save: {db_model.id}")  # Debugging line

        return db_model
    except Exception as e:
        db.rollback()
        raise Exception(f"Error saving model to DB: {e}")


def train_classification_model(
        file_content: bytes,
        target_column: str,
        db: Session, owner_id: int,
        custom_model_name: str = None
        ):
    from pycaret.classification import setup, compare_models, save_model
    try:
        with tempfile.NamedTemporaryFile(delete=False, mode='wb', suffix='.csv') as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name

        df = pd.read_csv(tmp_file_path)
        s = setup(data=df, target=target_column, html=False)
        best_model = compare_models()

        if custom_model_name:
            model_name = custom_model_name
        else:
            model_name = os.path.splitext(os.path.basename(tmp_file_path))[0]

        save_path = os.path.join("app", "models", f"{model_name}_classification_model")
        save_model(best_model, save_path)

        with open(f"{save_path}.pkl", "rb") as model_file:
            model_file_content = model_file.read()

        metrics = s.pull().to_dict(orient="records")[:1]

        saved_model = save_model_to_db(
            db,
            model_name=model_name,
            model_type="classification",
            model_file=model_file_content,
            metrics=metrics,
            owner_id=owner_id,
            target_column=target_column,
        )

        return {
            "model_path": f"{save_path}.pkl",
            "run_id": model_name,
            "metrics": metrics,
            "model_name": model_name,
            "model_id": saved_model.id
        }
    except Exception as e:
        raise Exception(f"Error during model training: {e}")


def train_clustering_model(file_content: bytes, db: Session, owner_id: int, custom_model_name: str = None):
    from pycaret.clustering import setup, create_model, save_model
    try:
        with tempfile.NamedTemporaryFile(delete=False, mode='wb', suffix='.csv') as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name

        df = pd.read_csv(tmp_file_path)
        s = setup(data=df, html=False)
        model = create_model("kmeans")

        if custom_model_name:
            model_name = custom_model_name
        else:
            model_name = os.path.splitext(os.path.basename(tmp_file_path))[0]

        save_path = os.path.join("app", "models", f"{model_name}_clustering_model")
        save_model(model, save_path)

        with open(f"{save_path}.pkl", "rb") as model_file:
            model_file_content = model_file.read()

        metrics = s.pull().to_dict(orient="records")[:1]

        saved_model = save_model_to_db(
            db=db,
            model_name=model_name,
            model_type="clustering",
            model_file=model_file_content,
            metrics=metrics,
            owner_id=owner_id,
            target_column="",
            training_data_path=None,
            hyperparameters=None
        )

        return {
            "model_name": model_name,
            "model_path": f"{save_path}.pkl",
            "run_id": model_name,
            "metrics": metrics,
            "model_id": saved_model.id
        }

    except Exception as e:
        raise Exception(f"Error during clustering model training: {e}")


def train_regression_model(
        file_content: bytes,
        target_column, db: Session,
        owner_id: int,
        custom_model_name: str = None
        ):
    from pycaret.regression import setup, compare_models, pull, save_model

    with tempfile.NamedTemporaryFile(delete=False, mode='wb', suffix='.csv') as tmp_file:
        tmp_file.write(file_content)
        tmp_file_path = tmp_file.name

    df = pd.read_csv(tmp_file_path)

    if custom_model_name:
        model_name = custom_model_name
    else:
        model_name = os.path.splitext(os.path.basename(tmp_file_path))[0]

    with tempfile.TemporaryDirectory() as tmpdir:
        setup(data=df, target=target_column, verbose=False,)

        best_model = compare_models()
        metrics_df = pull()

        model_path = os.path.join(tmpdir, model_name)
        save_model(best_model, model_path)

        with open(f"{model_path}.pkl", "rb") as f:
            model_bytes = f.read()

        saved_model = save_model_to_db(
            db=db,
            model_name=model_name,
            model_type="regression",
            model_file=model_bytes,
            metrics=[metrics_df.iloc[0].to_dict()],
            owner_id=owner_id,
            target_column=target_column
        )

        return {
            "model_path": f"{model_path}.pkl",
            "metrics": [metrics_df.iloc[0].to_dict()],
            "run_id": model_name,
            "model_id": saved_model.id
        }
