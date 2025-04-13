from typing import Annotated
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from app.services.train_service import train_classification_model, train_clustering_model, train_regression_model
from sqlalchemy.orm import Session
from ..database import SessionLocal
from ..db_models import Model, Users
from .auth import get_current_user

router = APIRouter(
    prefix="/train",
    tags=["train"]
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


db_dependency = Annotated[Session, Depends(get_db)]

MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB


@router.get("/saved-models")
async def get_saved_models(
    type_of_model: str = None,
    db: Session = Depends(get_db),
    current_user: Users = Depends(get_current_user)
):
    try:
        query = db.query(Model).filter(Model.owner_id == current_user.id)

        if type_of_model:
            query = query.filter(Model.model_type == type_of_model.casefold())

        saved_models = query.all()

        return [
            {
                "id": model.id,
                "model_name": model.model_name,
                "model_type": model.model_type,
                "metrics": model.metrics,
                "target_column": model.target_column,
                "training_data_path": model.training_data_path,
                "status": model.status,
                "hyperparameters": model.hyperparameters,
                "user_id": model.owner_id
            }
            for model in saved_models
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/classification")
async def train_model_classification(
    db: db_dependency,
    file: UploadFile = File(...),
    target_column: str = Form(...),
    custom_model_name: str = Form(None),
    current_user: Users = Depends(get_current_user)
):
    file_content = await file.read()
    file_size = len(file_content)

    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File size exceeds 25 MB limit. Please upload a smaller file."
        )

    try:
        result = train_classification_model(file_content, target_column, db, current_user.id, custom_model_name)

        if 'model_name' not in result or 'metrics' not in result:
            raise HTTPException(status_code=500, detail="Model training failed, missing required results.")

        return {
            "status": "success",
            "model_path": result["model_path"],
            "metrics": result["metrics"],
            "run_id": result["run_id"],
            "model_id": result["model_id"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clustering")
async def train_model_clustering(
    db: db_dependency,
    file: UploadFile = File(...),
    custom_model_name: str = Form(None),
    current_user: Users = Depends(get_current_user)
):
    file_content = await file.read()
    file_size = len(file_content)

    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File size exceeds 25 MB limit. Please upload a smaller file."
        )

    try:
        result = train_clustering_model(file_content, db, current_user.id, custom_model_name)

        if 'model_name' not in result or 'metrics' not in result:
            raise HTTPException(status_code=500, detail="Model training failed, missing required results.")

        return {
            "status": "success",
            "model_path": result["model_path"],
            "metrics": result["metrics"],
            "run_id": result["run_id"],
            "model_id": result["model_id"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/regression")
async def train_model_regression(
    db: db_dependency,
    file: UploadFile = File(...),
    target_column: str = Form(...),
    custom_model_name: str = Form(None),
    current_user: Users = Depends(get_current_user)
):
    file_content = await file.read()
    file_size = len(file_content)

    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File size exceeds 25 MB limit. Please upload a smaller file."
        )

    try:
        result = train_regression_model(file_content, target_column, db, current_user.id, custom_model_name)

        return {
            "status": "success",
            "model_path": result["model_path"],
            "metrics": result["metrics"],
            "run_id": result["run_id"],
            "model_id": result["model_id"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/models/{model_id}")
async def delete_model_from_db(
    db: db_dependency,
    model_id: int,
    current_user: Users = Depends(get_current_user)
):
    try:
        model = db.query(Model).filter(Model.id == model_id, Model.owner_id == current_user.id).first()

        if not model:
            raise HTTPException(
                status_code=404,
                detail="Model not found or you do not have permission to delete this model."
                )

        db.delete(model)
        db.commit()

        return {"status": "success", "detail": f"Model with ID {model_id} has been deleted."}
    except Exception as e:
        db.rollback()  # For safety :D
        raise HTTPException(status_code=500, detail=str(e))
