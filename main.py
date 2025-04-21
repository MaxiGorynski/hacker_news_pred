from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from pydantic import BaseModel
import torch
import wandb
from Hacker_News_Score_Predictor import call_and_response as model

def load_model_from_wandb():
    wandb.login()
    run = wandb.init(project="hack-news-predict", job_type="inference")

    artifact = run.use_artifact("joshua-oyekunle10/hacker-news-model:model", type="model")
    model_dir = artifact.download()
    model.load_state_dict(torch.load(f"{model_dir}/model.pth", map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    model.eval()
    
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
class InputData(BaseModel):
    title: str
    link: str
    user: str

@app.get("/", response_class=HTMLResponse)
async def form(request: Request):
    return templates.TemplateResponse("plain.html", {"request":request})

@app.post("/predictions")
async def get_prediction(data: InputData):
    result: int = model(data.title)         
    return {"prediction": result}