from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import os
from Hacker_News_Score_Predictor import call_and_response as model

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def form():
    return """
    <html>
      <head><title>Enter a Title</title></head>
      <body>
        <h1>Enter a Title</h1>
        <!-- form uses POST, so we switch our predictions route to @app.post -->
        <form action="/predictions" method="post">
          <input type="text" name="title" placeholder="Your title here"/>
          <button type="submit">Submit</button>
        </form>
      </body>
    </html>
    """

@app.post("/predictions")
async def get_prediction(title: str = Form(...)):
    result: int = model(title)         
    return {"prediction": result}