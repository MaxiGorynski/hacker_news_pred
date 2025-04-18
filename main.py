from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse

model.load("./gpu-files/hn_files/HN_Upvote_Predictor_Weights.pth")

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
    # your “model” that returns an int
    result: int = len(title)          
    return {"prediction": result}