from app import app

@app.route("/")
def home():
    return "Top bar <h1>Header<h1>"