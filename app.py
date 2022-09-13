from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return "Top bar <h1>Header<h1>"

if __name__ == "__main__":
    app.run()