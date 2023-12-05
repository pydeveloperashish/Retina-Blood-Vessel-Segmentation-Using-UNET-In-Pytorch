from flask import Flask, request, render_template, redirect, url_for
import os

app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def home():
    return "Hello Ashish, this is a trial application..."

if __name__ == "__main__":
    app.run(host = "0.0.0.0", port = 8080)
