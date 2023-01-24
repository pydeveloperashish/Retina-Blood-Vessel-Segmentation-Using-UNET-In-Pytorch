import os
import numpy as n
from flask import Flask,render_template,request


app = Flask(__name__)


# define the flask app
app = Flask(__name__)


@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')


if __name__=='__main__':
    app.run(debug = False, port = 5010)