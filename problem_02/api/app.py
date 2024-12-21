from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:postgres@localhost:5432/kitchen_buddy'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Models
class Ingredient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    quantity = db.Column(db.String(50))
    unit = db.Column(db.String(50))
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)

class Recipe(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    cuisine = db.Column(db.String(50))
    taste = db.Column(db.String(50))
    preparation_time = db.Column(db.Integer)
    ingredients = db.Column(db.JSON)
    steps = db.Column(db.Text)
    reviews = db.Column(db.Float, default=0.0)

db.create_all()

@app.route('/')
def home():
    return "Kitchen Buddy API is running!"

if __name__ == '__main__':
    app.run(debug=True)
