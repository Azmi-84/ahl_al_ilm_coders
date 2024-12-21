from flask import Blueprint, request, jsonify
from api.app import db, Recipe

recipes_bp = Blueprint('recipes', __name__)

@recipes_bp.route('/recipes', methods=['GET'])
def get_recipes():
    filters = request.args
    query = Recipe.query
    if 'taste' in filters:
        query = query.filter(Recipe.taste.ilike(f"%{filters['taste']}%"))
    if 'ingredient' in filters:
        query = query.filter(Recipe.ingredients.contains(filters['ingredient']))
    recipes = query.all()
    return jsonify([{
        "id": r.id, "name": r.name, "cuisine": r.cuisine,
        "preparation_time": r.preparation_time, "taste": r.taste
    } for r in recipes])

@recipes_bp.route('/recipes', methods=['POST'])
def add_recipe():
    data = request.get_json()
    new_recipe = Recipe(
        name=data['name'],
        cuisine=data.get('cuisine', ''),
        taste=data.get('taste', ''),
        preparation_time=data.get('preparation_time', 0),
        ingredients=data.get('ingredients', {}),
        steps=data.get('steps', '')
    )
    db.session.add(new_recipe)
    db.session.commit()
    return jsonify({"message": "Recipe added successfully!"})
