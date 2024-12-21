from flask import Blueprint, request, jsonify
from api.app import db, Ingredient

ingredients_bp = Blueprint('ingredients', __name__)

@ingredients_bp.route('/ingredients', methods=['GET'])
def get_ingredients():
    ingredients = Ingredient.query.all()
    return jsonify([{
        "id": i.id, "name": i.name, "quantity": i.quantity,
        "unit": i.unit, "last_updated": i.last_updated
    } for i in ingredients])

@ingredients_bp.route('/ingredients', methods=['POST'])
def add_ingredient():
    data = request.get_json()
    new_ingredient = Ingredient(
        name=data['name'],
        quantity=data.get('quantity', ''),
        unit=data.get('unit', '')
    )
    db.session.add(new_ingredient)
    db.session.commit()
    return jsonify({"message": "Ingredient added successfully!"})

@ingredients_bp.route('/ingredients/<int:id>', methods=['DELETE'])
def delete_ingredient(id):
    ingredient = Ingredient.query.get(id)
    if not ingredient:
        return jsonify({"message": "Ingredient not found"}), 404
    db.session.delete(ingredient)
    db.session.commit()
    return jsonify({"message": "Ingredient deleted successfully!"})
