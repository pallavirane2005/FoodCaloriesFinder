# calories.py
"""
Food Calorie Database
Values are calories per 100g (approximate averages)
"""

FOOD_CALORIES = {
    'Bread': 265,
    'Dairy product': 150,
    'Dessert': 350,
    'Egg': 155,
    'Fried food': 320,
    'Meat': 250,
    'Noodles-Pasta': 160,
    'Rice': 130,
    'Seafood': 200,
    'Soup': 50,
    'Vegetable-Fruit': 45,
}

CLASS_NAMES = list(FOOD_CALORIES.keys())

def get_calories(food_class, portion_grams=100):
    cal_per_100g = FOOD_CALORIES.get(food_class, 0)
    total_calories = (cal_per_100g * portion_grams) / 100
    return {
        'food_class': food_class,
        'calories_per_100g': cal_per_100g,
        'portion_grams': portion_grams,
        'total_calories': round(total_calories, 1)
    }

CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
INDEX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}

if __name__ == '__main__':
    print("Classes:", CLASS_NAMES)
    print(get_calories('Bread', 150))