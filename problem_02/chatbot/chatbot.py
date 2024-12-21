import openai

openai.api_key = "sk-1234567890abcdef1234567890abcdef"

def get_recipe_suggestion(preferences, available_ingredients):
    prompt = f"""
    Suggest a recipe based on these preferences:
    Preferences: {preferences}
    Ingredients: {', '.join(available_ingredients)}
    """
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150
    )
    return response['choices'][0]['text'].strip()
