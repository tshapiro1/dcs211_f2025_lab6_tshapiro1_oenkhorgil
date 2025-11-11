'''
Restaurant/Meal Review Classifier
''' 

from textblob.classifiers import NaiveBayesClassifier

# Step 1: Create training data with 20+ (generated) examples
# Format: (text, classification)
train = [
    # Positive reviews
    ('The pizza was amazing and delicious.', 'pos'),
    ('Best burger I ever had!', 'pos'),
    ('Great service and wonderful food.', 'pos'),
    ('Absolutely loved this place, will definitely come back!', 'pos'),
    ('The pasta was perfectly cooked and the sauce was incredible.', 'pos'),
    ('Outstanding meal, every dish was fantastic.', 'pos'),
    ('Excellent food and friendly staff.', 'pos'),
    ('This is my new favorite restaurant!', 'pos'),
    
    # Negative reviews
    ('This restaurant is terrible.', 'neg'),
    ('Worst meal ever, never coming back.', 'neg'),
    ('The food was cold and bland.', 'neg'),
    ('Horrible service, waited an hour for my food.', 'neg'),
    ('Overpriced and tasteless, very disappointed.', 'neg'),
    ('The chicken was undercooked and unsafe to eat.', 'neg'),
    ('Awful experience, the place was dirty.', 'neg'),
    ('Would not recommend, complete waste of money.', 'neg'),
    
    # Neutral reviews
    ('The food was okay, nothing special.', 'neu'),
    ('It was decent, I might come back.', 'neu'),
    ('Average restaurant, not bad but not great.', 'neu'),
    ('The meal was acceptable, prices were fair.', 'neu'),
    ('Standard diner food, what you would expect.', 'neu'),
    ('It was fine, service was average.', 'neu'),
]

# Step 2: Create the classifier with training data
cl = NaiveBayesClassifier(train)

# Step 3: Test the classifier
print("Testing the Basic Classifier:")
test_review = input("Enter a review to classify: ")
result = cl.classify(test_review)
print(f"Review: '{test_review}'")
print(f"Classification: {result}")
print()

test_review2 = input("Enter a review to classify: ")
result2 = cl.classify(test_review2)
print(f"Review: '{test_review2}'")
print(f"Classification: {result2}")
