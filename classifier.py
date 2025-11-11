'''
Restaurant/Meal Review Classifier
''' 

from textblob.classifiers import NaiveBayesClassifier

# Step 1: Create training data with 20+ (generated) examples
# Format: (text, classification)
train = [
    # Positive reviews
    ('The pizza was amazing and delicious.', 'positive'),
    ('Best burger I ever had!', 'positive'),
    ('Great service and wonderful food.', 'positive'),
    ('Absolutely loved this place, will definitely come back!', 'positive'),
    ('The pasta was perfectly cooked and the sauce was incredible.', 'positive'),
    ('Outstanding meal, every dish was fantastic.', 'positive'),
    ('Excellent food and friendly staff.', 'positive'),
    ('This is my new favorite restaurant!', 'positive'),
    ('The burgers were juicy, flavorful, and satisfying in that deeply American way that only a Five Guys meal can be.', 'positive'),
    ('The fries with the iconic peel were fantastic and the service was also pretty good.', 'positive'),
    ('There is a soda fountain with so many drink options. And their fries are just the best you can get.', 'positive'),
    ('I’m so glad I gave this place a try. The service was great and the food was delicious.', 'positive'),
    ('Food tasted great and the dessert was unexpectedly yummy.', 'positive'),
    ('I only tried this place because I was given a gift card.', 'neutral'),
    ("The steaks and burger came out med-rare instead of medium, my kids dont like to see too much blood", "negative"),
    ("I had to wait for a long time to get my food, the cashier was not even looking at me, he was just talking to his friend.", "negative"),
    ("The servers are welcoming and always made me feel invited.", "positive"),
    ("Overall, the sushi was decent.", "neutral"),
    ("Great sushi, clean place, very quiet set up inside", "positive"),
    ("The fish was fresh and the sushi was delicious. The only downside was the price.", "neutral"),
    ("The owners make you feel like family and make sure you have the best service and food.", "positive"),
    ("The sashimi was fresh and very refreshing.", "positive"),
    ("The sushi was good, but the service was slow.", "neutral"),
    ("It was in such a nice venue, the environment was clean and had such a pleasing atmosphere to it.", "positive"),
    ("The burrito was not good, the sauce was too sweet and the rice was not cooked well.", "negative"),
    ("It's a pretty straightforward ordering process as well where you queue and order, and basically just wait a while.", "neutral"),
    ("One of the best tacos I've had in a while.", "positive"),
    ("The whole vibe of the store is worthy, the service was amazing so friendly, professional, and quick. ", "positive"),
    ("There was hair in the food, the salsa was not good, and the chips were not fresh.", "negative"),
     # Negative reviews
    ('This restaurant is terrible.', 'negative'),
    ('Worst meal ever, never coming back.', 'negative'),
    ('The food was cold and bland.', 'negative'),
    ('Horrible service, waited an hour for my food.', 'negative'),
    ('Overpriced and tasteless, very disappointed.', 'negative'),
    ('The chicken was undercooked and unsafe to eat.', 'negative'),
    ('Awful experience, the place was dirty.', 'negative'),
    ('Would not recommend, complete waste of money.', 'negative'),
    ('I felt ripped off. The French fries were too salty and cooked in burnt oil.', 'negative'),
    ('The cashier was rude and careless. He didn’t even bother to speak, everything was on signs.', 'negative'),
    ('He took out the trash and hit every customer on his way, he even drop a table on a customer. Disappointed of the service.', 'negative'),
    
    # Neutral reviews
    ('The food was delivered', 'neutral'),
    ('It is located next to a bank.', 'neutral'),
    ('They have fish tacos', 'neutral'),
    ('They have a wall covered with magazines.', 'neutral'),
    ('They have booths and chairs as seating', 'neutral'),
    ('The pizza came with fries', 'neutral'),
    ('The restaurant is on Main street', 'neutral'),
    ('There is an option to add vegetables.', 'neutral'),
    ('The soup has potato and peas', 'neutral'),
    ('The dessert included ice cream', 'neutral'),
    ('There is a window next to the cashier', 'neutral'),
    ('The dessert included pudding', 'neutral')
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
