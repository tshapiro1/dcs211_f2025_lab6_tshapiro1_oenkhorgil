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
    ("The servers are welcoming and always made me feel invited.", "positive"),
    ("Great sushi, clean place, very quiet set up inside", "positive"),
    ("The owners make you feel like family and make sure you have the best service and food.", "positive"),
    ("The sashimi was fresh and very refreshing.", "positive"),
    ("It was in such a nice venue, the environment was clean and had such a pleasing atmosphere to it.", "positive"),
    ("One of the best tacos I've had in a while.", "positive"),
    ("The whole vibe of the store is worthy, the service was amazing so friendly, professional, and quick. ", "positive"),
    ('So good!', 'positive'),
    ('Love it!', 'positive'),
    ('Incredible experience from start to finish.', 'positive'),
    ('The ingredients were fresh and perfectly seasoned.', 'positive'),
    ('Cannot recommend this place enough!', 'positive'),
    ('Five stars all the way!', 'positive'),
    ('Delicious and worth every penny.', 'positive'),
    
    # Negative reviews
    ("There was hair in the food, the salsa was not good, and the chips were not fresh.", "negative"),
    ("The steaks and burger came out med-rare instead of medium, my kids dont like to see too much blood", "negative"),
    ("I had to wait for a long time to get my food, the cashier was not even looking at me, he was just talking to his friend.", "negative"),
    ("The burrito was not good, the sauce was too sweet and the rice was not cooked well.", "negative"),
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
    ('Everything is bad. Dont go.', 'negative'), 
    ('I was disgusted', 'negative'),
    ('Disappointing meal, expected much better.', 'negative'),
    ('The portions were tiny and overpriced.', 'negative'),
    ('Way too greasy, felt sick after.', 'negative'),
    ('Stale bread and bland soup.', 'negative'),
    ('Mediocre at best, waste of time.', 'negative'),
    ('The fish smelled off, did not eat it.', 'negative'),
    ('Rude staff and dirty tables.', 'negative'),
    ('Never again, horrible quality.', 'negative'),
    ('Too salty and burnt.', 'negative'),
    ('Not fresh at all, very old ingredients.', 'negative'),
    
    # Neutral reviews
    ('I only tried this place because I was given a gift card.', 'neutral'),
    ("Overall, the sushi was decent.", "neutral"),
    ("The fish was fresh and the sushi was delicious. The only downside was the price.", "neutral"),
    ("The sushi was good, but the service was slow.", "neutral"),
    ("It's a pretty straightforward ordering process as well where you queue and order, and basically just wait a while.", "neutral"),
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
    ('The dessert included pudding', 'neutral'),
    ("It's okay, nothing special.", 'neutral'),
    ('Average food, average price.', 'neutral'),
    ('The food was good but the service was slow.', 'neutral'),
    ('Decent burgers, small parking lot.', 'neutral'),
    ('Meh, probably won\'t come back.', 'neutral'),
    ('Solid choice if you\'re in the area.', 'neutral'),
    ('The decor was nice but the food was just okay.', 'neutral'),
    ('Quick service but nothing memorable.', 'neutral'),
    ('They offer takeout and delivery.', 'neutral'),
    ('Standard diner food, typical prices.', 'neutral'),
]

# Step 2: Create the classifier with training data
cl = NaiveBayesClassifier(train)

# Step 3: Test the classifier
print("Restaurant Review Classifier")
print()

another_review = 'yes'

while another_review.lower() == 'yes':
    # Get review from user
    test_review = input("Enter a review to classify: ")
    
    # Classify and display result
    result = cl.classify(test_review)
    
    # Get probability distribution for confidence
    prob_dist = cl.prob_classify(test_review)
    confidence = round(prob_dist.prob(result), 2)
    
    print(f"Review: '{test_review}'")
    print(f"Classification: {result}")
    print(f"Confidence: {confidence * 100}%")
    print()
    
    # Ask if they want to enter another review
    another_review = input("Would you like to enter another review? (yes/no): ")
    print()
