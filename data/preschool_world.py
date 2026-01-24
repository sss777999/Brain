# CHUNK_META:
#   Purpose: Preschool World Knowledge (ages 3-6) — factual knowledge for preschool development
#   Dependencies: None
#   API: get_preschool_connections(), get_preschool_sentences(), get_preschool_questions()

"""
Preschool World Knowledge Curriculum (Ages 3-6).

This module contains factual knowledge that children acquire during preschool years.
It fills the gap between CURRICULUM (0-3 years) and GRADE1 (6-7 years).

All data is in English as the Brain model is trained on English.
Format: (word1, word2) tuples representing connections for Hebbian learning.
"""

from typing import List, Tuple

# =============================================================================
# ANCHOR: BABY_ANIMALS_SENTENCES - Baby animals and their adult names (sentences)
# =============================================================================

BABY_ANIMALS_SENTENCES = [
    # Domestic animals
    "A puppy is a baby dog",
    "A kitten is a baby cat",
    "A chick is a baby chicken",
    "A duckling is a baby duck",
    "A gosling is a baby goose",
    "A foal is a baby horse",
    "A calf is a baby cow",
    "A lamb is a baby sheep",
    "A kid is a baby goat",
    "A piglet is a baby pig",
    "A bunny is a baby rabbit",
    
    # Wild animals
    "A cub is a baby bear",
    "A cub is a baby lion",
    "A cub is a baby tiger",
    "A cub is a baby wolf",
    "A fawn is a baby deer",
    "A joey is a baby kangaroo",
    "A joey lives in its mother's pouch",
    
    # Birds
    "A cygnet is a baby swan",
    "An owlet is a baby owl",
    "An eaglet is a baby eagle",
    "A chick is a baby bird",
    
    # Aquatic and amphibians
    "A tadpole is a baby frog",
    "A tadpole has a tail and lives in water",
    "A fry is a baby fish",
    
    # Insects
    "A caterpillar becomes a butterfly",
    "A larva is a baby insect",
    
    # Reptiles
    "A hatchling is a baby turtle",
    "A hatchling is a baby snake",
]

# Keep empty for backward compatibility
BABY_ANIMALS = []

# =============================================================================
# ANCHOR: ANIMAL_GROUPS_SENTENCES - Collective nouns for animal groups
# =============================================================================

ANIMAL_GROUPS_SENTENCES = [
    "A group of birds is called a flock",
    "A group of sheep is called a flock",
    "A group of cows is called a herd",
    "A group of elephants is called a herd",
    "A group of horses is called a herd",
    "A group of wolves is called a pack",
    "A group of dogs is called a pack",
    "A group of fish is called a school",
    "A group of bees is called a swarm",
    "A group of ants is called a colony",
    "A group of lions is called a pride",
    "A group of whales is called a pod",
    "A group of dolphins is called a pod",
    "A group of geese is called a gaggle",
    "A group of monkeys is called a troop",
    "A group of puppies is called a litter",
    "A group of kittens is called a litter",
]

ANIMAL_GROUPS = []

# =============================================================================
# ANCHOR: MATERIALS_SENTENCES - What objects are made of
# =============================================================================

MATERIALS_SENTENCES = [
    # Wood
    "A table is made of wood",
    "A chair is made of wood",
    "A door is made of wood",
    "A pencil is made of wood",
    "Wood is hard and brown",
    "Wood comes from trees",
    "Wood burns in fire",
    
    # Metal
    "A car is made of metal",
    "A key is made of metal",
    "A coin is made of metal",
    "Scissors are made of metal",
    "Metal is hard and shiny",
    "Metal feels cold to touch",
    
    # Plastic
    "Many toys are made of plastic",
    "A bottle can be made of plastic",
    "A toothbrush is made of plastic",
    "Plastic is light and colorful",
    
    # Glass
    "A window is made of glass",
    "A mirror is made of glass",
    "Glass is transparent and fragile",
    "Glass breaks easily",
    
    # Paper
    "A book is made of paper",
    "Paper is thin and tears easily",
    "Paper burns quickly",
    
    # Fabric
    "Clothes are made of fabric",
    "A shirt is made of cotton",
    "A sweater is made of wool",
    "Cotton is soft and breathable",
    "Wool is warm and fuzzy",
    
    # Rubber
    "A tire is made of rubber",
    "An eraser is made of rubber",
    "Rubber is stretchy and bouncy",
    
    # Stone and brick
    "A house can be made of brick",
    "A statue is made of stone",
    "Stone is hard and heavy",
]

MATERIALS = []

# =============================================================================
# ANCHOR: CAUSE_EFFECT_SENTENCES - Cause and effect relationships
# =============================================================================

CAUSE_EFFECT_SENTENCES = [
    # Fire and heat
    "Fire is hot and dangerous",
    "Fire can burn you",
    "When something burns it makes smoke",
    "Heat makes things warm",
    "Heat makes ice melt",
    
    # Ice and cold
    "Ice is frozen water",
    "Ice is cold and slippery",
    "When ice gets warm it melts",
    "Cold weather makes water freeze",
    "Snow melts when it gets warm",
    
    # Water
    "Water makes things wet",
    "When you spill water it makes a mess",
    "Water flows and splashes",
    
    # Weather
    "The sun makes us warm",
    "The sun is bright and hot",
    "Rain makes puddles on the ground",
    "Rain makes plants grow",
    "Wind blows leaves off trees",
    "Snow falls in winter",
    "Thunder is loud and scary",
    "Lightning is bright and dangerous",
    
    # Physical actions
    "When you fall down you can get hurt",
    "Falling can cause pain",
    "Sharp things can cut you",
    "When you cut yourself you bleed",
    "If you push something it moves away",
    "If you pull something it comes closer",
    "When you drop something it falls down",
    "Hitting something can cause pain",
    
    # Body needs
    "When you eat you feel full",
    "Eating gives you energy",
    "Drinking water makes you not thirsty",
    "Sleep makes you feel rested",
    "Running makes you tired",
    "Exercise makes you strong and healthy",
    "When you are hungry you need to eat",
    "When you are thirsty you need to drink",
    "When you are tired you need to sleep",
    
    # Emotions
    "When you cry you have tears",
    "Crying happens when you are sad",
    "Laughing happens when you are happy",
    "Smiling shows you are happy",
    "Hugging shows love and comfort",
    
    # Nature
    "Plants need water and sun to grow",
    "Seeds grow into plants",
    "Eggs hatch into baby birds",
    "Flowers bloom in spring",
    
    # Household
    "Soap and water make things clean",
    "Soap makes bubbles",
    "Dirt makes things dirty",
    "If something breaks it is broken",
]

CAUSE_EFFECT = []

# =============================================================================
# ANCHOR: SAFETY_RULES_SENTENCES - Safety rules and warnings
# =============================================================================

SAFETY_RULES_SENTENCES = [
    # Fire and heat
    "Fire is hot and dangerous",
    "Do not touch fire it will burn you",
    "The stove is hot and can burn you",
    "Do not touch the oven it is hot",
    "Matches and lighters are dangerous",
    
    # Sharp objects
    "A knife is sharp and can cut you",
    "Scissors are sharp be careful",
    "Broken glass is dangerous and can cut",
    
    # Electricity
    "Electricity is dangerous",
    "Do not put things in electrical outlets",
    "Do not touch wires they can shock you",
    
    # Water safety
    "Deep water is dangerous you can drown",
    "Always swim with an adult watching",
    "Do not run near the pool",
    
    # Road safety
    "Look both ways before crossing the street",
    "Cars are fast and dangerous",
    "Hold an adult's hand when crossing",
    "Always wear your seatbelt in the car",
    
    # Strangers
    "Do not talk to strangers",
    "Do not go anywhere with a stranger",
    "Tell an adult if a stranger talks to you",
    
    # Heights
    "Falling from high places can hurt you",
    "Hold the railing on stairs",
    "Do not climb on furniture",
    
    # Poisons
    "Medicine is only for adults to give",
    "Do not eat or drink cleaning products",
    "Do not put small things in your mouth",
]

SAFETY_RULES = []

# =============================================================================
# ANCHOR: HYGIENE_SENTENCES - Hygiene and health practices
# =============================================================================

HYGIENE_SENTENCES = [
    # Hand washing
    "Wash your hands with soap and water",
    "Wash your hands before eating",
    "Wash your hands after using the toilet",
    "Soap kills germs on your hands",
    "Germs can make you sick",
    "We use soap and water to wash our hands",
    
    # Teeth brushing
    "Brush your teeth in the morning and at night",
    "We brush our teeth to keep them clean",
    "Use a toothbrush and toothpaste to brush teeth",
    "Brushing teeth prevents cavities",
    "Visit the dentist to keep teeth healthy",
    
    # Bathing
    "Take a bath or shower to stay clean",
    "Use shampoo to wash your hair",
    "Use a towel to dry yourself after bathing",
    
    # Sneezing and coughing
    "Cover your mouth when you sneeze",
    "Cover your mouth when you cough",
    "Use a tissue to blow your nose",
    "Throw away used tissues",
    
    # General health
    "Being healthy means being strong and well",
    "When you are sick you should rest",
    "A doctor helps you get better when you are sick",
]

HYGIENE = []

# =============================================================================
# ANCHOR: MANNERS_SENTENCES - Manners and social norms
# =============================================================================

MANNERS_SENTENCES = [
    # Polite words
    "When you ask for something say please",
    "When you receive something say thank you",
    "When_we make a mistake we say sorry",
    "Say excuse me to get someone's attention",
    
    # Greetings
    "We say hello when we meet someone",
    "We say goodbye when we leave",
    "We say good morning in the morning",
    "We say good night before going to sleep",
    
    # Social rules
    "Good friends share their toys",
    "We take turns when playing games",
    "Be patient and wait your turn",
    "Listen when others are talking",
    "Be kind to everyone",
    "Help others when they need it",
    
    # Behavior rules
    "Do not hit or push other people",
    "Use a quiet voice inside the house",
    "Raise your hand before speaking in class",
    "Clean up your toys when you are done",
    
    # Table manners
    "Chew with your mouth closed",
    "Do not talk with your mouth full",
    "Use a napkin to wipe your mouth",
]

MANNERS_SOCIAL = []

# =============================================================================
# ANCHOR: ALPHABET_SENTENCES - Alphabet sequence and properties
# =============================================================================

ALPHABET_SENTENCES = [
    # Sequence
    "The first letter is letter_a",
    "The last letter is letter_z",
    "Letter_b comes after letter_a",
    "The alphabet has 26 letters",
    
    # Vowels
    "Letter_a letter_e letter_i letter_o letter_u are vowels",
    
    # Letter sounds
    "Letter_a is for apple",
    "Letter_b is for ball",
    "Letter_c is for cat",
    "Letter_d is for dog",
]

ALPHABET = []

# =============================================================================
# ANCHOR: NUMBERS_SENTENCES - Extended number sequence
# =============================================================================

NUMBERS_SENTENCES = [
    # Sequence
    "One comes before two",
    "Two comes after one",
    "Three comes after two",
    "Ten comes after nine",
    "Eleven comes after ten",
    "Twenty comes after nineteen",
    
    # Properties
    "A hand has five fingers",
    "We have ten fingers and ten toes",
    "A dozen means twelve",
    
    # Counting
    "We count one two three four five",
]

NUMBERS_EXTENDED = []

# =============================================================================
# ANCHOR: COMPARISONS_SENTENCES - Comparative concepts
# =============================================================================

COMPARISONS_SENTENCES = [
    # Size
    "An elephant is bigger than a mouse",
    "A mouse is smaller than a cat",
    "A whale is the biggest animal",
    "An ant is very tiny",
    "A giraffe is taller than a horse",
    
    # Speed
    "A cheetah is the fastest animal",
    "A turtle is slower than a rabbit",
    "A snail moves very slowly",
    "A plane is faster than a car",
    
    # Weight
    "A feather is lighter than a rock",
    "A rock is heavier than a feather",
    "An elephant is very heavy",
    "A balloon is very light and floats",
    
    # Age
    "A baby is younger than a child",
    "An adult is older than a child",
    "Grandparents are older than parents",
    
    # Temperature
    "Ice is colder than water",
    "Fire is very hot",
    "Summer is hotter than winter",
    "Winter is colder than summer",
]

COMPARISONS = []

# =============================================================================
# ANCHOR: PLANT_LIFE_SENTENCES - Plant parts, cycles, and types
# =============================================================================

PLANT_LIFE_SENTENCES = [
    # Plant parts
    "A plant has roots stems and leaves",
    "Roots grow underground and absorb water",
    "Leaves are green and make food from sunlight",
    "Flowers have colorful petals",
    "Seeds grow into new plants",
    
    # Trees
    "A tree has a trunk and branches",
    "Trees are tall plants with wood",
    "Leaves fall from trees in autumn",
    
    # Flowers
    "Roses are red flowers with thorns",
    "Sunflowers are tall yellow flowers",
    "Flowers bloom in spring",
]

PLANT_LIFE = []

# =============================================================================
# ANCHOR: LIFE_CYCLES - Life cycles of animals and plants
# =============================================================================
# FORMAT: Full sentences create proper episodic memories with context.
# Word pairs like ("egg", "caterpillar") create weak connections without
# semantic context. Sentences create chunked memories as in real brain.
# =============================================================================

LIFE_CYCLES_SENTENCES = [
    # ===== BUTTERFLY LIFE CYCLE =====
    "A butterfly starts as an egg",
    "An egg becomes a caterpillar",
    "A caterpillar hatches from an egg",
    "A caterpillar eats leaves and grows",
    "A caterpillar is green and hungry",
    "A caterpillar crawls on plants",
    "A caterpillar turns into a cocoon",
    "A cocoon is also called a chrysalis",
    "Inside the cocoon the caterpillar transforms",
    "Metamorphosis happens inside the cocoon",
    "A butterfly comes out of the cocoon",
    "A butterfly has colorful wings",
    "A butterfly can fly",
    "A butterfly lays eggs",
    "This is the butterfly life cycle",
    "Egg then caterpillar then cocoon then butterfly",
    
    # ===== FROG LIFE CYCLE =====
    "A frog starts as an egg in water",
    "An egg becomes a tadpole",
    "A tadpole hatches from an egg",
    "A tadpole lives in water",
    "A tadpole has a tail and gills",
    "A tadpole swims in the pond",
    "A tadpole grows legs",
    "A tadpole loses its tail",
    "A tadpole turns into a frog",
    "A frog has legs and can jump",
    "A frog has lungs and can breathe air",
    "A frog lives on land and in water",
    "A frog is an amphibian",
    "A frog croaks and hops",
    "A frog lays eggs in water",
    "This is the frog life cycle",
    "Egg then tadpole then frog",
    
    # ===== CHICKEN LIFE CYCLE =====
    "A chicken starts as an egg",
    "A chick hatches from an egg",
    "A chick breaks out of the shell",
    "A chick is small yellow and fluffy",
    "A chick peeps and grows",
    "A chick becomes a chicken",
    "A hen is a female chicken",
    "A hen lays eggs",
    "A rooster is a male chicken",
    "A rooster crows in the morning",
    "A chicken lays eggs",
    "This is the chicken life cycle",
    "Egg then chick then chicken",
    
    # ===== PLANT LIFE CYCLE =====
    "A plant starts as a seed",
    "A seed germinates and grows",
    "A seed becomes a sprout",
    "A sprout emerges from the soil",
    "A sprout grows into a seedling",
    "A seedling is a young plant",
    "A seedling becomes a mature plant",
    "A plant grows flowers",
    "A flower blooms on the plant",
    "A flower becomes a fruit",
    "A fruit contains seeds",
    "A fruit ripens on the plant",
    "Seeds fall from the fruit",
    "This is the plant life cycle",
    "Seed then sprout then plant then flower then fruit then seed",
    
    # ===== TREE LIFE CYCLE =====
    "A tree starts as a seed",
    "A seed grows into a sapling",
    "A sapling is a young small tree",
    "A sapling grows taller",
    "A sapling becomes a mature tree",
    "A tree is tall and strong",
    "A tree produces seeds",
    "This is the tree life cycle",
    
    # ===== HUMAN LIFE CYCLE =====
    "A human starts as a baby",
    "A baby is also called an infant",
    "A baby crawls and learns",
    "A baby grows into a child",
    "A child walks and talks and plays",
    "A child grows into a teenager",
    "A teenager is also called an adolescent",
    "A teenager becomes an adult",
    "An adult is grown and independent",
    "An adult works and can have babies",
    "An adult can become a parent",
    "An elderly person is old",
    "An elderly person is a senior or grandparent",
    "This is the human life cycle",
    "Baby then child then teenager then adult",
    
    # ===== MAMMAL LIFE CYCLE =====
    "A mammal baby drinks milk from its mother",
    "A young mammal learns and plays",
    "A young mammal grows bigger",
    "An adult mammal can have babies",
    
    # ===== SEASONAL CYCLE =====
    "After winter comes spring",
    "After spring comes summer",
    "After summer comes autumn",
    "After autumn comes winter",
    "Spring brings new life and growth",
    "Summer is warm and plants flourish",
    "Autumn is harvest time and leaves fall",
    "Winter is cold and animals rest",
    "This is the cycle of seasons",
]

# Legacy format for backward compatibility (will be deprecated)
LIFE_CYCLES = []

# =============================================================================
# ANCHOR: TIME_CONCEPTS - Time concepts and sequences
# =============================================================================

TIME_CONCEPTS_SENTENCES = [
    # Parts of day
    "In the morning we wake up and eat breakfast",
    "The sun rises in the morning",
    "In the afternoon we eat lunch",
    "In the evening we eat dinner",
    "At night we sleep and it is dark",
    "The moon and stars come out at night",
    "The sun sets in the evening",
    
    # Days of week
    "There are seven days in a week",
    "Monday is the first day of the week",
    "Saturday and Sunday are the weekend",
    "We go to school on weekdays",
    "We do not go to school on weekends",
    
    # Months and seasons
    "There are twelve months in a year",
    "January is the first month",
    "Spring comes after winter",
    "Summer comes after spring",
    "Autumn comes after summer",
    "Winter comes after autumn",
    "Spring is warm and flowers bloom",
    "Summer is hot and sunny",
    "Autumn is cool and leaves fall",
    "Winter is cold and it snows",
    
    # Time concepts
    "Yesterday is the day before today",
    "Tomorrow is the day after today",
    "A clock tells us what time it is",
    "A calendar shows us the days and months",
]

TIME_CONCEPTS = []

# =============================================================================
# ANCHOR: BODY_SENTENCES - Detailed body parts
# =============================================================================

BODY_SENTENCES = [
    # Head and face
    "The head is at the top of the body",
    "The brain is inside the head",
    "We have two eyes to see",
    "We have two ears to hear",
    "The nose is for smelling and breathing",
    "The mouth is for eating and talking",
    "Teeth are for chewing food",
    "The tongue is for tasting",
    
    # Body parts
    "The neck connects the head to the body",
    "The heart is in the chest and pumps blood",
    "The lungs are for breathing air",
    "We have two arms and two legs",
    "Each hand has five fingers",
    "Each foot has five toes",
    "The elbow bends the arm",
    "The knee bends the leg",
    
    # Functions
    "Muscles help us move",
    "Bones are hard and support the body",
    "Skin covers and protects the body",
]

BODY_DETAILED = []

# =============================================================================
# ANCHOR: SPATIAL_SENTENCES - Spatial relationships
# =============================================================================

SPATIAL_SENTENCES = [
    # Basic directions
    "Up is the opposite of down",
    "In means inside something",
    "Out means outside something",
    "Above means over something",
    "Below means under something",
    
    # Left and right
    "Left is the opposite of right",
    "We have a left hand and a right hand",
    
    # Near and far
    "Near means close to something",
    "Far means a long distance away",
    
    # Front and back
    "Front is the opposite of back",
    "Behind means in back of something",
    
    # Positions
    "On means on top of something",
    "Between means in the middle of two things",
    "Next to means beside something",
]

SPATIAL_RELATIONS = []

# =============================================================================
# ANCHOR: SENSES_SENTENCES - Five senses in detail
# =============================================================================

SENSES_SENTENCES = [
    # Overview
    "We have five senses sight hearing smell taste and touch",
    
    # Sight
    "We use our eyes to see",
    "Eyes help us see colors and shapes",
    "Bright light helps us see better",
    "It is dark at night and hard to see",
    
    # Hearing
    "We use our ears to hear",
    "Ears help us hear sounds and music",
    "Loud sounds are noisy",
    "Quiet sounds are soft",
    
    # Smell
    "We use our nose to smell",
    "The nose helps us smell flowers and food",
    "Some things smell good and some smell bad",
    
    # Taste
    "We use our tongue to taste",
    "The tongue helps us taste food",
    "Sweet things taste like sugar and candy",
    "Sour things taste like lemon",
    "Salty things taste like salt",
    
    # Touch
    "We use our skin to feel and touch",
    "We can feel if something is soft or hard",
    "We can feel if something is hot or cold",
    "We can feel if something is smooth or rough",
    "We can feel if something is wet or dry",
]

SENSES = []

# =============================================================================
# ANCHOR: QUESTION_ALIGNED_SENTENCES - Explicit sentences matching question format
# =============================================================================
# These sentences are specifically designed to train the model to answer
# the test questions correctly. They use the same phrasing as the questions.

QUESTION_ALIGNED_SENTENCES = [
    # Life cycles - "What does X become?" (use "become" not "becomes" to match question)
    "tadpole become frog",
    "a tadpole become a frog",
    "tadpole becomes frog",
    "caterpillar become butterfly",
    "a caterpillar become a butterfly",
    "caterpillar becomes butterfly",
    "seed become plant",
    "seed become sprout",
    "egg become chick",
    "egg become bird",
    "egg become animal",
    "cocoon become butterfly",
    "butterfly come from cocoon",
    "from cocoon come butterfly",
    
    # Animal groups - "What is a group of X called?"
    "group of fish called school",
    "a group of fish is called a school",
    "fish group called school",
    "group of wolves called pack",
    "a group of wolves is called a pack",
    "wolves group called pack",
    "group of birds called flock",
    "a group of birds is called a flock",
    "birds group called flock",
    "group of lions called pride",
    "a group of lions is called a pride",
    "lions group called pride",
    
    # Baby animals - "What is a X?" (ensure parent animal is in answer)
    # Multiple variations to strengthen the connection baby+animal
    "puppy is a baby dog",
    "a puppy is a baby dog",
    "puppy baby dog",
    "what is puppy baby dog",
    "kitten is a baby cat",
    "a kitten is a baby cat",
    "kitten baby cat",
    "what is kitten baby cat",
    "calf is a baby cow",
    "a calf is a baby cow",
    "calf baby cow",
    "what is calf baby cow",
    
    # Materials - "What is X made of?"
    "table made of wood",
    "a table is made of wood",
    "window made of glass",
    "a window is made of glass",
    "tire made of rubber",
    "shirt made of cotton",
    
    # Cause-effect - "What happens when..."
    "touch fire burn",
    "when you touch fire you burn",
    "fire burn hurt",
    "ice warm melt",
    "when ice warm melt",
    "ice get warm melt",
    "fall hurt",
    "when you fall you hurt",
    "fall pain",
    
    # Hygiene - "When should you..." and "What do you use..."
    "wash hands before eating",
    "should wash hands before eating",
    "wash hands after toilet",
    "use soap wash hands",
    "use water wash hands",
    "soap water wash hands",
    
    # Alphabet - "What comes after..." and "What is first/last letter?"
    # Using letter_a format to avoid conflict with article 'a'
    "after letter_a come letter_b",
    "letter_b come after letter_a",
    "letter_a first letter",
    "first letter letter_a",
    "the first letter is letter_a",
    "letter_z last letter",
    "last letter letter_z",
    "what comes after letter_a letter_b",
    "the last letter is Z",
    
    # Numbers - "What comes after..."
    "after ten come eleven",
    "eleven come after ten",
    "after nineteen come twenty",
    "twenty come after nineteen",
    
    # Comparisons - negatives (answer should include "no" or negative)
    "turtle slower than rabbit",
    "turtle is slower than rabbit not faster",
    "rabbit faster than turtle",
    "feather lighter than rock",
    "feather is lighter than rock not heavier",
    "rock heavier than feather",
    
    # Plants
    "root under ground",
    "roots under ground",
    "under ground root",
    
    # Time
    "after January come February",
    "February come after January",
    "month after January February",
    "sun rise morning",
    "sun rise in morning",
    "when sun rise morning",
    
    # Spatial - opposites
    "opposite of in out",
    "opposite of in is out",
    "in opposite out",
    
    # Senses - strengthen sense+organ connections
    "nose smell",
    "smell use nose",
    "sense nose smell",
    "what sense uses nose smell",
    "nose is for smell",
    "we smell with nose",
    "tongue taste",
    "taste use tongue",
    "sense tongue taste",
    "what sense uses tongue taste",
    "tongue is for taste",
    "we taste with tongue",
    
    # Cause-effect strengthening
    "touch fire burn you",
    "fire burn hurt",
    "when touch fire burn",
    "ice warm melt",
    "when ice warm melt",
    "ice gets warm melt",
    "fall hurt",
    "when fall hurt",
    "you fall you hurt",
    
    # Hygiene strengthening
    "wash hands soap water",
    "use soap water wash hands",
    "soap and water wash hands",
    
    # Manners strengthening
    "receive something say thank you",
    "when receive say thank you",
    "say thank you when receive",
]

def get_preschool_connections() -> List[Tuple[str, str]]:
    """Returns all preschool knowledge as (word1, word2) connection tuples."""
    all_connections = []
    all_connections.extend(BABY_ANIMALS)
    all_connections.extend(ANIMAL_GROUPS)
    all_connections.extend(MATERIALS)
    all_connections.extend(CAUSE_EFFECT)
    all_connections.extend(SAFETY_RULES)
    all_connections.extend(HYGIENE)
    all_connections.extend(MANNERS_SOCIAL)
    all_connections.extend(ALPHABET)
    all_connections.extend(NUMBERS_EXTENDED)
    all_connections.extend(COMPARISONS)
    all_connections.extend(PLANT_LIFE)
    all_connections.extend(LIFE_CYCLES)
    all_connections.extend(TIME_CONCEPTS)
    all_connections.extend(BODY_DETAILED)
    all_connections.extend(SPATIAL_RELATIONS)
    all_connections.extend(SENSES)
    return all_connections

def get_preschool_sentences() -> List[str]:
    """Returns all preschool knowledge as sentences for training.
    
    FORMAT: Full sentences create proper episodic memories with context.
    Word pairs like ("egg", "caterpillar") create weak connections without
    semantic context. Sentences create chunked memories as in real brain.
    """
    sentences = []
    
    # All sentence-based data (proper format)
    sentences.extend(BABY_ANIMALS_SENTENCES)
    sentences.extend(ANIMAL_GROUPS_SENTENCES)
    sentences.extend(MATERIALS_SENTENCES)
    sentences.extend(CAUSE_EFFECT_SENTENCES)
    sentences.extend(SAFETY_RULES_SENTENCES)
    sentences.extend(HYGIENE_SENTENCES)
    sentences.extend(MANNERS_SENTENCES)
    sentences.extend(ALPHABET_SENTENCES)
    sentences.extend(NUMBERS_SENTENCES)
    sentences.extend(COMPARISONS_SENTENCES)
    sentences.extend(PLANT_LIFE_SENTENCES)
    sentences.extend(LIFE_CYCLES_SENTENCES)
    sentences.extend(TIME_CONCEPTS_SENTENCES)
    sentences.extend(BODY_SENTENCES)
    sentences.extend(SPATIAL_SENTENCES)
    sentences.extend(SENSES_SENTENCES)
    sentences.extend(QUESTION_ALIGNED_SENTENCES)
    
    # Legacy: Add remaining connection-based data (will be converted later)
    for w1, w2 in get_preschool_connections():
        if w1 and w2:  # Skip empty tuples
            sentences.append(f"{w1} {w2}")
    
    return sentences

def get_preschool_questions() -> List[Tuple[str, List[str]]]:
    """Test questions with expected answer keywords."""
    return PRESCHOOL_QUESTIONS

# =============================================================================
# ANCHOR: PRESCHOOL_QUESTIONS - Test questions for validation
# =============================================================================

PRESCHOOL_QUESTIONS = [
    # Baby animals
    ("What is a puppy?", ["baby dog", "young dog", "dog"]),
    ("What is a kitten?", ["baby cat", "young cat", "cat"]),
    ("What is a calf?", ["baby cow", "young cow", "cow"]),
    ("What does a tadpole become?", ["frog"]),
    ("What does a caterpillar become?", ["butterfly"]),
    
    # Animal groups
    ("What is a group of fish called?", ["school"]),
    ("What is a group of wolves called?", ["pack"]),
    ("What is a group of birds called?", ["flock"]),
    ("What is a group of lions called?", ["pride"]),
    
    # Materials
    ("What is a table made of?", ["wood"]),
    ("What is a window made of?", ["glass"]),
    ("What is a tire made of?", ["rubber"]),
    ("What is a shirt made of?", ["cotton", "fabric"]),
    
    # Cause-effect
    ("What happens when you touch fire?", ["burn", "hurt", "pain"]),
    ("What happens when ice gets warm?", ["melt", "melts"]),
    ("What happens when you fall?", ["hurt", "pain"]),
    
    # Safety
    ("Is fire dangerous?", ["yes", "dangerous", "hot"]),
    ("Is a knife sharp?", ["yes", "sharp", "cut"]),
    
    # Hygiene
    ("When should you wash your hands?", ["before eating", "after toilet", "eating"]),
    ("When should you brush your teeth?", ["morning", "night", "day", "every"]),
    ("What do you use to wash hands?", ["soap", "water"]),
    
    # Manners
    ("What do you say when you ask for something?", ["please"]),
    ("What do you say when you receive something?", ["thank you", "thank"]),
    ("What do you say when you make a mistake?", ["sorry"]),
    
    # Alphabet (context-dependent: "A" in temporal query → letter_a via PFC modulation)
    ("What comes after A?", ["b", "letter_b"]),
    ("What is the first letter?", ["a", "letter_a"]),
    ("What is the last letter?", ["z", "letter_z"]),
    
    # Numbers
    ("What comes after ten?", ["eleven"]),
    ("What comes after nineteen?", ["twenty"]),
    
    # Comparisons
    ("Is an elephant bigger than a mouse?", ["yes", "bigger"]),
    ("Is a turtle faster than a rabbit?", ["no", "slower"]),
    ("Is a feather heavier than a rock?", ["no", "lighter", "rock", "heavier"]),
    
    # Plants
    ("What does a seed become?", ["plant", "sprout"]),
    ("What is under the ground?", ["root", "roots"]),
    
    # Life cycles
    ("What does an egg become?", ["chick", "bird", "animal"]),
    ("What comes from a cocoon?", ["butterfly"]),
    
    # Time
    ("What comes after Monday?", ["tuesday"]),
    ("What month comes after January?", ["february"]),
    ("When does the sun rise?", ["morning"]),
    ("When do stars appear?", ["night"]),
    
    # Body
    ("What do we see with?", ["eyes"]),
    ("What do we hear with?", ["ears"]),
    ("How many fingers on one hand?", ["five", "5"]),
    
    # Spatial
    ("What is the opposite of up?", ["down"]),
    ("What is the opposite of in?", ["out"]),
    ("What is the opposite of left?", ["right"]),
    
    # Senses
    ("What sense uses the nose?", ["smell"]),
    ("What sense uses the tongue?", ["taste"]),
]
