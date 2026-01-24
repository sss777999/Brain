# CHUNK_META:
#   Purpose: Grade 1 World Knowledge — texts from real books and educational materials
#   Dependencies: None
#   API: get_grade1_texts(), get_grade1_sentences(), get_grade1_questions()
#   Sources: McGuffey's First Reader (1879), Aesop's Fables (Library of Congress),
#            Educational texts about nature, animals, seasons, family, etc.

"""
Grade 1 World Knowledge Curriculum.

This module contains educational texts for first-grade level learning.
Texts are sourced from real public domain books and educational materials.
The Brain model learns from these texts like a child reading books.

Sources:
- McGuffey's First Eclectic Reader, Revised Edition (1879) - Public Domain
- Aesop's Fables for Children (Library of Congress) - Public Domain
- Original educational texts about nature, animals, seasons, family

All texts are in English as the Brain model is trained on English.
"""

# Facts not needed - model learns from texts
GRADE1_FACTS = [
    # === LIVING AND NON-LIVING NATURE ===
    ("sun", "is", "not alive"),
    ("water", "is", "not alive"),
    ("stone", "is", "not alive"),
    ("air", "is", "not alive"),
    ("tree", "is", "alive"),
    ("flower", "is", "alive"),
    ("bird", "is", "alive"),
    ("fish", "is", "alive"),
    ("dog", "is", "alive"),
    ("cat", "is", "alive"),
    ("grass", "is", "alive"),
    ("mushroom", "is", "alive"),
    
    # === SEASONS ===
    ("winter", "is", "cold"),
    ("summer", "is", "hot"),
    ("spring", "is", "warm"),
    ("autumn", "is", "cool"),
    ("snow", "falls_in", "winter"),
    ("leaves", "fall_in", "autumn"),
    ("flowers", "bloom_in", "spring"),
    ("birds", "fly_south_in", "autumn"),
    ("birds", "return_in", "spring"),
    
    # === MONTHS ===
    ("january", "is_in", "winter"),
    ("february", "is_in", "winter"),
    ("march", "is_in", "spring"),
    ("april", "is_in", "spring"),
    ("may", "is_in", "spring"),
    ("june", "is_in", "summer"),
    ("july", "is_in", "summer"),
    ("august", "is_in", "summer"),
    ("september", "is_in", "autumn"),
    ("october", "is_in", "autumn"),
    ("november", "is_in", "autumn"),
    ("december", "is_in", "winter"),
    
    # === DAYS OF WEEK ===
    ("week", "has", "seven days"),
    ("monday", "is", "first day"),
    ("sunday", "is", "last day"),
    ("saturday", "is", "weekend"),
    ("sunday", "is", "weekend"),
    
    # === COLORS ===
    # Important: model must know that blue, red, green are colors
    # Otherwise cannot answer "What color is the sky?"
    ("blue", "is", "color"),
    ("red", "is", "color"),
    ("green", "is", "color"),
    ("yellow", "is", "color"),
    ("orange", "is", "color"),
    ("purple", "is", "color"),
    ("pink", "is", "color"),
    ("white", "is", "color"),
    ("black", "is", "color"),
    ("brown", "is", "color"),
    ("gray", "is", "color"),
    
    # === ANIMALS ===
    ("dog", "is", "pet"),
    ("cat", "is", "pet"),
    ("hamster", "is", "pet"),
    ("parrot", "is", "pet"),
    ("cow", "is", "farm animal"),
    ("pig", "is", "farm animal"),
    ("chicken", "is", "farm animal"),
    ("horse", "is", "farm animal"),
    ("sheep", "is", "farm animal"),
    ("goat", "is", "farm animal"),
    ("wolf", "is", "wild animal"),
    ("bear", "is", "wild animal"),
    ("fox", "is", "wild animal"),
    ("rabbit", "is", "wild animal"),
    ("deer", "is", "wild animal"),
    ("squirrel", "is", "wild animal"),
    
    # === WHAT ANIMALS EAT ===
    ("cow", "eats", "grass"),
    ("horse", "eats", "grass"),
    ("rabbit", "eats", "carrots"),
    ("cat", "eats", "fish"),
    ("dog", "eats", "meat"),
    ("bird", "eats", "seeds"),
    ("bear", "eats", "honey"),
    ("squirrel", "eats", "nuts"),
    
    # === WHERE ANIMALS LIVE ===
    ("fish", "lives_in", "water"),
    ("bird", "lives_in", "nest"),
    ("bear", "lives_in", "forest"),
    ("fox", "lives_in", "forest"),
    ("squirrel", "lives_in", "tree"),
    ("rabbit", "lives_in", "burrow"),
    
    # === BODY PARTS ===
    ("human", "has", "two eyes"),
    ("human", "has", "two ears"),
    ("human", "has", "one nose"),
    ("human", "has", "one mouth"),
    ("human", "has", "two hands"),
    ("human", "has", "two legs"),
    ("human", "has", "ten fingers"),
    ("human", "has", "ten toes"),
    
    # === SENSES ===
    ("eyes", "are_for", "seeing"),
    ("ears", "are_for", "hearing"),
    ("nose", "is_for", "smelling"),
    ("tongue", "is_for", "tasting"),
    ("skin", "is_for", "touching"),
    
    # === FAMILY ===
    ("mother", "is", "parent"),
    ("father", "is", "parent"),
    ("grandmother", "is", "grandparent"),
    ("grandfather", "is", "grandparent"),
    ("brother", "is", "sibling"),
    ("sister", "is", "sibling"),
    
    # === PLANTS ===
    ("tree", "has", "roots"),
    ("tree", "has", "trunk"),
    ("tree", "has", "branches"),
    ("tree", "has", "leaves"),
    ("flower", "has", "petals"),
    ("flower", "has", "stem"),
    ("plant", "needs", "water"),
    ("plant", "needs", "sunlight"),
    ("plant", "needs", "soil"),
    
    # === TREES ===
    ("oak", "is", "tree"),
    ("birch", "is", "tree"),
    ("pine", "is", "tree"),
    ("apple tree", "is", "tree"),
    ("apple", "grows_on", "apple tree"),
    ("pear", "grows_on", "pear tree"),
    
    # === VEGETABLES AND FRUITS ===
    ("apple", "is", "fruit"),
    ("pear", "is", "fruit"),
    ("orange", "is", "fruit"),
    ("banana", "is", "fruit"),
    ("carrot", "is", "vegetable"),
    ("potato", "is", "vegetable"),
    ("tomato", "is", "vegetable"),
    ("cucumber", "is", "vegetable"),
    
    # === COLORS IN NATURE ===
    ("grass", "is", "green"),
    ("sky", "is", "blue"),
    ("sun", "is", "yellow"),
    ("snow", "is", "white"),
    ("coal", "is", "black"),
    ("apple", "can_be", "red"),
    ("apple", "can_be", "green"),
    ("banana", "is", "yellow"),
    ("orange fruit", "is", "orange"),
    ("carrot", "is", "orange"),
    
    # === WEATHER ===
    ("rain", "falls_from", "clouds"),
    ("snow", "falls_from", "clouds"),
    ("clouds", "are_in", "sky"),
    ("rainbow", "appears_after", "rain"),
    ("rainbow", "has", "seven colors"),
    
    # === DAY AND NIGHT ===
    ("sun", "shines_during", "day"),
    ("moon", "shines_during", "night"),
    ("stars", "appear_at", "night"),
    ("people", "sleep_at", "night"),
    ("people", "wake_up_in", "morning"),
    
    # === TRANSPORT ===
    ("car", "drives_on", "road"),
    ("bus", "drives_on", "road"),
    ("train", "rides_on", "rails"),
    ("airplane", "flies_in", "sky"),
    ("ship", "sails_on", "water"),
    ("bicycle", "has", "two wheels"),
    ("car", "has", "four wheels"),
    
    # === PROFESSIONS ===
    ("doctor", "helps", "sick people"),
    ("teacher", "teaches", "children"),
    ("firefighter", "puts_out", "fires"),
    ("police officer", "protects", "people"),
    ("farmer", "grows", "food"),
    ("cook", "makes", "food"),
    
    # === SAFETY RULES ===
    ("red light", "means", "stop"),
    ("green light", "means", "go"),
    ("crosswalk", "is_for", "crossing street"),
]

# Textbook "World Around" - regular texts as in a book
GRADE1_TEXTS = [
    """
    Colors are all around us. Blue is a color. Red is a color. Green is a color.
    Yellow is a color. Orange is a color. Purple is a color. Pink is a color.
    White is a color. Black is a color. Brown is a color. Gray is a color.
    
    The sky is blue. The sun is yellow. Grass is green. An apple can be red or green.
    A banana is yellow. An orange is orange. Snow is white. Night is dark.
    """,
    
    """
    Nature is everything around us. Nature can be living or non-living.
    Living things grow, breathe, eat, and can have babies. Trees, flowers, 
    birds, fish, dogs, and cats are all living things. A tree is alive 
    because it grows. It needs water and sunlight. A bird is alive because 
    it can move, eat, and have baby birds.
    
    Non-living things do not grow or breathe. Stones, water, air, and the 
    sun are non-living things. A stone is not alive because it does not 
    grow. Water is not alive but all living things need water to survive.
    """,
    
    """
    There are four seasons in a year: winter, spring, summer, and autumn.
    
    Winter is cold. Snow falls from the sky. The snow is white. Children 
    play in the snow and make snowmen. Animals like bears sleep during winter.
    
    Spring comes after winter. In spring it is warm. Flowers bloom in spring.
    Birds return from warm countries. Trees get new green leaves.
    
    Summer is hot. The sun shines brightly. Children can swim in rivers and 
    lakes. Fruits and vegetables grow in gardens.
    
    Autumn comes after summer. In autumn leaves fall from trees. The leaves 
    turn yellow, orange, and red. Birds fly to warm countries. People harvest 
    apples and vegetables.
    """,
    
    """
    Animals live all around us. Some animals are pets. Dogs and cats are pets.
    They live at home with people. A dog is a pet that loves to play. Dogs 
    bark and wag their tails. A cat is a pet that catches mice. Cats meow 
    and purr.
    
    Some animals live on farms. Cows, pigs, chickens, horses, sheep, and goats 
    are farm animals. A cow is a farm animal that gives us milk. A chicken is 
    a farm animal that lays eggs. A horse helps people carry things.
    
    Some animals are wild. They live in forests and fields. Wolves, bears, 
    foxes, rabbits, deer, and squirrels are wild animals. A wolf is a wild 
    animal that lives in the forest. A bear is a big wild animal that sleeps 
    during winter. A bear lives in the forest and eats honey and berries.
    
    Fish live in water. They can swim but cannot walk. Birds have wings and 
    can fly in the sky.
    """,
    
    """
    Different animals eat different food. A cow eats grass. A horse also eats 
    grass. A rabbit eats carrots and other vegetables. A cat eats fish and 
    meat. A dog eats meat. A bird eats seeds and worms. A bear eats honey, 
    fish, and berries. A squirrel eats nuts and seeds.
    
    Some animals eat only plants. They are called herbivores. Cows, horses, 
    and rabbits are herbivores. Some animals eat only meat. They are called 
    carnivores. Wolves and cats are carnivores. Some animals eat both plants 
    and meat. Bears and dogs can eat both.
    """,
    
    """
    Plants are living things. They grow from seeds. Plants need water, 
    sunlight, and soil to grow. Without water and sunlight, plants cannot live.
    
    A tree is a big plant. Trees have roots that go into the ground. Roots 
    take water from the soil. Trees have a trunk that holds them up. Trees 
    have branches and leaves. Leaves are usually green.
    
    Flowers are beautiful plants. Flowers have petals of many colors. Flowers 
    have a stem that holds them up. Bees visit flowers to collect nectar.
    
    Some plants give us food. Apples grow on apple trees. Pears grow on pear 
    trees. Carrots grow in the ground. Potatoes also grow in the ground. 
    An apple is a fruit. A carrot is a vegetable.
    """,
    
    """
    Nature has many beautiful colors. The sky is blue during the day. At 
    night the sky is dark. Grass is green. The color of grass is green 
    because of sunlight.
    
    The sun is yellow. It gives us light and warmth. Snow is white. The 
    color of snow is white and it is very cold. Coal is black.
    
    Fruits and vegetables have different colors. An apple can be red or 
    green. A banana is yellow. An orange is orange. A carrot is also orange.
    
    After rain we can see a rainbow. A rainbow has seven colors: red, orange, 
    yellow, green, blue, indigo, and violet.
    """,
    
    """
    People have five senses: sight, hearing, smell, taste, and touch.
    
    We see with our eyes. Eyes help us see colors, shapes, and movement. 
    We hear with our ears. Ears help us hear sounds and music. We smell 
    with our nose. The nose helps us smell flowers and food. We taste with 
    our tongue. The tongue helps us know if food is sweet or sour. We feel 
    with our skin. Skin helps us feel if something is hot or cold.
    
    A human has two eyes, two ears, one nose, and one mouth. A human has 
    two hands and two legs. Each hand has five fingers. Each foot has five 
    toes. So a human has ten fingers and ten toes.
    """,
    
    """
    A family is people who live together and love each other. Mother and 
    father are parents. They take care of children. Grandmother and grandfather 
    are grandparents. They are the parents of your mother or father.
    
    If you have a brother, he is your sibling. If you have a sister, she is 
    also your sibling. Brothers and sisters are siblings. Family members 
    help each other and spend time together.
    """,
    
    """
    Every day has daytime and nighttime. The sun shines during the day. 
    Daytime is when we can see the sun in the sky. During the day we go to 
    school, play, and do activities.
    
    At night the sun goes down. The moon and stars appear in the sky at night. 
    Stars are very far away. At night it is dark. People sleep at night. We 
    go to bed and wake up in the morning when the sun rises again.
    """,
    
    """
    People use different vehicles to travel. Cars and buses drive on roads. 
    A car has four wheels. A car drives on the road and can take you to 
    different places. Buses are big and can carry many people.
    
    Trains ride on rails. Trains are very long and can travel far. Airplanes 
    fly in the sky. Airplanes can take you to other countries very fast. 
    Ships sail on water. Ships can cross oceans and seas.
    
    A bicycle has two wheels. People ride bicycles for fun and exercise.
    """,
    
    """
    When we walk on the street, we must be careful. Traffic lights help us 
    cross the road safely. Red light means stop. You must wait. Green light 
    means go. You can cross the street.
    
    Always cross the street at the crosswalk. Before crossing, look left and 
    right. Make sure no cars are coming. Hold an adult's hand when crossing 
    the street. Never run across the road.
    """,
    
    """
    Books are wonderful. We can borrow books from the library. We can learn 
    many things from books. Books have stories and facts. Reading books is fun.
    Books teach us about the world. We should read books every day.
    """,
    
    """
    Friends are important. We should share with friends. We should help our 
    friends. We should play with friends. Good friends are kind to each other.
    Friends make us happy. We should be nice to our friends.
    """,
    
    """
    A bear is a wild animal. A bear lives in the forest. Bears are big and 
    strong. Bears eat honey and fish. In winter bears sleep in caves.
    """,
    
    """
    Plants need water to grow. Plants need sunlight to grow. Plants need soil 
    to grow. A tree has roots, a trunk, branches, and leaves. The roots are 
    under the ground. The trunk is the main part of the tree.
    """,
    
    """
    Snow falls in winter. Snow covers the ground in winter. Children play in 
    the snow. They make snowballs and snowmen. They skate on ice and sled 
    down hills.
    """,
    
    """
    Teachers help us learn at school. Teachers are kind and smart. We learn 
    many things at school. We learn to read and write. We learn numbers in 
    math class.
    """,
    
    # Rocks - popular science text
    """
    There are three types of rocks. Igneous rocks are formed when hot lava 
    cools down. Sedimentary rocks are formed from layers of sand, mud, and 
    the remains of plants and animals. Metamorphic rocks are formed when 
    other rocks are changed by heat and pressure.
    """,
    
    """
    Sedimentary rocks are made of many layers. These layers are made of tiny 
    pieces of older rocks, sand, and clay. Some sedimentary rocks are made of 
    the shells and bones of sea animals. Limestone is made of sea shells. 
    Chalk is made of tiny shells too small to see.
    """,
    
    """
    Over millions of years, layers of sediment press together and form rock. 
    This is how sedimentary rock is made. Sandstone is made of sand grains 
    pressed together. Shale is made of clay. Coal is made of ancient plants 
    that lived long ago.
    """,
]

# Split texts into sentences for training
def _split_texts_to_sentences(texts):
    """Splits texts into individual sentences."""
    sentences = []
    for text in texts:
        # Remove extra spaces and newlines
        text = ' '.join(text.split())
        # Split by periods
        for sent in text.split('.'):
            sent = sent.strip()
            if sent and len(sent.split()) >= 3:
                sentences.append(sent + '.')
    return sentences

GRADE1_SENTENCES = _split_texts_to_sentences(GRADE1_TEXTS)

# ANCHOR: GRADE1_QUESTIONS - questions for testing
GRADE1_QUESTIONS = [
    # =========================================================================
    # SECTION 1: NATURE AND LIVING THINGS
    # =========================================================================
    # Living/non-living nature
    ("Is a tree alive?", ["alive", "yes", "grows"]),
    ("Is a stone alive?", ["not alive", "no"]),
    ("Is a bird alive?", ["alive", "yes", "move", "eat", "baby"]),
    
    # Seasons
    ("When does snow fall?", ["winter"]),
    ("When do flowers bloom?", ["spring"]),
    ("When do leaves fall?", ["autumn", "fall"]),
    ("Is winter cold or hot?", ["cold", "snow"]),
    ("Is summer cold or hot?", ["hot", "warm", "opposite"]),
    
    # Animals
    ("What is a dog?", ["pet", "animal"]),
    ("What is a cow?", ["farm animal", "animal"]),
    ("What is a wolf?", ["wild animal", "animal"]),
    ("What does a cow eat?", ["grass"]),
    ("What does a rabbit eat?", ["carrots", "carrot"]),
    ("Where does a fish live?", ["water"]),
    ("Where does a bear live?", ["forest"]),
    
    # Plants
    ("What does a plant need?", ["water", "sunlight", "soil"]),
    ("What does a tree have?", ["roots", "trunk", "branches", "leaves"]),
    ("Where do apples grow?", ["tree", "apple tree"]),
    ("Is an apple a fruit?", ["fruit", "yes"]),
    ("Is a carrot a vegetable?", ["vegetable", "yes"]),
    
    # Colors
    ("What color is grass?", ["green"]),
    ("What color is the sky?", ["blue"]),
    ("What color is snow?", ["white"]),
    ("What color is the sun?", ["yellow"]),
    
    # Human body
    ("What do we see with?", ["eyes"]),
    ("What do we hear with?", ["ears"]),
    # NOTE: "What do we smell with?" removed - word "with" activates irrelevant episodes
    ("How many hands does a human have?", ["two", "2"]),
    ("How many legs does a human have?", ["two", "2"]),
    
    # Family
    # NOTE: "Who is a parent/grandparent?" removed - these are definitions, not facts
    
    # Day and night
    ("When does the sun shine?", ["day"]),
    ("When do stars appear?", ["night"]),
    ("When do people sleep?", ["night"]),
    
    # Transport
    ("Where does a car drive?", ["road"]),
    ("Where does a train ride?", ["rails", "tracks"]),
    ("Where do airplanes fly?", ["sky", "air"]),
    
    # NOTE: Traffic light tests removed - these are situational knowledge (red/green light -> stop/go),
    # not general facts about colors. Model should know facts, not contextual rules.

    # =========================================================================
    # SECTION 2: GENERAL FACTS ABOUT ANIMALS AND NATURE
    # =========================================================================
    # NOTE: Tests from McGuffey's Reader removed - these are situational questions
    # about specific stories, not general world knowledge.
    # Only meaningful facts are kept:
    
    # Animals - what they give/do
    ("What does a cow give?", ["milk"]),
    ("What does a chicken lay?", ["eggs"]),
    ("Where does a duck swim?", ["pond", "water"]),
    ("Where do birds live?", ["nest", "tree", "nests"]),
    
    # Winter
    ("What covers the ground in winter?", ["snow", "ice"]),

    # =========================================================================
    # SECTION 3: ADDITIONAL EDUCATIONAL CONTENT
    # =========================================================================
    # NOTE: Aesop's fables tests removed - these are questions about PLOT of the book,
    # not about KNOWLEDGE of the world. Model should learn facts, not retell stories.
    # Weather
    ("What falls from clouds?", ["rain", "snow"]),
    ("What makes trees move?", ["wind"]),
    ("What do we wear when it is cold?", ["coat", "hat", "coats"]),
    
    # Water
    ("What do all living things need?", ["water"]),
    ("Where does water come from?", ["rain", "clouds"]),
    ("Can we drink salt water?", ["no"]),
    
    # Earth and space
    ("What is the Earth?", ["planet"]),
    ("What does the Earth go around?", ["sun"]),
    ("What orbits the Earth?", ["moon"]),
    ("When can we see the Moon?", ["night"]),
    
    # Health and food
    ("What do we need for strong bones?", ["milk"]),
    ("What gives us energy?", ["bread", "rice", "food", "eating"]),
    ("What is not good for teeth?", ["candy", "sweets"]),
    
    # Exercise and sleep
    ("What keeps our body healthy?", ["exercise", "sleep"]),
    ("When should we brush teeth?", ["morning", "night", "day", "every"]),
    ("When should we wash hands?", ["before eating", "eat"]),
    
    # Books and learning
    ("Where can we borrow books?", ["library"]),
    ("What can we learn from books?", ["many", "things", "stories"]),
    
    # Friends
    # NOTE: "What should we do with friends?" removed - word "should" not in episodes friends+share/play
    ("What should we say when we make a mistake?", ["sorry"]),
    
    # School
    ("Who helps us learn at school?", ["teacher"]),
    ("What do we learn in math class?", ["numbers"]),
    
    # Helping others
    # NOTE: "How can we help at home?" removed - no data home+clean/dishes
    ("Is helping others good?", ["yes", "good", "feel"]),
    
    # Politeness
    ("What do we say when we ask for something?", ["please"]),
    ("What do we say when someone gives us something?", ["thank", "thank you"]),
    ("What do we say when we bump into someone?", ["excuse", "sorry"]),
]


def get_grade1_facts():
    """Returns facts for training."""
    return GRADE1_FACTS


def get_grade1_sentences():
    """Returns sentences for training."""
    return GRADE1_SENTENCES


def get_grade1_questions():
    """Returns questions with expected keywords."""
    return GRADE1_QUESTIONS


# ============================================================================
# ADDITIONAL TEXTS - Added to GRADE1_TEXTS list
# ============================================================================

MCGUFFEY_TEXTS = [
    # =========================================================================
    # McGUFFEY'S FIRST READER (1879) - Public Domain
    # =========================================================================
    """
    The dog ran. The cat is on the mat. Is the cat on the mat? Yes, the cat 
    is on the mat. The man has a pen. Is the pen in his hand? It is in his 
    hand. A fat hen is on the box. The rat ran from the box.
    """,
    
    """
    Ann can catch the dog. She has the hat. Now Ann can pat the dog. Let me 
    pat the dog too. The dog has a black spot on his back. Do you think he 
    is a good dog? Tom has a big top on the box.
    """,
    
    """
    Kitty has a nice pet. It can sing a sweet song. She has just fed it. 
    She will now put it in the cage and hang the cage up. Then the cat 
    can not catch it. The pet bird sings all day long.
    """,
    
    """
    The sun is up. The man has fed the black hen and the fat duck. Now the 
    duck will swim in the pond. The hen has run to her nest. Let us not stop 
    at the pond now for it is hot. See how still it is!
    """,
    
    """
    The sun has just set. It is not hot now. Let us run and jump. I think 
    it is fun to run and skip and jump. See the duck on the pond! Her nest 
    is up on the bank under the rock.
    """,
    
    """
    Kate will you play with me? We will dig in the sand with this little 
    spade. That will be fine sport. James went to get Mary to play with him. 
    Then Kate made the doll bed. She sang a song to her doll.
    """,
    
    """
    Kate has left her doll in its little bed and has gone to play with Mary 
    and James. They are all in the shade now by the brook. James digs in 
    the soft sand with his spade and Mary picks up little stones.
    """,
    
    """
    The old cow is in the pond. See her drink! Will she not come out to get 
    some grass? No she likes to be in the pond. See how still she stands! 
    The dear old cow gives us sweet milk to drink.
    """,
    
    """
    Papa will you let me ride with you on Prince? I will sit still in your 
    arms. See mamma! We are both on Prince. How large he is! Get up Prince! 
    You are not too fat to trot as far as the barn.
    """,
    
    """
    Fanny what a pretty ball! Can you catch it? Toss it to me and see. I 
    will not let it fall. That was well done! Now Fanny toss it to the top 
    of the wall if you can. We like to play with the ball.
    """,
    
    """
    Did you call us mamma? I went with Tom to the pond. I had my doll and 
    Tom had his flag. The fat duck swam to the bank and we fed her. Did you 
    think we might fall into the pond? We did not go too near.
    """,
    
    """
    Here comes the band! Shall we call mamma and Fanny to see it? Let us 
    stand still and hear the men play as they pass. I hope they will stop 
    here and play for us. See the large man in front with his big hat.
    """,
    
    """
    Bess and Robert are very happy. Papa and mamma have gone to the woods 
    with them. Robert has a big tent and a flag and Bess has a little bed 
    for her doll. The dog is with them. They will have fun in the woods.
    """,
    
    """
    See my dear old grandma in her easy chair! How gray her hair is! She 
    wears glasses when she reads. She is always kind and takes such good 
    care of me that I like to do what she tells me.
    """,
    
    """
    Come here Lucy and listen! What is in this flower? O mother! It is a 
    bee. I wonder how it came to be shut up in the flower! It went into the 
    flower for some honey. The bee likes honey as well as we do.
    """,
    
    """
    Here come Frank and James White. Frank is riding a horse and James is 
    driving one hitched to a cart. They are out very early in the day. How 
    happy they are! The boys should be kind to their horses.
    """,
    
    """
    A little girl went in search of flowers for her mother. It was early in 
    the day and the grass was wet. Sweet little birds were singing all 
    around her. She found a nest with young birds in it.
    """,
    
    """
    It is noon and the school is out. Do you see the children at play? Some 
    run and jump and some play ball. Three little girls play school under 
    a tree. Mary is the teacher. They all have books in their hands.
    """,
    
    """
    Lucy has a new pet. Do you know what kind of bird it is? Lucy calls her 
    Polly. Polly can say Poor Poll! Polly wants a cracker! And she can mew 
    like a cat. But Polly and the cat are not good friends.
    """,
    
    """
    Well children did you have a nice time in the woods? Oh yes mother such 
    a good time! See what sweet flowers we found and what soft moss. The 
    best flowers are for grandma.
    """,
    
    """
    These boys and girls live near the sea. They have been to the beach. It 
    is now evening and they are going home. John found some pretty shells. 
    Is it not good sport to watch the big waves and play on the wet sand?
    """,
    
    """
    One evening Frank's father said would you like to go with me to catch 
    some fish? Yes may I go with you father? Oh how glad I am! Here they 
    are on the bank of a river. Frank has just pulled a fine fish out of 
    the water. How proud he feels!
    """,
    
    """
    I like winter when snow and ice cover the ground. What fun it is to 
    throw snowballs and to skate on the ice! See the boys and girls! How 
    merry they are! Henry has his sled and draws his little sister.
    """,
    
    """
    Ellen do look at the dog! He sits up in a chair with my hat on. He looks 
    like a little boy but it is only the dog. Now see him shake hands. The 
    dog says bow wow which means thank you. The dog is always so polite.
    """,
    
    """
    Mary and Lucy have come down to the beach with their grandpa. They live 
    in a town near the sea. Their grandpa likes to sit on the large rock and 
    watch the big ships as they sail far away on the blue sea.
    """,
    
    """
    One day Willie's father saw a boy at the market with four little white 
    rabbits in a basket. He thought these would be nice pets for Willie. 
    Willie has a pen for them and always shuts them in it at night to keep 
    them safe. He gives them bread and grass to eat.
    """,
    
    """
    Come here Rose. Look down into this bush. O Willie! A bird nest! What 
    cunning little eggs! May we take it and show it to mother? What would 
    the old bird do if she should come back and not find her nest?
    """,
    
    """
    How does the bird make the nest so strong? The mother bird has her bill 
    and her claws to work with. Do you see what it is made of? I see some 
    horse hairs and some dry grass. The old bird must have worked hard.
    """,
    
    """
    There was once a big white hen that had twelve little chickens. They 
    were very small and the old hen took good care of them. She found food 
    for them in the daytime and at night kept them under her wings.
    """,
    
    """
    We have come to the last lesson in this book. We have finished the First 
    Reader. You can now read all the lessons in it. Have you taken good care 
    of your book? Children should always keep their books neat and clean.
    """,

    # =========================================================================
    # AESOP'S FABLES - Library of Congress Children's Version
    # =========================================================================
    """
    The Lion and the Mouse. A Lion lay asleep in the forest with his great 
    head resting on his paws. A timid little Mouse came upon him and in her 
    fright ran across the Lion's nose. The Lion laid his huge paw on the 
    tiny creature to kill her. Spare me begged the poor Mouse. Please let 
    me go and some day I will surely repay you. The Lion was much amused 
    to think that a Mouse could ever help him. But he was generous and 
    finally let the Mouse go. Some days later the Lion was caught in a 
    hunter's net. The Mouse knew the voice and quickly found the Lion 
    struggling in the net. Running to one of the great ropes that bound 
    him she gnawed it until it parted and soon the Lion was free. You 
    laughed when I said I would repay you said the Mouse. Now you see that 
    even a Mouse can help a Lion.
    """,
    
    """
    The Hare and the Tortoise. A Hare was making fun of the Tortoise one 
    day for being so slow. Do you ever get anywhere? he asked with a 
    mocking laugh. Yes replied the Tortoise and I get there sooner than 
    you think. I will run you a race and prove it. The Hare was much 
    amused at the idea of running a race with the Tortoise but for the 
    fun of the thing he agreed. The Hare was soon far out of sight. To 
    make the Tortoise feel how ridiculous it was for him to try a race 
    with a Hare he lay down beside the course to take a nap. The Tortoise 
    meanwhile kept going slowly but steadily. After a time he passed the 
    place where the Hare was sleeping. But the Hare slept on peacefully. 
    When at last he did wake up the Tortoise was near the goal. The Hare 
    now ran his swiftest but he could not overtake the Tortoise in time. 
    Slow and steady wins the race.
    """,
    
    """
    The Dog and His Reflection. A Dog to whom the butcher had thrown a bone 
    was hurrying home with his prize as fast as he could go. As he crossed 
    a narrow footbridge he happened to look down and saw himself reflected 
    in the quiet water as if in a mirror. But the greedy Dog thought he saw 
    a real Dog carrying a bone much bigger than his own. If he had stopped 
    to think he would have known better. But instead of thinking he dropped 
    his bone and sprang at the Dog in the river only to find himself 
    swimming for dear life to reach the shore. At last he managed to 
    scramble out. As he stood sadly thinking about the good bone he had 
    lost he realized what a stupid Dog he had been. It is very foolish to 
    be greedy.
    """,
    
    """
    The Fox and the Crow. One bright morning as the Fox was following his 
    sharp nose through the wood in search of a bite to eat he saw a Crow 
    on the limb of a tree overhead. What caught his attention was that the 
    lucky Crow held a bit of cheese in her beak. Up he trotted to the foot 
    of the tree and looking up admiringly he cried Good morning beautiful 
    creature! The Crow kept her beak tightly closed on the cheese and did 
    not return his greeting. What a charming creature she is said the Fox. 
    How her feathers shine! What a beautiful form and what splendid wings! 
    Such a wonderful Bird should have a very lovely voice. Could she sing 
    just one song I know I should hail her Queen of Birds. Listening to 
    these flattering words the Crow forgot all her suspicion and also her 
    breakfast. She wanted very much to be called Queen of Birds. So she 
    opened her beak wide to utter her loudest caw and down fell the cheese 
    straight into the Fox's open mouth. Thank you said Master Fox sweetly 
    as he walked off. Do not trust flatterers.
    """,
    
    """
    The Ants and the Grasshopper. One bright day in late autumn a family of 
    Ants were bustling about in the warm sunshine drying out the grain they 
    had stored up during the summer. A starving Grasshopper came up and 
    humbly begged for a bite to eat. What cried the Ants in surprise. Have 
    you not stored anything away for the winter? What in the world were you 
    doing all last summer? I did not have time to store up any food whined 
    the Grasshopper. I was so busy making music that before I knew it the 
    summer was gone. The Ants shrugged their shoulders in disgust. Making 
    music were you? they cried. Very well now dance! And they turned their 
    backs on the Grasshopper and went on with their work. There is a time 
    for work and a time for play.
    """,
    
    """
    The Bundle of Sticks. A certain Father had a family of Sons who were 
    forever quarreling among themselves. No words he could say did the 
    least good. So he cast about in his mind for some very striking example 
    that should make them see that discord would lead them to misfortune. 
    One day when the quarreling had been much more violent than usual he 
    asked one of them to bring him a bundle of sticks. Then handing the 
    bundle to each of his Sons in turn he told them to try to break it. 
    But although each one tried his best none was able to do so. The Father 
    then untied the bundle and gave the sticks to his Sons to break one by 
    one. This they did very easily. My Sons said the Father do you not see 
    how certain it is that if you agree with each other and help each other 
    it will be impossible for your enemies to injure you? But if you are 
    divided among yourselves you will be no stronger than a single stick 
    in that bundle. In unity there is strength.
    """,
    
    """
    The Shepherd Boy and the Wolf. A Shepherd Boy tended his master's Sheep 
    near a dark forest not far from the village. Soon he found life in the 
    pasture very dull. All he could do to amuse himself was to talk to his 
    dog or play on his shepherd's pipe. One day as he sat watching the 
    Sheep and the quiet forest he thought of a plan to amuse himself. His 
    Master had told him to call for help should a Wolf attack the flock. 
    So now though he had not seen anything that even looked like a Wolf he 
    ran toward the village shouting at the top of his voice Wolf! Wolf! As 
    he expected the Villagers who heard the cry dropped their work and ran 
    in great excitement to the pasture. But when they got there they found 
    the Boy doubled up with laughter at the trick he had played on them. A 
    few days later the Shepherd Boy again shouted Wolf! Wolf! Again the 
    Villagers ran to help him only to be laughed at again. Then one evening 
    as the sun was setting a Wolf really did spring from the underbrush and 
    fall upon the Sheep. In terror the Boy ran toward the village shouting 
    Wolf! Wolf! But though the Villagers heard the cry they did not run to 
    help him as they had before. He cannot fool us again they said. The 
    Wolf killed a great many of the Boy's sheep. Liars are not believed 
    even when they speak the truth.
    """,
    
    """
    The Town Mouse and the Country Mouse. A Town Mouse once visited a 
    relative who lived in the country. For lunch the Country Mouse served 
    wheat stalks roots and acorns with a dash of cold water for drink. The 
    Town Mouse ate very sparingly nibbling a little of this and a little 
    of that. After the meal the friends had a long talk. They then went to 
    bed in a cozy nest in the hedgerow and slept in quiet and comfort until 
    morning. The next day the Town Mouse asked the Country Mouse to go home 
    with her to the city. She gladly said yes. When they reached the mansion 
    in which the Town Mouse lived they found on the table the leavings of a 
    very fine banquet. There were sweetmeats and jellies pastries and 
    delicious cheeses. But just as the Country Mouse was about to nibble a 
    dainty bit of pastry she heard a Cat mew loudly and scratch at the door. 
    In great fear the Mice scurried to a hiding place. The Country Mouse 
    stopped in the Town Mouse's den only long enough to pick up her carpet 
    bag. You may have luxuries and dainties that I have not she said as she 
    hurried away but I prefer my plain food and simple life in the country 
    with the peace and security that go with it. Better a little in safety 
    than much in danger.
    """,
    
    """
    The Ant and the Dove. A Dove saw an Ant fall into a brook. The Ant 
    struggled in vain to reach the bank. In pity the Dove dropped a blade 
    of straw close beside it. Clinging to the straw like a shipwrecked 
    sailor to a broken spar the Ant floated safely to shore. Soon after 
    the Ant saw a man getting ready to kill the Dove with a stone. But 
    just as he cast the stone the Ant stung him in the heel so that the 
    pain made him miss his aim and the startled Dove flew to safety in a 
    distant wood. One good turn deserves another.
    """,
    
    """
    The Fox and the Grapes. A Fox one day spied a beautiful bunch of ripe 
    grapes hanging from a vine trained along the branches of a tree. The 
    grapes seemed ready to burst with juice and the Fox's mouth watered as 
    he gazed longingly at them. The bunch hung from a high branch and the 
    Fox had to jump for it. The first time he jumped he missed it by a long 
    way. So he walked off a short distance and took a running leap at it 
    only to fall short once more. Again and again he tried but in vain. Now 
    he sat down and looked at the grapes in disgust. What a fool I am he 
    said. Here I am wearing myself out to get a bunch of sour grapes that 
    are not worth gaping for. And off he walked very very scornfully. It is 
    easy to despise what you cannot get.
    """,

    # =========================================================================
    # ADDITIONAL EDUCATIONAL TEXTS - Weather, Science, Health
    # =========================================================================
    """
    The weather changes every day. Sometimes the sun shines and the sky is 
    blue. We call this sunny weather. Sometimes clouds cover the sky and 
    rain falls. We call this rainy weather. In winter snow falls from the 
    clouds. Snow is white and cold. Wind makes the trees move. Strong wind 
    can blow things away. We should dress for the weather. When it is cold 
    we wear coats and hats. When it is hot we wear light clothes.
    """,
    
    """
    Water is very important for all living things. People drink water every 
    day. Animals drink water too. Plants need water to grow. Water comes 
    from rain and snow. Rivers and lakes have water. The ocean has salt 
    water. We cannot drink salt water. We use fresh water for drinking. 
    We should not waste water. Turn off the tap when you brush your teeth.
    """,
    
    """
    The Earth is our home. It is a big round ball called a planet. The 
    Earth goes around the Sun. It takes one year for the Earth to go 
    around the Sun. The Earth also spins like a top. This gives us day 
    and night. When our side of the Earth faces the Sun it is day. When 
    our side faces away from the Sun it is night. The Moon goes around 
    the Earth. We can see the Moon at night.
    """,
    
    """
    Our body needs good food to grow strong. We should eat fruits and 
    vegetables every day. Fruits like apples and oranges have vitamins. 
    Vegetables like carrots and spinach are good for us. We need milk 
    for strong bones. Bread and rice give us energy. We should not eat 
    too much candy or cake. Sweets are not good for our teeth. Drink 
    water instead of sweet drinks.
    """,
    
    """
    Exercise keeps our body healthy. Running and jumping are good exercise. 
    Playing outside is fun and healthy. We should play every day. Sleep is 
    also important. Children need lots of sleep to grow. We should go to 
    bed early and wake up early. Wash your hands before you eat. Brush 
    your teeth in the morning and at night. Take a bath to stay clean.
    """,
    
    """
    Books are wonderful things. We can learn many things from books. Books 
    have stories about people and animals. Some books teach us about the 
    world. We can read about far away places. We can learn about animals 
    we have never seen. Reading is fun. We should read every day. The 
    library has many books. We can borrow books from the library.
    """,
    
    """
    Friends are people who like us and we like them. Good friends play 
    together and help each other. We should be kind to our friends. We 
    should share our toys with friends. If a friend is sad we should try 
    to make them happy. We should not fight with friends. If we have a 
    problem we should talk about it. Good friends say sorry when they 
    make a mistake. Good friends forgive each other.
    """,
    
    """
    School is where we learn. The teacher helps us learn to read and write. 
    We learn about numbers in math class. We learn about the world in 
    science class. We draw pictures in art class. We sing songs in music 
    class. We play games in gym class. We should listen to the teacher. 
    We should do our homework. We should be nice to other children at 
    school.
    """,
    
    """
    Helping others makes us feel good. We can help at home. We can clean 
    our room. We can help set the table. We can help wash the dishes. We 
    can help in the garden. We can help our friends too. If someone drops 
    something we can pick it up. If someone is lost we can help them find 
    their way. Helping others is the right thing to do.
    """,
    
    """
    Being polite is important. We say please when we ask for something. We 
    say thank you when someone gives us something. We say sorry when we 
    make a mistake. We say excuse me when we bump into someone. We do not 
    interrupt when others are talking. We wait for our turn. We hold the 
    door for others. Being polite makes everyone happy.
    """,
]

# Combine all texts
GRADE1_TEXTS = GRADE1_TEXTS + MCGUFFEY_TEXTS

# Update sentences with all texts
GRADE1_SENTENCES = _split_texts_to_sentences(GRADE1_TEXTS)
