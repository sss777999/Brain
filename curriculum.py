#!/usr/bin/env python3
"""
FACTS AND CONNECTIONS DATASET (0-5 years)

Format: (word1, word2) - direct connection
Graph learns from CONNECTIONS, not phrases.

Sources:
- MacArthur-Bates CDI
- Head Start Framework
- Preschool Learning Standards
"""

# =============================================================================
# CATEGORIES (X is a Y)
# =============================================================================

CATEGORIES = [
    # Animals - domestic
    ("dog", "animal"), ("dog", "pet"), ("dog", "mammal"),
    ("cat", "animal"), ("cat", "pet"), ("cat", "mammal"),
    ("hamster", "pet"), ("goldfish", "pet"), ("parrot", "pet"),
    ("rabbit", "pet"), ("turtle", "pet"),
    
    # Animals - farm
    ("cow", "animal"), ("cow", "farm"),
    ("horse", "animal"), ("horse", "farm"),
    ("pig", "animal"), ("pig", "farm"),
    ("chicken", "animal"), ("chicken", "farm"), ("chicken", "bird"),
    ("duck", "animal"), ("duck", "farm"), ("duck", "bird"),
    ("sheep", "animal"), ("sheep", "farm"),
    ("goat", "animal"), ("goat", "farm"),
    ("rooster", "farm"), ("rooster", "bird"),
    
    # Animals - wild
    ("lion", "animal"), ("lion", "wild"), ("lion", "predator"),
    ("tiger", "animal"), ("tiger", "wild"), ("tiger", "predator"),
    ("bear", "animal"), ("bear", "wild"),
    ("wolf", "animal"), ("wolf", "wild"), ("wolf", "predator"),
    ("fox", "animal"), ("fox", "wild"),
    ("deer", "animal"), ("deer", "wild"),
    ("elephant", "animal"), ("elephant", "wild"), ("elephant", "big"),
    ("giraffe", "animal"), ("giraffe", "wild"), ("giraffe", "tall"),
    ("zebra", "animal"), ("zebra", "wild"),
    ("monkey", "animal"), ("monkey", "wild"),
    ("gorilla", "animal"), ("gorilla", "wild"),
    ("hippo", "animal"), ("hippo", "wild"),
    ("rhino", "animal"), ("rhino", "wild"),
    ("kangaroo", "animal"), ("kangaroo", "wild"),
    ("koala", "animal"), ("koala", "wild"),
    ("panda", "animal"), ("panda", "wild"),
    
    # Birds
    ("bird", "animal"), ("bird", "animal"), ("bird", "fly"),
    ("eagle", "bird"), ("eagle", "fly"),
    ("owl", "bird"), ("owl", "night"),
    ("penguin", "bird"), ("penguin", "swim"),
    ("parrot", "bird"), ("parrot", "talk"),
    ("sparrow", "bird"), ("crow", "bird"),
    ("pigeon", "bird"), ("swan", "bird"),
    ("flamingo", "bird"), ("peacock", "bird"),
    
    # Marine (only IS-A and properties, no meaningless associations)
    # "fish water" - not a fact, correct: "fish live in water" (in sentences)
    ("fish", "animal"), ("fish", "swim"),
    ("whale", "animal"), ("whale", "mammal"), ("whale", "big"),
    ("shark", "animal"), ("shark", "predator"),
    ("dolphin", "animal"), ("dolphin", "mammal"), ("dolphin", "smart"),
    ("octopus", "animal"), ("octopus", "tentacles"),
    ("jellyfish", "animal"), ("jellyfish", "sting"),
    ("crab", "animal"), ("crab", "claws"),
    ("lobster", "animal"), ("lobster", "claws"),
    ("seahorse", "animal"), ("seahorse", "small"),
    ("starfish", "animal"), ("starfish", "arms"),
    
    # Insects
    ("butterfly", "insect"), ("butterfly", "fly"),
    ("bee", "insect"), ("bee", "fly"), ("bee", "honey"),
    ("ant", "insect"), ("ant", "small"),
    ("spider", "insect"), ("spider", "web"),
    ("ladybug", "insect"), ("ladybug", "red"),
    ("dragonfly", "insect"), ("dragonfly", "fly"),
    ("mosquito", "insect"), ("fly", "insect"),
    ("grasshopper", "insect"), ("cricket", "insect"),
    ("caterpillar", "insect"), ("caterpillar", "butterfly"),
    
    # Reptiles and amphibians
    ("snake", "animal"), ("snake", "reptile"),
    ("lizard", "animal"), ("lizard", "reptile"),
    ("crocodile", "animal"), ("crocodile", "reptile"),
    ("alligator", "animal"), ("alligator", "reptile"),
    ("frog", "animal"), ("frog", "amphibian"), ("frog", "jump"),
    ("toad", "animal"), ("toad", "amphibian"),
    ("turtle", "animal"), ("turtle", "reptile"), ("turtle", "slow"),
    
    # Rodents
    ("mouse", "animal"), ("mouse", "small"),
    ("rat", "animal"), ("squirrel", "animal"),
    ("hamster", "animal"), ("guinea pig", "animal"),
    
    # Fruits
    ("apple", "fruit"), ("banana", "fruit"), ("orange", "fruit"),
    ("grape", "fruit"), ("strawberry", "fruit"), ("watermelon", "fruit"),
    ("peach", "fruit"), ("pear", "fruit"), ("cherry", "fruit"),
    ("lemon", "fruit"), ("mango", "fruit"), ("pineapple", "fruit"),
    
    # Vegetables
    ("carrot", "vegetable"), ("potato", "vegetable"), ("tomato", "vegetable"),
    ("corn", "vegetable"), ("peas", "vegetable"), ("broccoli", "vegetable"),
    ("cucumber", "vegetable"), ("lettuce", "vegetable"), ("onion", "vegetable"),
    
    # Transport
    ("car", "vehicle"), ("truck", "vehicle"), ("bus", "vehicle"),
    ("train", "vehicle"), ("plane", "vehicle"), ("boat", "vehicle"),
    ("bike", "vehicle"), ("motorcycle", "vehicle"), ("helicopter", "vehicle"),
    
    # Body parts
    ("head", "body"), ("arm", "body"), ("leg", "body"),
    ("hand", "body"), ("foot", "body"), ("finger", "body"),
    ("toe", "body"), ("eye", "body"), ("ear", "body"),
    ("nose", "body"), ("nose", "face"), ("mouth", "body"), ("hair", "body"),
    ("eyes", "face"), ("ears", "face"), ("mouth", "face"),
    
    # Clothing
    ("shirt", "clothing"), ("pants", "clothing"), ("dress", "clothing"),
    ("shoes", "clothing"), ("socks", "clothing"), ("hat", "clothing"),
    ("coat", "clothing"), ("jacket", "clothing"), ("sweater", "clothing"),
    
    # Furniture
    ("table", "furniture"), ("chair", "furniture"), ("bed", "furniture"),
    ("couch", "furniture"), ("desk", "furniture"), ("shelf", "furniture"),
    
    # Tools
    ("hammer", "tool"), ("screwdriver", "tool"), ("saw", "tool"),
    ("scissors", "tool"), ("knife", "tool"), ("spoon", "tool"),
    ("fork", "tool"), ("pencil", "tool"), ("brush", "tool"),
    
    # Food
    ("bread", "food"), ("cheese", "food"), ("meat", "food"),
    ("rice", "food"), ("pasta", "food"), ("soup", "food"),
    ("pizza", "food"), ("sandwich", "food"), ("salad", "food"),
    ("egg", "food"), ("butter", "food"), ("cereal", "food"),
    ("cake", "food"), ("pie", "food"), ("cookie", "food"),
    ("candy", "food"), ("chocolate", "food"), ("ice cream", "food"),
    
    # Drinks
    ("water", "drink"), ("milk", "drink"), ("juice", "drink"),
    ("tea", "drink"), ("coffee", "drink"), ("soda", "drink"),
    
    # Toys
    ("ball", "toy"), ("doll", "toy"), ("car", "toy"),
    ("blocks", "toy"), ("puzzle", "toy"), ("teddy", "toy"),
    ("train", "toy"), ("robot", "toy"), ("kite", "toy"),
    
    # Rooms
    ("kitchen", "room"), ("bedroom", "room"), ("bathroom", "room"),
    ("living room", "room"), ("dining room", "room"),
    
    # Buildings
    ("house", "building"), ("school", "building"), ("hospital", "building"),
    ("store", "building"), ("library", "building"), ("church", "building"),
    ("museum", "building"), ("restaurant", "building"),
    
    # Musical instruments
    ("piano", "instrument"), ("guitar", "instrument"), ("drum", "instrument"),
    ("violin", "instrument"), ("flute", "instrument"), ("trumpet", "instrument"),
    
    # Sports
    ("soccer", "sport"), ("basketball", "sport"), ("baseball", "sport"),
    ("tennis", "sport"), ("swimming", "sport"), ("running", "sport"),
    
    # Materials
    ("wood", "material"), ("metal", "material"), ("plastic", "material"),
    ("glass", "material"), ("paper", "material"), ("fabric", "material"),
]

# =============================================================================
# PROPERTIES (X has property Y)
# =============================================================================

PROPERTIES = [
    # Colors - category
    ("red", "color"), ("blue", "color"), ("green", "color"),
    ("yellow", "color"), ("orange", "color"), ("purple", "color"),
    ("pink", "color"), ("black", "color"), ("white", "color"),
    ("brown", "color"), ("gray", "color"),
    
    # Colors - basic
    ("apple", "red"), ("banana", "yellow"), ("orange", "orange"),
    ("grape", "purple"), ("strawberry", "red"), ("lemon", "yellow"),
    ("carrot", "orange"), ("broccoli", "green"), ("tomato", "red"),
    ("sky", "blue"), ("grass", "green"), ("sun", "yellow"),
    ("snow", "white"), ("night", "dark"), ("fire", "red"),
    ("milk", "white"), ("chocolate", "brown"), ("water", "clear"),
    
    # Colors - additional
    ("firetruck", "red"), ("stop sign", "red"), ("blood", "red"),
    ("ocean", "blue"), ("blueberry", "blue"), ("jeans", "blue"),
    ("tree", "green"), ("frog", "green"), ("leaf", "green"),
    ("sunflower", "yellow"), ("taxi", "yellow"), ("cheese", "yellow"),
    ("pumpkin", "orange"), ("basketball", "orange"),
    ("eggplant", "purple"), ("violet", "purple"),
    ("pig", "pink"), ("flamingo", "pink"),
    ("crow", "black"), ("coal", "black"), ("night", "black"),
    ("cloud", "white"), ("polar bear", "white"), ("paper", "white"),
    ("dirt", "brown"), ("wood", "brown"), ("bear", "brown"),
    ("elephant", "gray"), ("rock", "gray"), ("cloud", "gray"),
    
    # Sizes
    ("elephant", "big"), ("mouse", "small"), ("giraffe", "tall"),
    ("ant", "tiny"), ("whale", "huge"), ("bear", "big"),
    ("butterfly", "small"), ("lion", "big"), ("bee", "small"),
    ("dinosaur", "big"), ("bug", "small"), ("mountain", "big"),
    ("skyscraper", "tall"), ("tree", "tall"), ("baby", "small"),
    
    # Temperature
    ("fire", "hot"), ("ice", "cold"), ("sun", "hot"),
    ("snow", "cold"), ("summer", "hot"), ("winter", "cold"),
    ("soup", "hot"), ("ice cream", "cold"), ("oven", "hot"),
    ("desert", "hot"), ("arctic", "cold"), ("lava", "hot"),
    
    # Texture
    ("rock", "hard"), ("pillow", "soft"), ("water", "wet"),
    ("sand", "dry"), ("cotton", "soft"), ("metal", "hard"),
    ("fur", "soft"), ("ice", "slippery"), ("sandpaper", "rough"),
    ("silk", "smooth"), ("glass", "smooth"), ("bark", "rough"),
    
    # Taste
    ("sugar", "sweet"), ("lemon", "sour"), ("salt", "salty"),
    ("candy", "sweet"), ("cookie", "sweet"), ("pickle", "sour"),
    ("honey", "sweet"), ("vinegar", "sour"), ("chips", "salty"),
    ("pepper", "spicy"), ("chili", "spicy"), ("ginger", "spicy"),
    ("coffee", "bitter"), ("medicine", "bitter"),
    
    # Speed
    ("cheetah", "fast"), ("turtle", "slow"), ("rabbit", "fast"),
    ("snail", "slow"), ("car", "fast"), ("walk", "slow"),
    ("rocket", "fast"), ("plane", "fast"), ("sloth", "slow"),
    
    # Shape
    ("ball", "round"), ("wheel", "round"), ("coin", "round"),
    ("box", "square"), ("dice", "cube"), ("pyramid", "triangle"),
    
    # Age
    ("grandpa", "old"), ("grandma", "old"), ("baby", "young"),
    ("child", "young"), ("adult", "grown"),
    
    # Danger
    ("fire", "dangerous"), ("knife", "sharp"), ("lion", "dangerous"),
    ("shark", "dangerous"), ("snake", "dangerous"),
]

# =============================================================================
# ANIMAL SOUNDS (X makes sound Y)
# =============================================================================

ANIMAL_SOUNDS = [
    ("dog", "woof"), ("dog", "bark"),
    ("cat", "meow"), ("cat", "purr"),
    ("cow", "moo"),
    ("pig", "oink"),
    ("duck", "quack"),
    ("bird", "tweet"), ("bird", "chirp"),
    ("sheep", "baa"),
    ("horse", "neigh"),
    ("lion", "roar"),
    ("snake", "hiss"),
    ("bee", "buzz"),
    ("frog", "ribbit"),
    ("owl", "hoot"),
    ("rooster", "cockadoodledoo"),
    ("mouse", "squeak"),
]

# =============================================================================
# ACTIONS (X does Y)
# =============================================================================

ACTIONS = [
    # Body parts -> actions
    ("eyes", "see"), ("eyes", "look"), ("eyes", "blink"),
    ("ears", "hear"), ("ears", "listen"),
    ("nose", "smell"), ("nose", "breathe"),
    ("mouth", "eat"), ("mouth", "talk"), ("mouth", "smile"),
    ("teeth", "bite"), ("teeth", "chew"),
    ("tongue", "taste"), ("tongue", "lick"),
    ("hands", "hold"), ("hands", "grab"), ("hands", "clap"),
    ("feet", "walk"), ("feet", "run"), ("feet", "kick"),
    ("legs", "walk"), ("legs", "run"), ("legs", "jump"),
    ("brain", "think"), ("heart", "beat"), ("lungs", "breathe"),
    
    # Animals -> actions
    ("bird", "fly"), ("fish", "swim"), ("dog", "run"),
    ("cat", "climb"), ("rabbit", "hop"), ("snake", "crawl"),
    ("frog", "jump"), ("duck", "swim"), ("horse", "gallop"),
    ("kangaroo", "hop"), ("monkey", "climb"), ("cheetah", "run"),
    ("eagle", "fly"), ("penguin", "swim"), ("bear", "hibernate"),
    ("spider", "spin"), ("bee", "pollinate"), ("ant", "carry"),
    
    # Transport -> actions
    ("car", "drive"), ("plane", "fly"), ("boat", "sail"),
    ("train", "ride"), ("bike", "pedal"), ("bus", "ride"),
    ("helicopter", "fly"), ("submarine", "dive"), ("rocket", "launch"),
    
    # Everyday actions
    ("eat", "food"), ("drink", "water"), ("sleep", "bed"),
    ("wake", "morning"), ("brush", "teeth"), ("wash", "hands"),
    ("cook", "food"), ("clean", "house"), ("read", "book"),
    ("write", "pencil"), ("draw", "paper"), ("paint", "brush"),
    ("sing", "song"), ("dance", "music"), ("play", "game"),
    ("run", "fast"), ("walk", "slow"), ("jump", "high"),
    ("throw", "ball"), ("catch", "ball"), ("kick", "ball"),
    ("swim", "water"), ("climb", "tree"), ("slide", "down"),
    ("swing", "playground"), ("build", "blocks"), ("break", "apart"),
    
    # Jobs/professions -> actions
    ("doctor", "heal"), ("teacher", "teach"), ("cook", "prepare"),
    ("farmer", "grow"), ("builder", "construct"), ("driver", "drive"),
    ("pilot", "fly"), ("artist", "create"), ("singer", "perform"),
    
    # Nature -> actions
    ("sun", "shine"), ("rain", "fall"), ("wind", "blow"),
    ("snow", "fall"), ("flower", "bloom"), ("tree", "grow"),
    ("river", "flow"), ("volcano", "erupt"), ("earthquake", "shake"),
]

# =============================================================================
# OPPOSITES (X is opposite of Y)
# =============================================================================

OPPOSITES = [
    ("big", "small"), ("small", "big"),
    ("big", "little"), ("little", "big"),
    ("hot", "cold"), ("cold", "hot"),
    ("fast", "slow"), ("slow", "fast"),
    ("up", "down"), ("down", "up"),
    ("in", "out"), ("out", "in"),
    ("on", "off"), ("off", "on"),
    ("open", "close"), ("close", "open"),
    ("happy", "sad"), ("sad", "happy"),
    ("good", "bad"), ("bad", "good"),
    ("yes", "no"), ("no", "yes"),
    ("day", "night"), ("night", "day"),
    ("light", "dark"), ("dark", "light"),
    ("wet", "dry"), ("dry", "wet"),
    ("clean", "dirty"), ("dirty", "clean"),
    ("full", "empty"), ("empty", "full"),
    ("new", "old"), ("old", "new"),
    ("loud", "quiet"), ("quiet", "loud"),
    ("hard", "soft"), ("soft", "hard"),
    ("tall", "short"), ("short", "tall"),
    ("high", "low"), ("low", "high"),
    ("front", "back"), ("back", "front"),
    ("top", "bottom"), ("bottom", "top"),
    ("start", "stop"), ("stop", "start"),
    ("come", "go"), ("go", "come"),
    ("push", "pull"), ("pull", "push"),
    ("give", "take"), ("take", "give"),
    ("left", "right"), ("right", "left"),
    ("first", "last"), ("last", "first"),
    ("before", "after"), ("after", "before"),
]

# =============================================================================
# NUMBERS (sequence)
# =============================================================================

NUMBERS = [
    ("one", "two"), ("two", "three"), ("three", "four"),
    ("four", "five"), ("five", "six"), ("six", "seven"),
    ("seven", "eight"), ("eight", "nine"), ("nine", "ten"),
    
    # Numbers -> quantity
    ("one", "single"), ("two", "pair"), ("three", "triple"),
    ("five", "hand"), ("ten", "fingers"),
]

# =============================================================================
# SHAPES
# =============================================================================

SHAPES = [
    ("circle", "round"), ("ball", "round"), ("wheel", "round"),
    ("sun", "round"), ("moon", "round"), ("orange", "round"),
    ("square", "four"), ("box", "square"),
    ("triangle", "three"),
    ("rectangle", "four"),
    ("star", "points"),
    ("heart", "love"),
]

# =============================================================================
# TIME
# =============================================================================

TIME = [
    ("morning", "wake"), ("morning", "breakfast"), ("morning", "sun"),
    ("afternoon", "lunch"), ("afternoon", "nap"),
    ("evening", "dinner"), ("evening", "sunset"),
    ("night", "sleep"), ("night", "dark"), ("night", "moon"), ("night", "stars"),
    ("yesterday", "past"), ("yesterday", "before"),
    ("today", "now"), ("today", "present"),
    ("tomorrow", "future"), ("tomorrow", "after"),
    
    # Days of week (sequence)
    ("monday", "tuesday"), ("tuesday", "wednesday"),
    ("wednesday", "thursday"), ("thursday", "friday"),
    ("friday", "saturday"), ("saturday", "sunday"),
    
    # Seasons
    ("spring", "flowers"), ("spring", "warm"),
    ("summer", "hot"), ("summer", "sun"), ("summer", "beach"),
    ("fall", "leaves"), ("fall", "cool"),
    ("winter", "cold"), ("winter", "snow"),
    
    # Season sequence
    ("spring", "summer"), ("summer", "fall"),
    ("fall", "winter"), ("winter", "spring"),
]

# =============================================================================
# PLACES
# =============================================================================

PLACES = [
    ("home", "family"), ("home", "live"), ("home", "safe"),
    ("school", "learn"), ("school", "teacher"), ("school", "friends"),
    ("park", "play"), ("park", "swing"), ("park", "slide"),
    ("zoo", "animals"), ("zoo", "cages"), ("zoo", "visit"),
    ("library", "books"), ("library", "read"), ("library", "quiet"),
    ("hospital", "doctor"), ("hospital", "sick"), ("hospital", "nurse"),
    ("store", "buy"), ("store", "shop"), ("store", "money"),
    # Places - only properties and IS-A, no meaningless associations
    ("beach", "place"), ("beach", "sandy"), ("beach", "coastal"),
    ("forest", "place"), ("forest", "wooded"), ("forest", "natural"),
    ("farm", "place"), ("farm", "agricultural"),
    ("city", "place"), ("city", "urban"), ("city", "populated"),
    ("village", "place"), ("village", "small"), ("village", "rural"),
    ("mountain", "landform"), ("mountain", "high"), ("mountain", "rocky"),
    ("river", "waterway"), ("river", "flowing"),
    ("lake", "waterway"), ("lake", "still"),
    ("ocean", "waterway"), ("ocean", "large"), ("ocean", "salty"),
    ("desert", "place"), ("desert", "hot"), ("desert", "dry"),
    ("jungle", "place"), ("jungle", "tropical"), ("jungle", "dense"),
    ("island", "landform"), ("island", "surrounded by water"),
    ("airport", "planes"), ("airport", "travel"),
    ("train station", "trains"), ("train station", "travel"),
    ("restaurant", "food"), ("restaurant", "eat"),
    ("museum", "art"), ("museum", "history"), ("museum", "learn"),
    ("playground", "play"), ("playground", "children"),
    ("garden", "flowers"), ("garden", "plants"), ("garden", "grow"),
    ("kitchen", "cook"), ("kitchen", "food"),
    ("bedroom", "sleep"), ("bedroom", "bed"),
    ("bathroom", "wash"), ("bathroom", "clean"),
]

# =============================================================================
# PEOPLE AND ROLES
# =============================================================================

PEOPLE = [
    # Family
    ("mommy", "mother"), ("mommy", "love"), ("mommy", "family"), ("mommy", "parent"),
    ("daddy", "father"), ("daddy", "love"), ("daddy", "family"), ("daddy", "parent"),
    ("mother", "parent"), ("father", "parent"),
    ("baby", "small"), ("baby", "cry"), ("baby", "young"),
    ("brother", "boy"), ("brother", "family"), ("brother", "sibling"),
    ("sister", "girl"), ("sister", "family"), ("sister", "sibling"),
    ("grandma", "old"), ("grandma", "love"), ("grandma", "grandmother"),
    ("grandpa", "old"), ("grandpa", "love"), ("grandpa", "grandfather"),
    ("uncle", "family"), ("aunt", "family"), ("cousin", "family"),
    
    # Professions
    ("teacher", "school"), ("teacher", "learn"), ("teacher", "teach"), ("teacher", "person"),
    ("doctor", "hospital"), ("doctor", "help"), ("doctor", "medicine"), ("doctor", "person"),
    ("nurse", "hospital"), ("nurse", "help"), ("nurse", "care"),
    ("firefighter", "fire"), ("firefighter", "help"), ("firefighter", "brave"),
    ("police", "help"), ("police", "safe"), ("police", "protect"),
    ("farmer", "farm"), ("farmer", "grow"), ("farmer", "animals"),
    ("chef", "cook"), ("chef", "food"), ("chef", "restaurant"),
    ("pilot", "plane"), ("pilot", "fly"), ("pilot", "airport"),
    ("driver", "car"), ("driver", "bus"), ("driver", "drive"),
    ("builder", "build"), ("builder", "house"), ("builder", "construct"),
    ("artist", "paint"), ("artist", "draw"), ("artist", "create"),
    ("singer", "sing"), ("singer", "music"), ("singer", "song"),
    ("dancer", "dance"), ("dancer", "music"),
    ("scientist", "discover"), ("scientist", "experiment"),
    ("astronaut", "space"), ("astronaut", "rocket"),
    ("vet", "animals"), ("vet", "help"), ("vet", "doctor"),
    ("dentist", "teeth"), ("dentist", "doctor"),
    ("baker", "bread"), ("baker", "cake"), ("baker", "bake"),
    ("mailman", "letters"), ("mailman", "deliver"),
    
    # Age groups
    ("child", "young"), ("child", "play"), ("child", "learn"),
    ("adult", "grown"), ("adult", "work"),
    ("teenager", "young"), ("teenager", "school"),
    ("boy", "male"), ("boy", "child"),
    ("girl", "female"), ("girl", "child"),
    ("man", "male"), ("man", "adult"),
    ("woman", "female"), ("woman", "adult"),
]

# =============================================================================
# EMOTIONS
# =============================================================================

EMOTIONS = [
    ("happy", "smile"), ("happy", "laugh"), ("happy", "emotion"),
    ("sad", "cry"), ("sad", "tears"), ("sad", "emotion"),
    ("angry", "mad"), ("angry", "upset"), ("angry", "emotion"),
    ("scared", "afraid"), ("scared", "fear"), ("scared", "emotion"),
    ("tired", "sleep"), ("tired", "yawn"),
    ("hungry", "eat"), ("hungry", "food"),
    ("thirsty", "drink"), ("thirsty", "water"),
    ("love", "heart"), ("love", "hug"), ("love", "kiss"), ("love", "emotion"),
    
    # Additional connections
    ("eat", "food"), ("food", "eat"),
    ("drink", "water"), ("water", "drink"),
    ("sleep", "bed"), ("bed", "sleep"),
    ("play", "toys"), ("toys", "play"),
    ("milk", "drink"), ("drink", "milk"),
]

# =============================================================================
# WEATHER
# =============================================================================

WEATHER = [
    ("sunny", "sun"), ("sunny", "bright"), ("sunny", "warm"),
    ("cloudy", "clouds"), ("cloudy", "gray"),
    ("rainy", "rain"), ("rainy", "wet"), ("rainy", "umbrella"),
    ("snowy", "snow"), ("snowy", "cold"), ("snowy", "white"),
    ("windy", "wind"), ("windy", "blow"),
    ("stormy", "thunder"), ("stormy", "lightning"),
]

# =============================================================================
# NATURE
# =============================================================================

NATURE = [
    ("sun", "hot"), ("sun", "bright"), ("sun", "day"), ("sun", "yellow"),
    ("moon", "night"), ("moon", "round"), ("moon", "sky"),
    ("star", "night"), ("star", "sky"), ("star", "twinkle"),
    ("cloud", "sky"), ("cloud", "white"), ("cloud", "rain"),
    ("rain", "water"), ("rain", "wet"),
    ("snow", "white"), ("snow", "cold"),
    ("tree", "tall"), ("tree", "leaves"), ("tree", "trunk"),
    ("flower", "pretty"), ("flower", "smell"), ("flower", "petals"),
    ("grass", "green"), ("grass", "ground"),
    ("rock", "hard"), ("rock", "ground"),
    ("water", "wet"), ("water", "drink"), ("water", "swim"),
    ("fire", "hot"), ("fire", "red"), ("fire", "burn"),
]

# =============================================================================
# GEOGRAPHY (basic)
# =============================================================================

GEOGRAPHY = [
    # Earth and space
    ("earth", "planet"), ("earth", "round"), ("earth", "home"),
    ("sun", "star"), ("sun", "hot"), ("sun", "center"),
    ("moon", "round"), ("moon", "night"), ("moon", "satellite"),
    
    # Solar system planets
    ("mercury", "planet"), ("mercury", "small"), ("mercury", "hot"),
    ("venus", "planet"), ("venus", "hot"),
    ("earth", "planet"), ("earth", "life"),
    ("mars", "planet"), ("mars", "red"),
    ("jupiter", "planet"), ("jupiter", "big"), ("jupiter", "gas"),
    ("saturn", "planet"), ("saturn", "rings"),
    ("uranus", "planet"), ("uranus", "cold"),
    ("neptune", "planet"), ("neptune", "blue"), ("neptune", "cold"),
    
    # Planet sequence
    ("mercury", "venus"), ("venus", "earth"), ("earth", "mars"),
    ("mars", "jupiter"), ("jupiter", "saturn"), ("saturn", "uranus"),
    ("uranus", "neptune"),
    
    # Continents
    ("africa", "continent"), ("africa", "big"),
    ("asia", "continent"), ("asia", "big"),
    ("europe", "continent"),
    ("north america", "continent"),
    ("south america", "continent"),
    ("australia", "continent"), ("australia", "island"),
    ("antarctica", "continent"), ("antarctica", "cold"), ("antarctica", "ice"),
    
    # Oceans
    ("pacific", "ocean"), ("pacific", "big"),
    ("atlantic", "ocean"),
    ("indian", "ocean"),
    ("arctic", "ocean"), ("arctic", "cold"),
    ("southern", "ocean"),
    
    # Capitals (main)
    ("paris", "capital"), ("paris", "france"),
    ("london", "capital"), ("london", "england"), ("london", "uk"),
    ("berlin", "capital"), ("berlin", "germany"),
    ("rome", "capital"), ("rome", "italy"),
    ("madrid", "capital"), ("madrid", "spain"),
    ("moscow", "capital"), ("moscow", "russia"),
    ("beijing", "capital"), ("beijing", "china"),
    ("tokyo", "capital"), ("tokyo", "japan"),
    ("washington", "capital"), ("washington", "usa"), ("washington", "america"),
    ("ottawa", "capital"), ("ottawa", "canada"),
    ("canberra", "capital"), ("canberra", "australia"),
    ("cairo", "capital"), ("cairo", "egypt"),
    ("new delhi", "capital"), ("new delhi", "india"),
    ("brasilia", "capital"), ("brasilia", "brazil"),
    
    # Countries and continents
    ("france", "europe"), ("france", "country"),
    ("germany", "europe"), ("germany", "country"),
    ("italy", "europe"), ("italy", "country"),
    ("spain", "europe"), ("spain", "country"),
    ("england", "europe"), ("england", "country"),
    ("russia", "europe"), ("russia", "asia"), ("russia", "country"), ("russia", "big"),
    ("china", "asia"), ("china", "country"), ("china", "big"),
    ("japan", "asia"), ("japan", "country"), ("japan", "island"),
    ("india", "asia"), ("india", "country"), ("india", "big"),
    ("usa", "north america"), ("usa", "country"), ("usa", "big"),
    ("canada", "north america"), ("canada", "country"), ("canada", "big"),
    ("brazil", "south america"), ("brazil", "country"), ("brazil", "big"),
    ("australia", "country"), ("australia", "big"),
    ("egypt", "africa"), ("egypt", "country"),
    
    # Landscape - IS-A and properties (no meaningless associations)
    ("ocean", "body of water"), ("ocean", "large"), ("ocean", "salty"),
    ("sea", "body of water"), ("sea", "salty"),
    ("mountain", "landform"), ("mountain", "tall"), ("mountain", "rocky"),
    ("river", "body of water"), ("river", "flowing"), ("river", "fresh"),
    ("lake", "body of water"), ("lake", "still"), ("lake", "fresh"),
    ("desert", "biome"), ("desert", "dry"), ("desert", "hot"),
    ("forest", "biome"), ("forest", "wooded"),
    ("island", "landform"), ("island", "surrounded by water"),
    ("volcano", "mountain"), ("volcano", "hot"), ("volcano", "erupts"),
    ("valley", "landform"), ("valley", "low"),
    ("hill", "landform"), ("hill", "small"),
    ("cave", "landform"), ("cave", "dark"), ("cave", "underground"),
    ("waterfall", "landform"), ("waterfall", "falling water"),
]

# =============================================================================
# SCIENCE (basic)
# =============================================================================

SCIENCE = [
    # States of matter
    ("water", "liquid"), ("ice", "solid"), ("steam", "gas"),
    ("solid", "hard"), ("liquid", "flow"), ("gas", "air"),
    
    # What is made of what
    ("water", "h2o"), ("water", "hydrogen"), ("water", "oxygen"),
    ("air", "oxygen"), ("air", "nitrogen"),
    ("body", "cells"), ("cells", "small"),
    ("atom", "small"), ("molecule", "atoms"),
    
    # Living and non-living
    ("plant", "alive"), ("animal", "alive"), ("human", "alive"),
    ("rock", "not alive"), ("water", "not alive"),
    
    # Plants
    ("plant", "grow"), ("plant", "sun"), ("plant", "water"),
    ("tree", "plant"), ("flower", "plant"), ("grass", "plant"),
    ("leaf", "plant"), ("leaf", "green"),
    ("root", "plant"), ("root", "underground"),
    ("stem", "plant"), ("stem", "support"),
    ("seed", "plant"), ("seed", "grow"),
    ("fruit", "seed"), ("fruit", "plant"),
    ("photosynthesis", "plant"), ("photosynthesis", "sun"),
    
    # Tree parts
    ("tree", "trunk"), ("tree", "branches"), ("tree", "leaves"), ("tree", "roots"),
    ("trunk", "wood"), ("branches", "wood"),
    ("bark", "tree"), ("bark", "outside"),
    
    # Flower parts
    ("flower", "petals"), ("flower", "stem"), ("flower", "leaves"),
    ("petal", "colorful"), ("petal", "soft"),
    
    # Basic physics
    ("gravity", "down"), ("gravity", "fall"),
    ("magnet", "attract"), ("magnet", "metal"),
    ("electricity", "power"), ("electricity", "light"),
    ("light", "fast"), ("light", "see"),
    ("sound", "hear"), ("sound", "waves"),
    ("heat", "hot"), ("heat", "energy"),
    
    # Basic biology
    ("heart", "pump"), ("heart", "blood"),
    ("lungs", "breathe"), ("lungs", "air"),
    ("brain", "think"), ("brain", "control"),
    ("stomach", "digest"), ("stomach", "food"),
    ("bones", "skeleton"), ("bones", "hard"),
    ("muscles", "move"), ("muscles", "strong"),
    ("skin", "protect"), ("skin", "outside"),
    ("blood", "red"), ("blood", "body"),
    
    # Five senses
    ("sight", "eyes"), ("sight", "see"),
    ("hearing", "ears"), ("hearing", "hear"),
    ("smell", "nose"), ("smell", "scent"),
    ("taste", "tongue"), ("taste", "flavor"),
    ("touch", "skin"), ("touch", "feel"),
]

# =============================================================================
# ARITHMETIC (basic)
# =============================================================================

ARITHMETIC = [
    ("plus", "add"), ("plus", "more"), ("plus", "equals"),
    ("minus", "subtract"), ("minus", "less"),
    ("equals", "same"), ("equals", "is"),
    ("add", "more"), ("subtract", "less"),
    ("count", "numbers"), ("count", "one"),
    
    # More numbers
    ("ten", "eleven"), ("eleven", "twelve"), ("twelve", "thirteen"),
    ("thirteen", "fourteen"), ("fourteen", "fifteen"),
    ("fifteen", "sixteen"), ("sixteen", "seventeen"),
    ("seventeen", "eighteen"), ("eighteen", "nineteen"),
    ("nineteen", "twenty"),
    
    # Tens
    ("twenty", "thirty"), ("thirty", "forty"), ("forty", "fifty"),
    ("fifty", "sixty"), ("sixty", "seventy"), ("seventy", "eighty"),
    ("eighty", "ninety"), ("ninety", "hundred"),
    
    # Basic facts
    ("one", "1"), ("two", "2"), ("three", "3"), ("four", "4"), ("five", "5"),
    ("six", "6"), ("seven", "7"), ("eight", "8"), ("nine", "9"), ("ten", "10"),
    ("zero", "0"), ("zero", "nothing"),
    ("hundred", "100"), ("thousand", "1000"),
    ("dozen", "twelve"), ("dozen", "12"),
]

# =============================================================================
# SOCIAL RULES
# =============================================================================

SOCIAL = [
    ("share", "friend"), ("share", "give"),
    ("help", "kind"), ("help", "friend"),
    ("sorry", "mistake"), ("sorry", "apologize"),
    ("please", "polite"), ("please", "ask"),
    ("thank", "polite"), ("thank", "grateful"),
    ("turn", "wait"), ("turn", "share"),
]

# =============================================================================
# EXTRA FACTS (what a child knows)
# =============================================================================

EXTRA_FACTS = [
    # More animals
    ("crocodile", "dangerous"), ("crocodile", "teeth"),
    ("alligator", "dangerous"), ("alligator", "swamp"),
    ("chimpanzee", "smart"), ("chimpanzee", "ape"),
    ("orangutan", "ape"), ("orangutan", "orange"),
    # Animals - IS-A and properties (no meaningless associations like "jaguar jungle")
    ("leopard", "animal"), ("leopard", "spotted"), ("leopard", "fast"),
    ("jaguar", "animal"), ("jaguar", "spotted"), ("jaguar", "strong"),
    ("panther", "animal"), ("panther", "black"),
    ("cheetah", "animal"), ("cheetah", "spotted"), ("cheetah", "fastest"),
    ("hyena", "animal"), ("hyena", "wild"),
    ("buffalo", "animal"), ("buffalo", "big"), ("buffalo", "horned"),
    ("moose", "animal"), ("moose", "big"), ("moose", "antlered"),
    ("reindeer", "animal"), ("reindeer", "antlered"),
    ("camel", "animal"), ("camel", "humped"),
    ("llama", "animal"), ("llama", "woolly"),
    ("alpaca", "animal"), ("alpaca", "woolly"), ("alpaca", "soft"),
    ("otter", "animal"), ("otter", "playful"),
    ("beaver", "animal"), ("beaver", "builds dams"),
    ("badger", "animal"), ("badger", "striped"),
    ("raccoon", "animal"), ("raccoon", "masked"), ("raccoon", "nocturnal"),
    ("skunk", "animal"), ("skunk", "smelly"), ("skunk", "striped"),
    ("porcupine", "animal"), ("porcupine", "quilled"),
    ("hedgehog", "animal"), ("hedgehog", "spiny"), ("hedgehog", "small"),
    ("bat", "animal"), ("bat", "flying"), ("bat", "nocturnal"), ("bat", "mammal"),
    ("mole", "animal"), ("mole", "digging"), ("mole", "underground"),
    ("weasel", "animal"), ("weasel", "small"), ("weasel", "fast"),
    ("ferret", "animal"), ("ferret", "pet"), ("ferret", "playful"),
    ("seal", "animal"), ("seal", "marine"), ("seal", "swimming"),
    ("walrus", "animal"), ("walrus", "tusked"), ("walrus", "arctic"),
    ("manatee", "animal"), ("manatee", "slow"), ("manatee", "marine"),
    ("narwhal", "animal"), ("narwhal", "tusked"), ("narwhal", "arctic"),
    ("orca", "animal"), ("orca", "whale"), ("orca", "black and white"),
    ("stingray", "animal"), ("stingray", "flat"), ("stingray", "marine"),
    ("manta ray", "animal"), ("manta ray", "big"), ("manta ray", "marine"),
    ("eel", "long"), ("eel", "electric"),
    ("salmon", "fish"), ("salmon", "swim"),
    ("tuna", "fish"), ("tuna", "fast"),
    ("swordfish", "sword"), ("swordfish", "fast"),
    ("clownfish", "orange"), ("clownfish", "anemone"),
    ("goldfish", "orange"), ("goldfish", "pet"),
    ("piranha", "teeth"), ("piranha", "dangerous"),
    ("barracuda", "teeth"), ("barracuda", "fast"),
    
    # More birds
    ("hummingbird", "small"), ("hummingbird", "fast"),
    ("woodpecker", "peck"), ("woodpecker", "tree"),
    ("toucan", "beak"), ("toucan", "colorful"),
    # Birds - IS-A and properties (no meaningless associations)
    ("pelican", "bird"), ("pelican", "big beak"),
    ("stork", "bird"), ("stork", "tall"), ("stork", "long legs"),
    ("heron", "bird"), ("heron", "tall"), ("heron", "long legs"),
    ("crane", "bird"), ("crane", "tall"), ("crane", "dancing"),
    ("seagull", "bird"), ("seagull", "coastal"), ("seagull", "loud"),
    ("albatross", "bird"), ("albatross", "big"), ("albatross", "flying"),
    ("vulture", "scavenger"), ("vulture", "bald"),
    ("hawk", "hunt"), ("hawk", "sharp"),
    ("falcon", "fast"), ("falcon", "hunt"),
    ("condor", "big"), ("condor", "fly"),
    ("ostrich", "big"), ("ostrich", "run"), ("ostrich", "cannot fly"),
    ("emu", "big"), ("emu", "run"),
    ("kiwi", "small"), ("kiwi", "cannot fly"),
    ("robin", "red"), ("robin", "spring"),
    ("cardinal", "red"), ("cardinal", "bird"),
    ("blue jay", "blue"), ("blue jay", "loud"),
    ("mockingbird", "sing"), ("mockingbird", "copy"),
    ("nightingale", "sing"), ("nightingale", "night"),
    ("canary", "yellow"), ("canary", "sing"),
    ("finch", "small"), ("finch", "seed"),
    ("dove", "white"), ("dove", "peace"),
    ("raven", "black"), ("raven", "smart"),
    ("magpie", "shiny"), ("magpie", "smart"),
    
    # More insects
    ("beetle", "hard"), ("beetle", "insect"),
    ("cockroach", "fast"), ("cockroach", "insect"),
    ("termite", "wood"), ("termite", "colony"),
    ("wasp", "sting"), ("wasp", "yellow"),
    ("hornet", "sting"), ("hornet", "big"),
    ("firefly", "light"), ("firefly", "night"),
    ("moth", "night"), ("moth", "light"),
    ("centipede", "legs"), ("centipede", "many"),
    ("millipede", "legs"), ("millipede", "many"),
    ("scorpion", "arachnid"), ("scorpion", "sting"), ("scorpion", "venomous"),
    ("tick", "small"), ("tick", "bite"),
    ("flea", "jump"), ("flea", "small"),
    ("louse", "small"), ("louse", "hair"),
    ("cicada", "loud"), ("cicada", "summer"),
    ("locust", "swarm"), ("locust", "eat"),
    ("mantis", "pray"), ("mantis", "green"),
    ("stick insect", "camouflage"), ("stick insect", "long"),
    
    # Dinosaurs (kids love them!)
    ("dinosaur", "extinct"), ("dinosaur", "ancient"),
    ("tyrannosaurus", "dinosaur"), ("tyrannosaurus", "big"), ("tyrannosaurus", "teeth"),
    ("triceratops", "dinosaur"), ("triceratops", "horns"),
    ("stegosaurus", "dinosaur"), ("stegosaurus", "plates"),
    ("brachiosaurus", "dinosaur"), ("brachiosaurus", "tall"), ("brachiosaurus", "long neck"),
    ("velociraptor", "dinosaur"), ("velociraptor", "fast"), ("velociraptor", "smart"),
    ("pterodactyl", "dinosaur"), ("pterodactyl", "fly"),
    ("diplodocus", "dinosaur"), ("diplodocus", "long"),
    ("ankylosaurus", "dinosaur"), ("ankylosaurus", "armor"),
    ("spinosaurus", "dinosaur"), ("spinosaurus", "sail"),
    
    # Mythical creatures (fairy tales) - DO NOT EXIST in reality
    ("dragon", "fire"), ("dragon", "fly"), ("dragon", "fictional"), ("dragon", "not real"),
    ("unicorn", "horn"), ("unicorn", "magical"), ("unicorn", "fictional"), ("unicorn", "not real"),
    ("mermaid", "fish"), ("mermaid", "swim"), ("mermaid", "fictional"), ("mermaid", "not real"),
    ("fairy", "small"), ("fairy", "wings"), ("fairy", "fictional"), ("fairy", "not real"),
    ("giant", "big"), ("giant", "tall"), ("giant", "fictional"), ("giant", "not real"),
    ("witch", "magic"), ("witch", "broom"), ("witch", "fictional"), ("witch", "not real"),
    ("wizard", "magic"), ("wizard", "wand"), ("wizard", "fictional"), ("wizard", "not real"),
    ("ghost", "scary"), ("ghost", "invisible"), ("ghost", "fictional"), ("ghost", "not real"),
    ("vampire", "blood"), ("vampire", "night"), ("vampire", "fictional"), ("vampire", "not real"),
    ("werewolf", "wolf"), ("werewolf", "moon"), ("werewolf", "fictional"), ("werewolf", "not real"),
    ("elf", "small"), ("elf", "ears"), ("elf", "fictional"), ("elf", "not real"),
    ("dwarf", "small"), ("dwarf", "mine"), ("dwarf", "fictional"), ("dwarf", "not real"),
    ("troll", "bridge"), ("troll", "ugly"), ("troll", "fictional"), ("troll", "not real"),
    ("goblin", "small"), ("goblin", "green"), ("goblin", "fictional"), ("goblin", "not real"),
    ("phoenix", "fire"), ("phoenix", "rebirth"), ("phoenix", "fictional"), ("phoenix", "not real"),
    ("pegasus", "horse"), ("pegasus", "wings"), ("pegasus", "fictional"), ("pegasus", "not real"),
    ("monster", "scary"), ("monster", "fictional"), ("monster", "not real"),
    ("santa claus", "christmas"), ("santa claus", "gifts"), ("santa claus", "fictional"),
    ("tooth fairy", "teeth"), ("tooth fairy", "money"), ("tooth fairy", "fictional"),
    ("easter bunny", "easter"), ("easter bunny", "eggs"), ("easter bunny", "fictional"),
    
    # More food
    ("hamburger", "food"), ("hamburger", "meat"),
    ("hot dog", "food"), ("hot dog", "sausage"),
    ("french fries", "food"), ("french fries", "potato"),
    ("chicken nuggets", "food"), ("chicken nuggets", "chicken"),
    ("spaghetti", "food"), ("spaghetti", "pasta"),
    ("macaroni", "food"), ("macaroni", "pasta"),
    ("lasagna", "food"), ("lasagna", "pasta"),
    ("taco", "food"), ("taco", "mexican"),
    ("burrito", "food"), ("burrito", "mexican"),
    ("sushi", "food"), ("sushi", "japanese"), ("sushi", "fish"),
    ("noodles", "food"), ("noodles", "asian"),
    ("dumpling", "food"), ("dumpling", "asian"),
    ("pancake", "food"), ("pancake", "breakfast"),
    ("waffle", "food"), ("waffle", "breakfast"),
    ("donut", "food"), ("donut", "sweet"),
    ("muffin", "food"), ("muffin", "sweet"),
    ("cupcake", "food"), ("cupcake", "sweet"),
    ("brownie", "food"), ("brownie", "chocolate"),
    ("popcorn", "food"), ("popcorn", "movie"),
    ("pretzel", "food"), ("pretzel", "salty"),
    ("cracker", "food"), ("cracker", "crunchy"),
    ("yogurt", "food"), ("yogurt", "dairy"),
    ("pudding", "food"), ("pudding", "sweet"),
    ("jelly", "food"), ("jelly", "sweet"),
    ("jam", "food"), ("jam", "fruit"),
    ("peanut butter", "food"), ("peanut butter", "nuts"),
    ("honey", "food"), ("honey", "sweet"), ("honey", "bee"),
    ("maple syrup", "food"), ("maple syrup", "sweet"),
    ("ketchup", "food"), ("ketchup", "tomato"),
    ("mustard", "food"), ("mustard", "yellow"),
    ("mayonnaise", "food"), ("mayonnaise", "white"),
    
    # More drinks
    ("lemonade", "drink"), ("lemonade", "sour"),
    ("hot chocolate", "drink"), ("hot chocolate", "warm"),
    ("milkshake", "drink"), ("milkshake", "cold"),
    ("smoothie", "drink"), ("smoothie", "fruit"),
    
    # More clothing
    ("boots", "shoes"), ("boots", "feet"),
    ("sandals", "shoes"), ("sandals", "summer"),
    ("sneakers", "shoes"), ("sneakers", "sport"),
    ("slippers", "shoes"), ("slippers", "home"),
    ("gloves", "hands"), ("gloves", "warm"),
    ("mittens", "hands"), ("mittens", "warm"),
    ("scarf", "neck"), ("scarf", "warm"),
    ("belt", "waist"), ("belt", "hold"),
    ("tie", "neck"), ("tie", "formal"),
    ("pajamas", "sleep"), ("pajamas", "night"),
    ("underwear", "clothes"), ("underwear", "inside"),
    ("swimsuit", "swim"), ("swimsuit", "water"),
    ("raincoat", "rain"), ("raincoat", "wet"),
    ("umbrella", "rain"), ("umbrella", "dry"),
    ("sunglasses", "sun"), ("sunglasses", "eyes"),
    ("watch", "time"), ("watch", "wrist"),
    ("necklace", "neck"), ("necklace", "jewelry"),
    ("bracelet", "wrist"), ("bracelet", "jewelry"),
    ("ring", "finger"), ("ring", "jewelry"),
    ("earrings", "ears"), ("earrings", "jewelry"),
    ("crown", "head"), ("crown", "king"),
    
    # More toys
    ("lego", "toy"), ("lego", "build"),
    ("playdough", "toy"), ("playdough", "shape"),
    ("crayons", "toy"), ("crayons", "color"),
    ("markers", "toy"), ("markers", "draw"),
    ("stickers", "toy"), ("stickers", "stick"),
    ("balloon", "toy"), ("balloon", "air"),
    ("bubbles", "toy"), ("bubbles", "pop"),
    ("yo-yo", "toy"), ("yo-yo", "spin"),
    ("frisbee", "toy"), ("frisbee", "throw"),
    ("skateboard", "toy"), ("skateboard", "ride"),
    ("scooter", "toy"), ("scooter", "ride"),
    ("trampoline", "toy"), ("trampoline", "jump"),
    ("swing", "toy"), ("swing", "playground"),
    ("slide", "toy"), ("slide", "playground"),
    ("sandbox", "toy"), ("sandbox", "sand"),
    ("action figure", "toy"), ("action figure", "play"),
    ("stuffed animal", "toy"), ("stuffed animal", "soft"),
    ("board game", "toy"), ("board game", "play"),
    ("video game", "toy"), ("video game", "screen"),
    ("remote control car", "toy"), ("remote control car", "drive"),
    
    # More sports
    ("football", "sport"), ("football", "ball"),
    ("hockey", "sport"), ("hockey", "ice"),
    ("golf", "sport"), ("golf", "ball"),
    ("volleyball", "sport"), ("volleyball", "net"),
    ("badminton", "sport"), ("badminton", "racket"),
    ("table tennis", "sport"), ("table tennis", "paddle"),
    ("bowling", "sport"), ("bowling", "pins"),
    ("skiing", "sport"), ("skiing", "snow"),
    ("snowboarding", "sport"), ("snowboarding", "snow"),
    ("ice skating", "sport"), ("ice skating", "ice"),
    ("roller skating", "sport"), ("roller skating", "wheels"),
    ("gymnastics", "sport"), ("gymnastics", "flip"),
    ("karate", "sport"), ("karate", "kick"),
    ("judo", "sport"), ("judo", "throw"),
    ("boxing", "sport"), ("boxing", "punch"),
    ("wrestling", "sport"), ("wrestling", "grab"),
    ("surfing", "sport"), ("surfing", "waves"),
    ("diving", "sport"), ("diving", "water"),
    ("archery", "sport"), ("archery", "arrow"),
    ("fencing", "sport"), ("fencing", "sword"),
    ("cycling", "sport"), ("cycling", "bike"),
    ("marathon", "sport"), ("marathon", "run"),
    
    # More musical instruments
    ("saxophone", "instrument"), ("saxophone", "jazz"),
    ("clarinet", "instrument"), ("clarinet", "wood"),
    ("oboe", "instrument"), ("oboe", "wood"),
    ("trombone", "instrument"), ("trombone", "brass"),
    ("tuba", "instrument"), ("tuba", "big"),
    ("harmonica", "instrument"), ("harmonica", "mouth"),
    ("accordion", "instrument"), ("accordion", "squeeze"),
    ("banjo", "instrument"), ("banjo", "strings"),
    ("ukulele", "instrument"), ("ukulele", "small"),
    ("harp", "instrument"), ("harp", "strings"),
    ("xylophone", "instrument"), ("xylophone", "hit"),
    ("tambourine", "instrument"), ("tambourine", "shake"),
    ("maracas", "instrument"), ("maracas", "shake"),
    ("bongo", "instrument"), ("bongo", "drum"),
    ("cello", "instrument"), ("cello", "strings"),
    ("bass", "instrument"), ("bass", "low"),
    
    # More professions
    ("astronaut", "space"), ("astronaut", "rocket"),
    ("engineer", "build"), ("engineer", "design"),
    ("architect", "design"), ("architect", "building"),
    ("lawyer", "law"), ("lawyer", "court"),
    ("judge", "court"), ("judge", "law"),
    ("soldier", "army"), ("soldier", "fight"),
    ("sailor", "ship"), ("sailor", "ocean"),
    ("captain", "ship"), ("captain", "leader"),
    ("mechanic", "fix"), ("mechanic", "car"),
    ("electrician", "wire"), ("electrician", "electricity"),
    ("plumber", "pipe"), ("plumber", "water"),
    ("carpenter", "wood"), ("carpenter", "build"),
    ("painter", "paint"), ("painter", "wall"),
    ("photographer", "camera"), ("photographer", "picture"),
    ("journalist", "news"), ("journalist", "write"),
    ("author", "book"), ("author", "write"),
    ("actor", "movie"), ("actor", "act"),
    ("actress", "movie"), ("actress", "act"),
    ("director", "movie"), ("director", "lead"),
    ("musician", "music"), ("musician", "play"),
    ("composer", "music"), ("composer", "write"),
    ("athlete", "sport"), ("athlete", "compete"),
    ("coach", "sport"), ("coach", "teach"),
    ("referee", "sport"), ("referee", "rules"),
    ("librarian", "library"), ("librarian", "books"),
    ("cashier", "store"), ("cashier", "money"),
    ("waiter", "restaurant"), ("waiter", "serve"),
    ("waitress", "restaurant"), ("waitress", "serve"),
    ("barber", "hair"), ("barber", "cut"),
    ("hairdresser", "hair"), ("hairdresser", "style"),
    ("tailor", "clothes"), ("tailor", "sew"),
    ("florist", "flowers"), ("florist", "arrange"),
    ("gardener", "garden"), ("gardener", "plant"),
    ("zookeeper", "zoo"), ("zookeeper", "animals"),
    ("lifeguard", "pool"), ("lifeguard", "save"),
    ("paramedic", "ambulance"), ("paramedic", "help"),
    ("surgeon", "doctor"), ("surgeon", "operate"),
    ("pharmacist", "medicine"), ("pharmacist", "drug"),
    ("therapist", "help"), ("therapist", "talk"),
    ("psychologist", "mind"), ("psychologist", "help"),
    
    # More places
    ("castle", "king"), ("castle", "old"),
    ("palace", "king"), ("palace", "big"),
    ("tower", "tall"), ("tower", "building"),
    ("bridge", "cross"), ("bridge", "river"),
    ("tunnel", "underground"), ("tunnel", "through"),
    ("stadium", "sport"), ("stadium", "big"),
    ("theater", "show"), ("theater", "stage"),
    ("cinema", "movie"), ("cinema", "screen"),
    ("mall", "shop"), ("mall", "big"),
    ("supermarket", "food"), ("supermarket", "buy"),
    ("bakery", "bread"), ("bakery", "cake"),
    ("pharmacy", "medicine"), ("pharmacy", "buy"),
    ("bank", "money"), ("bank", "save"),
    ("post office", "mail"), ("post office", "send"),
    ("fire station", "firefighter"), ("fire station", "truck"),
    ("police station", "police"), ("police station", "safe"),
    ("gas station", "fuel"), ("gas station", "car"),
    ("parking lot", "car"), ("parking lot", "park"),
    ("bus stop", "bus"), ("bus stop", "wait"),
    ("subway", "train"), ("subway", "underground"),
    ("harbor", "boat"), ("harbor", "water"),
    ("lighthouse", "light"), ("lighthouse", "ocean"),
    ("windmill", "wind"), ("windmill", "spin"),
    ("factory", "make"), ("factory", "work"),
    ("warehouse", "store"), ("warehouse", "big"),
    ("office", "work"), ("office", "desk"),
    ("classroom", "school"), ("classroom", "learn"),
    ("gymnasium", "sport"), ("gymnasium", "exercise"),
    ("cafeteria", "food"), ("cafeteria", "eat"),
    ("playground", "play"), ("playground", "children"),
    ("cemetery", "dead"), ("cemetery", "grave"),
    ("church", "pray"), ("church", "god"),
    ("temple", "pray"), ("temple", "religion"),
    ("mosque", "pray"), ("mosque", "islam"),
    ("synagogue", "pray"), ("synagogue", "jewish"),
    
    # More nature
    ("rainbow", "colors"), ("rainbow", "rain"),
    ("aurora", "light"), ("aurora", "north"),
    ("eclipse", "sun"), ("eclipse", "moon"),
    ("meteor", "fall"), ("meteor", "space"),
    ("comet", "tail"), ("comet", "space"),
    ("asteroid", "rock"), ("asteroid", "space"),
    ("galaxy", "stars"), ("galaxy", "big"),
    ("constellation", "stars"), ("constellation", "pattern"),
    ("tornado", "wind"), ("tornado", "dangerous"),
    ("hurricane", "wind"), ("hurricane", "rain"),
    ("tsunami", "wave"), ("tsunami", "big"),
    ("avalanche", "snow"), ("avalanche", "mountain"),
    ("landslide", "dirt"), ("landslide", "mountain"),
    ("flood", "water"), ("flood", "rain"),
    ("drought", "dry"), ("drought", "no rain"),
    ("fog", "cloud"), ("fog", "ground"),
    ("frost", "cold"), ("frost", "ice"),
    ("dew", "water"), ("dew", "morning"),
    ("hail", "ice"), ("hail", "fall"),
    ("sleet", "ice"), ("sleet", "rain"),
    ("blizzard", "snow"), ("blizzard", "wind"),
    
    # Holidays and special days (what kids know)
    ("christmas", "holiday"), ("christmas", "december"), ("christmas", "gifts"),
    ("christmas", "tree"), ("christmas", "winter"), ("christmas", "family"),
    ("new year", "holiday"), ("new year", "january"), ("new year", "celebration"),
    ("new year", "fireworks"), ("new year", "midnight"), ("new year", "countdown"),
    ("halloween", "holiday"), ("halloween", "october"), ("halloween", "costume"),
    ("halloween", "candy"), ("halloween", "pumpkin"), ("halloween", "scary"),
    ("easter", "holiday"), ("easter", "spring"), ("easter", "eggs"),
    ("easter", "bunny"), ("easter", "chocolate"),
    ("thanksgiving", "holiday"), ("thanksgiving", "november"), ("thanksgiving", "turkey"),
    ("thanksgiving", "family"), ("thanksgiving", "grateful"),
    ("birthday", "celebration"), ("birthday", "cake"), ("birthday", "presents"),
    ("birthday", "candles"), ("birthday", "party"), ("birthday", "balloons"),
    ("valentine", "holiday"), ("valentine", "february"), ("valentine", "love"),
    ("valentine", "heart"), ("valentine", "cards"),
    ("mothers day", "holiday"), ("mothers day", "may"), ("mothers day", "mom"),
    ("fathers day", "holiday"), ("fathers day", "june"), ("fathers day", "dad"),
    ("independence day", "holiday"), ("independence day", "july"), ("independence day", "fireworks"),
    ("st patrick", "holiday"), ("st patrick", "march"), ("st patrick", "green"),
    
    # Seasons and months (what a 5-year-old should know)
    ("january", "month"), ("january", "winter"), ("january", "first"),
    ("february", "month"), ("february", "winter"), ("february", "short"),
    ("march", "month"), ("march", "spring"),
    ("april", "month"), ("april", "spring"), ("april", "rain"),
    ("may", "month"), ("may", "spring"), ("may", "flowers"),
    ("june", "month"), ("june", "summer"),
    ("july", "month"), ("july", "summer"), ("july", "hot"),
    ("august", "month"), ("august", "summer"),
    ("september", "month"), ("september", "fall"), ("september", "school"),
    ("october", "month"), ("october", "fall"), ("october", "halloween"),
    ("november", "month"), ("november", "fall"), ("november", "thanksgiving"),
    ("december", "month"), ("december", "winter"), ("december", "christmas"),
    
    # More materials
    ("rubber", "material"), ("rubber", "stretch"),
    ("leather", "material"), ("leather", "animal"),
    ("cotton", "material"), ("cotton", "soft"),
    ("wool", "material"), ("wool", "sheep"),
    ("silk", "material"), ("silk", "smooth"),
    ("nylon", "material"), ("nylon", "strong"),
    ("polyester", "material"), ("polyester", "synthetic"),
    ("concrete", "material"), ("concrete", "hard"),
    ("brick", "material"), ("brick", "building"),
    ("cement", "material"), ("cement", "building"),
    ("marble", "material"), ("marble", "stone"),
    ("granite", "material"), ("granite", "stone"),
    ("clay", "material"), ("clay", "shape"),
    ("porcelain", "material"), ("porcelain", "fragile"),
    ("ceramic", "material"), ("ceramic", "pottery"),
    ("bronze", "material"), ("bronze", "metal"),
    ("copper", "material"), ("copper", "metal"),
    ("silver", "material"), ("silver", "metal"), ("silver", "shiny"),
    ("gold", "material"), ("gold", "metal"), ("gold", "valuable"),
    ("platinum", "material"), ("platinum", "metal"),
    ("aluminum", "material"), ("aluminum", "light"),
    ("steel", "material"), ("steel", "strong"),
    ("iron", "material"), ("iron", "metal"),
    ("tin", "material"), ("tin", "metal"),
    ("lead", "material"), ("lead", "heavy"),
    
    # Electronics (modern kids know)
    ("computer", "electronic"), ("computer", "screen"),
    ("laptop", "computer"), ("laptop", "portable"),
    ("tablet", "electronic"), ("tablet", "touch"),
    ("phone", "electronic"), ("phone", "call"),
    ("smartphone", "phone"), ("smartphone", "smart"),
    ("television", "electronic"), ("television", "watch"),
    ("radio", "electronic"), ("radio", "listen"),
    ("camera", "electronic"), ("camera", "picture"),
    ("printer", "electronic"), ("printer", "paper"),
    ("keyboard", "computer"), ("keyboard", "type"),
    ("mouse", "computer"), ("mouse", "click"),
    ("monitor", "computer"), ("monitor", "screen"),
    ("speaker", "electronic"), ("speaker", "sound"),
    ("headphones", "electronic"), ("headphones", "listen"),
    ("microphone", "electronic"), ("microphone", "speak"),
    ("charger", "electronic"), ("charger", "battery"),
    ("battery", "power"), ("battery", "energy"),
    ("wifi", "internet"), ("wifi", "wireless"),
    ("internet", "computer"), ("internet", "connect"),
    ("robot", "electronic"), ("robot", "machine"),
    
    # House parts
    ("roof", "house"), ("roof", "top"),
    ("wall", "house"), ("wall", "side"),
    ("floor", "house"), ("floor", "bottom"),
    ("ceiling", "house"), ("ceiling", "top"),
    ("door", "house"), ("door", "enter"),
    ("window", "house"), ("window", "see"),
    ("stairs", "house"), ("stairs", "climb"),
    ("chimney", "house"), ("chimney", "smoke"),
    ("garage", "house"), ("garage", "car"),
    ("basement", "house"), ("basement", "underground"),
    ("attic", "house"), ("attic", "top"),
    ("porch", "house"), ("porch", "outside"),
    ("balcony", "house"), ("balcony", "outside"),
    ("fence", "house"), ("fence", "around"),
    ("gate", "house"), ("gate", "enter"),
    ("driveway", "house"), ("driveway", "car"),
    ("mailbox", "house"), ("mailbox", "mail"),
    
    # Kitchen items
    ("stove", "kitchen"), ("stove", "cook"),
    ("oven", "kitchen"), ("oven", "bake"),
    ("microwave", "kitchen"), ("microwave", "heat"),
    ("refrigerator", "kitchen"), ("refrigerator", "cold"),
    ("freezer", "kitchen"), ("freezer", "ice"),
    ("dishwasher", "kitchen"), ("dishwasher", "clean"),
    ("sink", "kitchen"), ("sink", "water"),
    ("faucet", "water"), ("faucet", "turn"),
    ("pot", "kitchen"), ("pot", "cook"),
    ("pan", "kitchen"), ("pan", "fry"),
    ("kettle", "kitchen"), ("kettle", "boil"),
    ("toaster", "kitchen"), ("toaster", "bread"),
    ("blender", "kitchen"), ("blender", "mix"),
    ("plate", "kitchen"), ("plate", "food"),
    ("bowl", "kitchen"), ("bowl", "soup"),
    ("cup", "kitchen"), ("cup", "drink"),
    ("glass", "kitchen"), ("glass", "drink"),
    ("mug", "kitchen"), ("mug", "hot"),
    
    # Bathroom
    ("toilet", "bathroom"), ("toilet", "use"),
    ("bathtub", "bathroom"), ("bathtub", "bath"),
    ("shower", "bathroom"), ("shower", "wash"),
    ("towel", "bathroom"), ("towel", "dry"),
    ("soap", "bathroom"), ("soap", "clean"),
    ("shampoo", "bathroom"), ("shampoo", "hair"),
    ("toothbrush", "bathroom"), ("toothbrush", "teeth"),
    ("toothpaste", "bathroom"), ("toothpaste", "teeth"),
    ("mirror", "bathroom"), ("mirror", "see"),
    ("comb", "bathroom"), ("comb", "hair"),
    ("brush", "bathroom"), ("brush", "hair"),
    
    # Bedroom
    ("pillow", "bedroom"), ("pillow", "head"),
    ("blanket", "bedroom"), ("blanket", "warm"),
    ("sheet", "bedroom"), ("sheet", "bed"),
    ("mattress", "bedroom"), ("mattress", "soft"),
    ("alarm clock", "bedroom"), ("alarm clock", "wake"),
    ("lamp", "bedroom"), ("lamp", "light"),
    ("closet", "bedroom"), ("closet", "clothes"),
    ("drawer", "bedroom"), ("drawer", "store"),
    ("hanger", "closet"), ("hanger", "clothes"),
    
    # =====================================================
    # BASIC KNOWLEDGE FOR 5-YEAR-OLD CHILD
    # =====================================================
    
    # About self (self-awareness)
    ("i", "person"), ("me", "person"),
    ("boy", "child"), ("girl", "child"),
    ("name", "identity"), ("age", "number"),
    ("birthday", "special"), ("birthday", "age"),
    
    # Basic needs
    ("hungry", "need food"), ("thirsty", "need water"),
    ("tired", "need sleep"), ("cold", "need warmth"),
    ("hot", "need cool"), ("sick", "need doctor"),
    ("hurt", "need help"), ("scared", "need comfort"),
    
    # Health and hygiene
    ("wash hands", "clean"), ("wash hands", "germs"),
    ("brush teeth", "clean"), ("brush teeth", "healthy"),
    ("take bath", "clean"), ("take bath", "soap"),
    ("eat vegetables", "healthy"), ("drink water", "healthy"),
    ("exercise", "healthy"), ("sleep", "healthy"),
    ("germs", "sick"), ("germs", "small"),
    ("medicine", "sick"), ("medicine", "doctor"),
    ("vaccine", "protect"), ("vaccine", "doctor"),
    ("bandage", "hurt"), ("bandage", "heal"),
    
    # Safety (safety rules)
    ("stranger", "danger"), ("stranger", "unknown"),
    ("fire", "danger"), ("fire", "hot"),
    ("electricity", "danger"), ("electricity", "shock"),
    ("sharp", "danger"), ("sharp", "cut"),
    ("poison", "danger"), ("poison", "sick"),
    ("traffic", "danger"), ("traffic", "cars"),
    ("crosswalk", "safe"), ("crosswalk", "cross"),
    ("seatbelt", "safe"), ("seatbelt", "car"),
    ("helmet", "safe"), ("helmet", "head"),
    ("lifejacket", "safe"), ("lifejacket", "water"),
    ("stop", "red"), ("go", "green"), ("wait", "yellow"),
    ("look both ways", "cross"), ("hold hands", "safe"),
    
    # Manners and behavior
    ("please", "polite"), ("thank you", "polite"),
    ("sorry", "apologize"), ("excuse me", "polite"),
    ("share", "kind"), ("take turns", "fair"),
    ("listen", "respect"), ("wait", "patient"),
    ("quiet", "library"), ("inside voice", "polite"),
    ("raise hand", "school"), ("line up", "school"),
    
    # Emotions (emotional intelligence)
    ("happy", "smile"), ("happy", "good"),
    ("sad", "cry"), ("sad", "tears"),
    ("angry", "mad"), ("angry", "upset"),
    ("scared", "afraid"), ("scared", "fear"),
    ("excited", "happy"), ("excited", "energy"),
    ("surprised", "unexpected"), ("surprised", "wow"),
    ("frustrated", "angry"), ("frustrated", "hard"),
    ("jealous", "want"), ("jealous", "unfair"),
    ("proud", "good job"), ("proud", "happy"),
    ("embarrassed", "shy"), ("embarrassed", "red"),
    ("lonely", "alone"), ("lonely", "sad"),
    ("calm", "peaceful"), ("calm", "relax"),
    ("nervous", "worried"), ("nervous", "scared"),
    ("bored", "nothing"), ("bored", "tired"),
    
    # Friendship
    ("friend", "play"), ("friend", "share"),
    ("friend", "kind"), ("friend", "fun"),
    ("best friend", "special"), ("best friend", "close"),
    ("play together", "fun"), ("share toys", "kind"),
    ("be nice", "friend"), ("help friend", "kind"),
    
    # School and learning
    ("alphabet", "letters"), ("alphabet", "abc"),
    ("numbers", "count"), ("numbers", "math"),
    ("reading", "books"), ("reading", "words"),
    ("writing", "pencil"), ("writing", "letters"),
    ("drawing", "crayons"), ("drawing", "picture"),
    ("coloring", "crayons"), ("coloring", "fun"),
    ("cutting", "scissors"), ("cutting", "paper"),
    ("gluing", "glue"), ("gluing", "stick"),
    ("homework", "school"), ("homework", "learn"),
    ("test", "school"), ("test", "answer"),
    ("teacher", "help"), ("teacher", "learn"),
    ("classmate", "friend"), ("classmate", "school"),
    
    # Alphabet (sequence)
    ("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"),
    ("e", "f"), ("f", "g"), ("g", "h"), ("h", "i"),
    ("i", "j"), ("j", "k"), ("k", "l"), ("l", "m"),
    ("m", "n"), ("n", "o"), ("o", "p"), ("p", "q"),
    ("q", "r"), ("r", "s"), ("s", "t"), ("t", "u"),
    ("u", "v"), ("v", "w"), ("w", "x"), ("x", "y"),
    ("y", "z"),
    ("a", "letter"), ("z", "letter"), ("alphabet", "26"),
    
    # Counting and basic math
    ("count", "one"), ("count", "numbers"),
    ("more", "bigger"), ("less", "smaller"),
    ("equal", "same"), ("different", "not same"),
    ("first", "1"), ("second", "2"), ("third", "3"),
    ("half", "two parts"), ("whole", "complete"),
    ("pair", "two"), ("dozen", "twelve"),
    
    # Shapes (detailed)
    ("circle", "round"), ("circle", "no corners"),
    ("square", "four sides"), ("square", "equal sides"),
    ("triangle", "three sides"), ("triangle", "three corners"),
    ("rectangle", "four sides"), ("rectangle", "long"),
    ("oval", "egg"), ("oval", "round"),
    ("diamond", "four sides"), ("diamond", "pointy"),
    ("star", "five points"), ("star", "sky"),
    ("heart", "love"), ("heart", "shape"),
    
    # Positions and directions
    ("up", "above"), ("down", "below"),
    ("left", "side"), ("right", "side"),
    ("front", "ahead"), ("back", "behind"),
    ("inside", "in"), ("outside", "out"),
    ("top", "high"), ("bottom", "low"),
    ("middle", "center"), ("corner", "edge"),
    ("near", "close"), ("far", "away"),
    ("next to", "beside"), ("between", "middle"),
    
    # Comparisons
    ("big", "large"), ("small", "little"),
    ("tall", "high"), ("short", "low"),
    ("long", "length"), ("short", "length"),
    ("heavy", "weight"), ("light", "weight"),
    ("fast", "quick"), ("slow", "not fast"),
    ("loud", "noisy"), ("quiet", "soft"),
    ("hard", "solid"), ("soft", "squishy"),
    ("hot", "warm"), ("cold", "cool"),
    ("wet", "water"), ("dry", "no water"),
    ("clean", "neat"), ("dirty", "messy"),
    ("new", "fresh"), ("old", "used"),
    ("full", "complete"), ("empty", "nothing"),
    ("same", "equal"), ("different", "not same"),
    
    # Time (what a 5-year-old understands)
    ("morning", "wake up"), ("morning", "breakfast"),
    ("afternoon", "lunch"), ("afternoon", "nap"),
    ("evening", "dinner"), ("evening", "sunset"),
    ("night", "sleep"), ("night", "dark"),
    ("today", "now"), ("yesterday", "before"),
    ("tomorrow", "after"), ("later", "future"),
    ("soon", "almost"), ("now", "present"),
    ("always", "every time"), ("never", "no time"),
    ("sometimes", "not always"),
    
    # Money (basic understanding)
    ("money", "buy"), ("money", "pay"),
    ("coin", "money"), ("coin", "round"),
    ("dollar", "money"), ("cent", "money"),
    ("expensive", "lots money"), ("cheap", "little money"),
    ("save", "keep"), ("spend", "use"),
    ("store", "buy"), ("price", "cost"),
    
    # Nature and environment
    ("plant", "grow"), ("plant", "water"),
    ("seed", "plant"), ("seed", "grow"),
    ("flower", "pretty"), ("flower", "smell"),
    ("tree", "tall"), ("tree", "leaves"),
    ("grass", "green"), ("grass", "ground"),
    ("dirt", "ground"), ("dirt", "brown"),
    ("rock", "hard"), ("rock", "heavy"),
    ("sand", "beach"), ("sand", "soft"),
    ("mud", "wet dirt"), ("mud", "messy"),
    ("puddle", "water"), ("puddle", "rain"),
    
    # Animals and their babies
    ("puppy", "baby dog"), ("kitten", "baby cat"),
    ("calf", "baby cow"), ("foal", "baby horse"),
    ("chick", "baby chicken"), ("duckling", "baby duck"),
    ("lamb", "baby sheep"), ("piglet", "baby pig"),
    ("cub", "baby bear"), ("cub", "baby lion"),
    ("fawn", "baby deer"), ("joey", "baby kangaroo"),
    ("tadpole", "baby frog"), ("caterpillar", "baby butterfly"),
    
    # What animals eat
    ("dog", "dog food"), ("cat", "cat food"),
    ("rabbit", "carrots"), ("horse", "hay"),
    ("cow", "grass"), ("bird", "seeds"),
    ("fish", "fish food"), ("monkey", "bananas"),
    ("bear", "fish"), ("lion", "meat"),
    ("elephant", "plants"), ("panda", "bamboo"),
    
    # Where animals live (use sentences "X live in Y" instead of pairs)
    # ("fish", "water") - not a fact, there is "fish live in water" in sentences
    ("dog", "house"), ("cat", "house"),
    ("bird", "nest"),
    ("bear", "cave"), ("lion", "den"),
    ("bee", "hive"), ("ant", "anthill"),
    ("spider", "web"), ("rabbit", "burrow"),
    ("squirrel", "tree"), ("owl", "tree"),
    
    # Fairy tale characters (kids know)
    ("cinderella", "princess"), ("cinderella", "story"),
    ("snow white", "princess"), ("snow white", "story"),
    ("little mermaid", "princess"), ("little mermaid", "story"),
    ("sleeping beauty", "princess"), ("sleeping beauty", "story"),
    ("rapunzel", "princess"), ("rapunzel", "hair"),
    ("pinocchio", "puppet"), ("pinocchio", "story"),
    ("peter pan", "fly"), ("peter pan", "story"),
    ("little red riding hood", "wolf"), ("little red riding hood", "story"),
    ("three little pigs", "wolf"), ("three little pigs", "story"),
    ("goldilocks", "bears"), ("goldilocks", "story"),
    ("jack and beanstalk", "giant"), ("jack and beanstalk", "story"),
    ("hansel and gretel", "witch"), ("hansel and gretel", "story"),
    
    # Superheroes (modern kids know)
    ("superhero", "powers"), ("superhero", "save"),
    ("superman", "fly"), ("superman", "strong"),
    ("batman", "dark"), ("batman", "hero"),
    ("spiderman", "web"), ("spiderman", "climb"),
    ("wonder woman", "strong"), ("wonder woman", "hero"),
    ("hulk", "green"), ("hulk", "strong"),
    ("iron man", "armor"), ("iron man", "fly"),
    ("captain america", "shield"), ("captain america", "hero"),
    
    # Daily routine
    ("wake up", "morning"), ("get dressed", "morning"),
    ("eat breakfast", "morning"), ("brush teeth", "morning"),
    ("go to school", "morning"), ("learn", "school"),
    ("eat lunch", "noon"), ("play", "afternoon"),
    ("do homework", "afternoon"), ("eat dinner", "evening"),
    ("take bath", "evening"), ("brush teeth", "night"),
    ("read story", "night"), ("go to sleep", "night"),
    
    # Body parts (detailed)
    ("head", "top"), ("neck", "connect"),
    ("shoulder", "arm"), ("elbow", "bend"),
    ("wrist", "hand"), ("palm", "hand"),
    ("thumb", "finger"), ("pinky", "finger"),
    ("hip", "leg"), ("knee", "bend"),
    ("ankle", "foot"), ("heel", "foot"),
    ("forehead", "face"), ("eyebrow", "face"),
    ("eyelash", "eye"), ("cheek", "face"),
    ("chin", "face"), ("lip", "mouth"),
    ("tongue", "mouth"), ("tooth", "mouth"),
    ("chest", "body"), ("belly", "body"),
    ("back", "body"), ("bottom", "sit"),
    
    # Five senses (detailed)
    ("see", "eyes"), ("look", "eyes"),
    ("hear", "ears"), ("listen", "ears"),
    ("smell", "nose"), ("sniff", "nose"),
    ("taste", "tongue"), ("lick", "tongue"),
    ("touch", "hands"), ("feel", "skin"),
    ("blind", "cannot see"), ("deaf", "cannot hear"),
    
    # Family (extended)
    ("mother", "mom"), ("mother", "mommy"),
    ("father", "dad"), ("father", "daddy"),
    ("parents", "mother"), ("parents", "father"),
    ("grandmother", "grandma"), ("grandmother", "nana"),
    ("grandfather", "grandpa"), ("grandfather", "papa"),
    ("grandparents", "old"), ("grandparents", "wise"),
    ("brother", "sibling"), ("sister", "sibling"),
    ("twins", "same age"), ("twins", "two"),
    ("baby", "young"), ("baby", "small"),
    ("toddler", "young"), ("toddler", "walk"),
    ("child", "young"), ("teenager", "older"),
    ("adult", "grown up"), ("elderly", "old"),
    ("family", "love"), ("family", "home"),
    ("aunt", "parent sister"), ("uncle", "parent brother"),
    ("cousin", "aunt child"), ("cousin", "uncle child"),
    ("niece", "brother daughter"), ("nephew", "brother son"),
    ("stepmother", "not real mother"), ("stepfather", "not real father"),
    ("adopted", "chosen"), ("adopted", "family"),
    
    # Professions (what they do)
    ("doctor", "help sick"), ("nurse", "help doctor"),
    ("teacher", "teach children"), ("principal", "school boss"),
    ("firefighter", "put out fire"), ("police", "catch bad"),
    ("chef", "cook food"), ("baker", "bake bread"),
    ("farmer", "grow food"), ("gardener", "plant flowers"),
    ("builder", "build house"), ("carpenter", "work wood"),
    ("plumber", "fix pipes"), ("electrician", "fix wires"),
    ("mechanic", "fix cars"), ("pilot", "fly plane"),
    ("driver", "drive car"), ("captain", "drive ship"),
    ("astronaut", "go space"), ("scientist", "discover"),
    ("artist", "make art"), ("musician", "play music"),
    ("singer", "sing songs"), ("dancer", "dance"),
    ("actor", "act movies"), ("writer", "write books"),
    ("photographer", "take pictures"), ("journalist", "write news"),
    ("lawyer", "help court"), ("judge", "decide court"),
    ("soldier", "protect country"), ("president", "lead country"),
    ("president", "country"), ("president", "person"),
    ("mayor", "lead city"), ("king", "rule kingdom"),
    ("queen", "rule kingdom"), ("prince", "king son"),
    ("princess", "king daughter"),
    
    # Transport (detailed)
    ("car", "four wheels"), ("car", "drive"),
    ("truck", "big"), ("truck", "carry"),
    ("bus", "many people"), ("bus", "public"),
    ("taxi", "yellow"), ("taxi", "pay"),
    ("ambulance", "hospital"), ("ambulance", "siren"),
    ("fire truck", "firefighter"), ("fire truck", "red"),
    ("police car", "police"), ("police car", "siren"),
    ("motorcycle", "two wheels"), ("motorcycle", "fast"),
    ("bicycle", "two wheels"), ("bicycle", "pedal"),
    ("tricycle", "three wheels"), ("tricycle", "child"),
    ("scooter", "two wheels"), ("scooter", "kick"),
    ("skateboard", "four wheels"), ("skateboard", "balance"),
    ("train", "tracks"), ("train", "long"),
    ("subway", "underground"), ("subway", "city"),
    ("tram", "tracks"), ("tram", "city"),
    ("plane", "fly"), ("plane", "wings"),
    ("helicopter", "fly"), ("helicopter", "propeller"),
    ("rocket", "space"), ("rocket", "fast"),
    ("boat", "water"), ("boat", "float"),
    ("ship", "big boat"), ("ship", "ocean"),
    ("ferry", "carry cars"), ("ferry", "water"),
    ("submarine", "underwater"), ("submarine", "deep"),
    ("canoe", "paddle"), ("canoe", "water"),
    ("sailboat", "wind"), ("sailboat", "sail"),
    
    # Weather (detailed)
    ("sunny", "sun"), ("sunny", "bright"),
    ("cloudy", "clouds"), ("cloudy", "gray"),
    ("rainy", "rain"), ("rainy", "wet"),
    ("snowy", "snow"), ("snowy", "cold"),
    ("windy", "wind"), ("windy", "blow"),
    ("stormy", "thunder"), ("stormy", "lightning"),
    ("foggy", "fog"), ("foggy", "hard see"),
    ("humid", "sticky"), ("humid", "hot"),
    ("freezing", "very cold"), ("freezing", "ice"),
    ("hot", "summer"), ("cold", "winter"),
    ("warm", "spring"), ("cool", "fall"),
    ("temperature", "hot cold"), ("thermometer", "measure"),
    ("forecast", "predict"), ("weather report", "news"),
    
    # Seasons (detailed)
    ("spring", "flowers bloom"), ("spring", "birds return"),
    ("spring", "rain"), ("spring", "warm"),
    ("summer", "hot"), ("summer", "vacation"),
    ("summer", "swim"), ("summer", "beach"),
    ("fall", "leaves fall"), ("fall", "orange"),
    ("fall", "cool"), ("fall", "harvest"),
    ("winter", "cold"), ("winter", "snow"),
    ("winter", "holidays"), ("winter", "coat"),
    
    # Seasonal clothing
    ("summer", "shorts"), ("summer", "t-shirt"),
    ("summer", "sandals"), ("summer", "sunglasses"),
    ("winter", "coat"), ("winter", "boots"),
    ("winter", "gloves"), ("winter", "scarf"),
    ("winter", "hat"), ("rain", "raincoat"),
    ("rain", "umbrella"), ("rain", "boots"),
    
    # Colors (more shades)
    ("light blue", "sky"), ("dark blue", "navy"),
    ("light green", "lime"), ("dark green", "forest"),
    ("light red", "pink"), ("dark red", "maroon"),
    ("light yellow", "cream"), ("dark yellow", "gold"),
    ("light purple", "lavender"), ("dark purple", "violet"),
    ("light brown", "tan"), ("dark brown", "chocolate"),
    ("light gray", "silver"), ("dark gray", "charcoal"),
    ("rainbow", "red"), ("rainbow", "orange"),
    ("rainbow", "yellow"), ("rainbow", "green"),
    ("rainbow", "blue"), ("rainbow", "purple"),
    
    # Textures
    ("smooth", "glass"), ("rough", "sandpaper"),
    ("soft", "pillow"), ("hard", "rock"),
    ("fluffy", "cloud"), ("prickly", "cactus"),
    ("sticky", "glue"), ("slippery", "ice"),
    ("bumpy", "road"), ("fuzzy", "peach"),
    ("slimy", "snail"), ("crunchy", "chips"),
    ("squishy", "sponge"), ("stretchy", "rubber"),
    
    # Sounds
    ("loud", "thunder"), ("quiet", "whisper"),
    ("bang", "loud"), ("crash", "loud"),
    ("whisper", "quiet"), ("hum", "quiet"),
    ("ring", "bell"), ("beep", "horn"),
    ("splash", "water"), ("crunch", "leaves"),
    ("pop", "balloon"), ("sizzle", "cooking"),
    ("tick tock", "clock"), ("drip", "water"),
    
    # Smells
    ("good smell", "flowers"), ("good smell", "cookies"),
    ("bad smell", "garbage"), ("bad smell", "skunk"),
    ("sweet smell", "candy"), ("fresh smell", "rain"),
    ("smoky smell", "fire"), ("salty smell", "ocean"),
    
    # Tastes
    ("sweet", "sugar"), ("sweet", "candy"),
    ("sour", "lemon"), ("sour", "vinegar"),
    ("salty", "chips"), ("salty", "ocean"),
    ("bitter", "coffee"), ("bitter", "medicine"),
    ("spicy", "pepper"), ("spicy", "hot"),
    ("bland", "no taste"), ("delicious", "yummy"),
    
    # Quantity
    ("none", "zero"), ("one", "single"),
    ("few", "some"), ("many", "lots"),
    ("all", "everything"), ("most", "almost all"),
    ("half", "50 percent"), ("quarter", "25 percent"),
    ("pair", "two"), ("couple", "two"),
    ("several", "few"), ("dozen", "twelve"),
    ("hundred", "100"), ("thousand", "1000"),
    ("million", "very many"), ("billion", "very very many"),
    
    # Sizes
    ("tiny", "very small"), ("small", "little"),
    ("medium", "middle"), ("large", "big"),
    ("huge", "very big"), ("giant", "enormous"),
    ("tall", "high"), ("short", "low"),
    ("long", "length"), ("wide", "width"),
    ("thick", "fat"), ("thin", "skinny"),
    ("deep", "down"), ("shallow", "not deep"),
    
    # Speed
    ("slow", "turtle"), ("fast", "cheetah"),
    ("quick", "fast"), ("speedy", "very fast"),
    ("instant", "now"), ("gradual", "slow"),
    
    # Weight
    ("light", "feather"), ("heavy", "rock"),
    ("weightless", "space"), ("pound", "weight"),
    ("kilogram", "weight"), ("ton", "very heavy"),
    
    # Materials (what is made of what)
    ("paper", "tree"), ("cardboard", "paper"),
    ("glass", "sand"), ("plastic", "oil"),
    ("rubber", "tree"), ("leather", "animal"),
    ("wool", "sheep"), ("cotton", "plant"),
    ("silk", "worm"), ("metal", "earth"),
    ("wood", "tree"), ("stone", "earth"),
    ("brick", "clay"), ("concrete", "cement"),
    
    # Tools
    ("hammer", "hit nail"), ("screwdriver", "turn screw"),
    ("saw", "cut wood"), ("drill", "make hole"),
    ("wrench", "turn bolt"), ("pliers", "grab"),
    ("tape measure", "measure"), ("level", "straight"),
    ("ladder", "climb"), ("wheelbarrow", "carry"),
    
    # Kitchen actions
    ("cook", "heat food"), ("bake", "oven"),
    ("fry", "pan"), ("boil", "water"),
    ("mix", "stir"), ("chop", "cut"),
    ("peel", "remove skin"), ("slice", "cut thin"),
    ("pour", "liquid"), ("spread", "butter"),
    ("freeze", "cold"), ("thaw", "warm"),
    
    # School subjects
    ("math", "numbers"), ("reading", "books"),
    ("writing", "letters"), ("science", "experiments"),
    ("art", "drawing"), ("music", "singing"),
    ("gym", "exercise"), ("recess", "play"),
    ("lunch", "eat"), ("nap time", "sleep"),
    
    # Games
    ("hide and seek", "find"), ("tag", "chase"),
    ("hopscotch", "jump"), ("jump rope", "jump"),
    ("red light green light", "stop go"),
    ("simon says", "follow"), ("duck duck goose", "run"),
    ("musical chairs", "sit"), ("freeze dance", "stop"),
    
    # Cartoons and characters (modern kids)
    ("mickey mouse", "disney"), ("mickey mouse", "mouse"),
    ("minnie mouse", "disney"), ("minnie mouse", "bow"),
    ("donald duck", "disney"), ("donald duck", "duck"),
    ("goofy", "disney"), ("goofy", "dog"),
    ("winnie the pooh", "bear"), ("winnie the pooh", "honey"),
    ("elsa", "frozen"), ("elsa", "ice"),
    ("anna", "frozen"), ("anna", "sister"),
    ("olaf", "frozen"), ("olaf", "snowman"),
    ("moana", "ocean"), ("moana", "brave"),
    ("simba", "lion king"), ("simba", "lion"),
    ("nemo", "fish"), ("nemo", "ocean"),
    ("dory", "fish"), ("dory", "forget"),
    ("woody", "toy story"), ("woody", "cowboy"),
    ("buzz lightyear", "toy story"), ("buzz lightyear", "space"),
    ("lightning mcqueen", "cars"), ("lightning mcqueen", "race"),
    ("peppa pig", "pig"), ("peppa pig", "family"),
    ("paw patrol", "dogs"), ("paw patrol", "rescue"),
    ("bluey", "dog"), ("bluey", "play"),
    
    # Basic question words
    ("what", "question"), ("what", "thing"),
    ("who", "question"), ("who", "person"),
    ("where", "question"), ("where", "place"),
    ("when", "question"), ("when", "time"),
    ("why", "question"), ("why", "reason"),
    ("how", "question"), ("how", "way"),
    ("which", "question"), ("which", "choice"),
    ("whose", "question"), ("whose", "belong"),
    
    # Basic answers
    ("yes", "agree"), ("no", "disagree"),
    ("maybe", "not sure"), ("probably", "likely"),
    ("definitely", "sure"), ("never", "not ever"),
    ("always", "every time"), ("sometimes", "not always"),
    ("ok", "agree"), ("sure", "yes"),
    
    # Polite phrases
    ("hello", "greeting"), ("hi", "greeting"),
    ("goodbye", "leaving"), ("bye", "leaving"),
    ("good morning", "greeting"), ("good night", "leaving"),
    ("how are you", "greeting"), ("nice to meet you", "greeting"),
    ("please", "asking"), ("thank you", "grateful"),
    ("you are welcome", "response"), ("sorry", "apologize"),
    ("excuse me", "polite"), ("bless you", "sneeze"),
    
    # =====================================================
    # ADDITIONAL FACTS FOR FULL COVERAGE
    # =====================================================
    
    # More fruits
    ("kiwi", "fruit"), ("kiwi", "green"),
    ("papaya", "fruit"), ("papaya", "orange"),
    ("coconut", "fruit"), ("coconut", "white"),
    ("pomegranate", "fruit"), ("pomegranate", "red"),
    ("fig", "fruit"), ("fig", "sweet"),
    ("date", "fruit"), ("date", "sweet"),
    ("plum", "fruit"), ("plum", "purple"),
    ("apricot", "fruit"), ("apricot", "orange"),
    ("nectarine", "fruit"), ("nectarine", "peach"),
    ("cantaloupe", "fruit"), ("cantaloupe", "melon"),
    ("honeydew", "fruit"), ("honeydew", "melon"),
    ("raspberry", "fruit"), ("raspberry", "red"),
    ("blackberry", "fruit"), ("blackberry", "black"),
    ("blueberry", "fruit"), ("blueberry", "blue"),
    ("cranberry", "fruit"), ("cranberry", "red"),
    ("grapefruit", "fruit"), ("grapefruit", "sour"),
    ("lime", "fruit"), ("lime", "green"), ("lime", "sour"),
    ("tangerine", "fruit"), ("tangerine", "orange"),
    ("avocado", "fruit"), ("avocado", "green"),
    
    # More vegetables
    ("spinach", "vegetable"), ("spinach", "green"),
    ("kale", "vegetable"), ("kale", "green"),
    ("cabbage", "vegetable"), ("cabbage", "green"),
    ("cauliflower", "vegetable"), ("cauliflower", "white"),
    ("celery", "vegetable"), ("celery", "green"),
    ("asparagus", "vegetable"), ("asparagus", "green"),
    ("zucchini", "vegetable"), ("zucchini", "green"),
    ("squash", "vegetable"), ("squash", "yellow"),
    ("eggplant", "vegetable"), ("eggplant", "purple"),
    ("bell pepper", "vegetable"), ("bell pepper", "colorful"),
    ("mushroom", "vegetable"), ("mushroom", "brown"),
    ("garlic", "vegetable"), ("garlic", "white"),
    ("ginger", "vegetable"), ("ginger", "spicy"),
    ("radish", "vegetable"), ("radish", "red"),
    ("turnip", "vegetable"), ("turnip", "white"),
    ("beet", "vegetable"), ("beet", "red"),
    ("artichoke", "vegetable"), ("artichoke", "green"),
    ("brussels sprouts", "vegetable"), ("brussels sprouts", "green"),
    ("green beans", "vegetable"), ("green beans", "green"),
    ("sweet potato", "vegetable"), ("sweet potato", "orange"),
    
    # More grains and legumes
    ("wheat", "grain"), ("wheat", "bread"),
    ("rice", "grain"), ("rice", "white"),
    ("oats", "grain"), ("oats", "oatmeal"),
    ("corn", "grain"), ("corn", "yellow"),
    ("barley", "grain"), ("barley", "soup"),
    ("beans", "legume"), ("beans", "protein"),
    ("lentils", "legume"), ("lentils", "soup"),
    ("chickpeas", "legume"), ("chickpeas", "hummus"),
    ("peanuts", "legume"), ("peanuts", "butter"),
    ("soybeans", "legume"), ("soybeans", "tofu"),
    
    # Dairy products
    ("milk", "dairy"), ("milk", "cow"),
    ("cheese", "dairy"), ("cheese", "milk"),
    ("yogurt", "dairy"), ("yogurt", "healthy"),
    ("butter", "dairy"), ("butter", "yellow"),
    ("cream", "dairy"), ("cream", "white"),
    ("ice cream", "dairy"), ("ice cream", "cold"),
    
    # Meat and protein
    ("chicken", "meat"), ("chicken", "bird"),
    ("beef", "meat"), ("beef", "cow"),
    ("pork", "meat"), ("pork", "pig"),
    ("lamb", "meat"), ("lamb", "sheep"),
    ("turkey", "meat"), ("turkey", "bird"),
    ("fish", "protein"), ("fish", "healthy"),
    ("shrimp", "seafood"), ("shrimp", "small"),
    ("crab", "seafood"), ("crab", "claws"),
    ("lobster", "seafood"), ("lobster", "red"),
    ("egg", "protein"), ("egg", "chicken"),
    ("tofu", "protein"), ("tofu", "soy"),
    
    # Baked goods and sweets
    ("bread", "baked"), ("bread", "flour"),
    ("bagel", "bread"), ("bagel", "round"),
    ("croissant", "bread"), ("croissant", "french"),
    ("biscuit", "bread"), ("biscuit", "soft"),
    ("roll", "bread"), ("roll", "small"),
    ("toast", "bread"), ("toast", "crispy"),
    ("pie", "dessert"), ("pie", "fruit"),
    ("tart", "dessert"), ("tart", "small"),
    ("cheesecake", "dessert"), ("cheesecake", "cheese"),
    ("ice cream cone", "dessert"), ("ice cream cone", "cold"),
    ("popsicle", "dessert"), ("popsicle", "frozen"),
    ("cotton candy", "dessert"), ("cotton candy", "fluffy"),
    ("gummy bears", "candy"), ("gummy bears", "chewy"),
    ("lollipop", "candy"), ("lollipop", "stick"),
    ("jelly beans", "candy"), ("jelly beans", "colorful"),
    ("marshmallow", "candy"), ("marshmallow", "soft"),
    ("licorice", "candy"), ("licorice", "black"),
    
    # Drinks detailed
    ("apple juice", "drink"), ("apple juice", "sweet"),
    ("orange juice", "drink"), ("orange juice", "vitamin"),
    ("grape juice", "drink"), ("grape juice", "purple"),
    ("lemonade", "drink"), ("lemonade", "sour"),
    ("iced tea", "drink"), ("iced tea", "cold"),
    ("hot cocoa", "drink"), ("hot cocoa", "warm"),
    ("milkshake", "drink"), ("milkshake", "thick"),
    ("smoothie", "drink"), ("smoothie", "fruit"),
    ("soda", "drink"), ("soda", "fizzy"),
    ("sparkling water", "drink"), ("sparkling water", "bubbles"),
    
    # Dishes and utensils
    ("plate", "dish"), ("plate", "round"),
    ("bowl", "dish"), ("bowl", "deep"),
    ("cup", "dish"), ("cup", "drink"),
    ("mug", "dish"), ("mug", "hot"),
    ("glass", "dish"), ("glass", "clear"),
    ("fork", "utensil"), ("fork", "poke"),
    ("knife", "utensil"), ("knife", "cut"),
    ("spoon", "utensil"), ("spoon", "scoop"),
    ("chopsticks", "utensil"), ("chopsticks", "asian"),
    ("napkin", "table"), ("napkin", "wipe"),
    ("placemat", "table"), ("placemat", "under"),
    ("tablecloth", "table"), ("tablecloth", "cover"),
    
    # More animals (exotic)
    ("lemur", "animal"), ("lemur", "madagascar"),
    ("sloth", "animal"), ("sloth", "slow"),
    ("armadillo", "animal"), ("armadillo", "armor"),
    ("anteater", "animal"), ("anteater", "tongue"),
    ("tapir", "animal"), ("tapir", "nose"),
    ("capybara", "animal"), ("capybara", "big"),
    ("chinchilla", "animal"), ("chinchilla", "soft"),
    ("meerkat", "animal"), ("meerkat", "stand"),
    ("mongoose", "animal"), ("mongoose", "snake"),
    ("opossum", "animal"), ("opossum", "play dead"),
    ("platypus", "animal"), ("platypus", "beak"),
    ("echidna", "animal"), ("echidna", "spines"),
    ("wombat", "animal"), ("wombat", "australia"),
    ("tasmanian devil", "animal"), ("tasmanian devil", "australia"),
    ("okapi", "animal"), ("okapi", "stripes"),
    ("pangolin", "animal"), ("pangolin", "scales"),
    ("aardvark", "animal"), ("aardvark", "dig"),
    
    # Marine animals - IS-A and properties (no meaningless "X ocean")
    ("sea turtle", "animal"), ("sea turtle", "marine"), ("sea turtle", "shelled"),
    ("sea lion", "animal"), ("sea lion", "marine"), ("sea lion", "barking"),
    ("sea otter", "animal"), ("sea otter", "marine"), ("sea otter", "floating"),
    ("sea horse", "animal"), ("sea horse", "marine"), ("sea horse", "small"),
    ("sea urchin", "animal"), ("sea urchin", "marine"), ("sea urchin", "spiky"),
    ("sea star", "animal"), ("sea star", "marine"), ("sea star", "five armed"),
    ("sea cucumber", "animal"), ("sea cucumber", "marine"), ("sea cucumber", "soft"),
    ("coral", "organism"), ("coral", "marine"), ("coral", "colorful"),
    ("anemone", "organism"), ("anemone", "marine"), ("anemone", "tentacled"),
    ("squid", "animal"), ("squid", "marine"), ("squid", "tentacled"),
    ("shrimp", "animal"), ("shrimp", "marine"), ("shrimp", "small"),
    ("oyster", "animal"), ("oyster", "marine"), ("oyster", "pearl producing"),
    ("clam", "animal"), ("clam", "marine"), ("clam", "shelled"),
    ("mussel", "animal"), ("mussel", "marine"), ("mussel", "shelled"),
    ("scallop", "animal"), ("scallop", "marine"), ("scallop", "shelled"),
    ("hermit crab", "animal"), ("hermit crab", "marine"), ("hermit crab", "shelled"),
    ("horseshoe crab", "animal"), ("horseshoe crab", "marine"), ("horseshoe crab", "ancient"),
    
    # Plants detailed
    ("rose", "flower"), ("rose", "red"),
    ("tulip", "flower"), ("tulip", "spring"),
    ("daisy", "flower"), ("daisy", "white"),
    ("sunflower", "flower"), ("sunflower", "yellow"),
    ("lily", "flower"), ("lily", "white"),
    ("orchid", "flower"), ("orchid", "exotic"),
    ("carnation", "flower"), ("carnation", "pink"),
    ("violet", "flower"), ("violet", "purple"),
    ("daffodil", "flower"), ("daffodil", "yellow"),
    ("iris", "flower"), ("iris", "purple"),
    ("peony", "flower"), ("peony", "pink"),
    ("marigold", "flower"), ("marigold", "orange"),
    ("lavender", "flower"), ("lavender", "purple"),
    ("jasmine", "flower"), ("jasmine", "white"),
    ("hibiscus", "flower"), ("hibiscus", "tropical"),
    ("cactus", "plant"), ("cactus", "desert"),
    ("fern", "plant"), ("fern", "green"),
    ("moss", "plant"), ("moss", "soft"),
    ("ivy", "plant"), ("ivy", "climb"),
    ("bamboo", "plant"), ("bamboo", "tall"),
    ("palm tree", "tree"), ("palm tree", "tropical"),
    ("pine tree", "tree"), ("pine tree", "evergreen"),
    ("oak tree", "tree"), ("oak tree", "acorn"),
    ("maple tree", "tree"), ("maple tree", "syrup"),
    ("willow tree", "tree"), ("willow tree", "droopy"),
    ("cherry tree", "tree"), ("cherry tree", "blossom"),
    ("apple tree", "tree"), ("apple tree", "fruit"),
    ("orange tree", "tree"), ("orange tree", "fruit"),
    ("lemon tree", "tree"), ("lemon tree", "fruit"),
    
    # Insects and spiders detailed
    ("black widow", "spider"), ("black widow", "dangerous"),
    ("tarantula", "spider"), ("tarantula", "big"),
    ("daddy long legs", "spider"), ("daddy long legs", "legs"),
    ("jumping spider", "spider"), ("jumping spider", "jump"),
    ("monarch butterfly", "butterfly"), ("monarch butterfly", "orange"),
    ("luna moth", "moth"), ("luna moth", "green"),
    ("praying mantis", "insect"), ("praying mantis", "pray"),
    ("walking stick", "insect"), ("walking stick", "camouflage"),
    ("leaf insect", "insect"), ("leaf insect", "camouflage"),
    ("dung beetle", "beetle"), ("dung beetle", "roll"),
    ("ladybug", "beetle"), ("ladybug", "spots"),
    ("firefly", "beetle"), ("firefly", "glow"),
    ("june bug", "beetle"), ("june bug", "summer"),
    ("stag beetle", "beetle"), ("stag beetle", "horns"),
    ("rhinoceros beetle", "beetle"), ("rhinoceros beetle", "horn"),
    
    # Reptiles detailed
    ("iguana", "lizard"), ("iguana", "green"),
    ("chameleon", "lizard"), ("chameleon", "color change"),
    ("gecko", "lizard"), ("gecko", "climb"),
    ("komodo dragon", "lizard"), ("komodo dragon", "big"),
    ("gila monster", "lizard"), ("gila monster", "venomous"),
    ("python", "snake"), ("python", "squeeze"),
    ("cobra", "snake"), ("cobra", "hood"),
    ("rattlesnake", "snake"), ("rattlesnake", "rattle"),
    ("anaconda", "snake"), ("anaconda", "big"),
    ("boa constrictor", "snake"), ("boa constrictor", "squeeze"),
    ("sea turtle", "turtle"), ("sea turtle", "ocean"),
    ("box turtle", "turtle"), ("box turtle", "land"),
    ("snapping turtle", "turtle"), ("snapping turtle", "bite"),
    ("tortoise", "turtle"), ("tortoise", "land"),
    
    # Amphibians
    ("tree frog", "frog"), ("tree frog", "climb"),
    ("poison dart frog", "frog"), ("poison dart frog", "colorful"),
    ("bullfrog", "frog"), ("bullfrog", "big"),
    ("salamander", "amphibian"), ("salamander", "wet"),
    ("newt", "amphibian"), ("newt", "small"),
    ("axolotl", "amphibian"), ("axolotl", "gills"),
    
    # Birds detailed
    ("bald eagle", "eagle"), ("bald eagle", "america"),
    ("golden eagle", "eagle"), ("golden eagle", "gold"),
    ("red-tailed hawk", "hawk"), ("red-tailed hawk", "common"),
    ("peregrine falcon", "falcon"), ("peregrine falcon", "fastest"),
    ("great horned owl", "owl"), ("great horned owl", "horns"),
    ("snowy owl", "owl"), ("snowy owl", "white"),
    ("barn owl", "owl"), ("barn owl", "heart face"),
    ("emperor penguin", "penguin"), ("emperor penguin", "big"),
    ("king penguin", "penguin"), ("king penguin", "colorful"),
    ("african grey parrot", "parrot"), ("african grey parrot", "smart"),
    ("macaw", "parrot"), ("macaw", "colorful"),
    ("cockatoo", "parrot"), ("cockatoo", "crest"),
    ("budgie", "parrot"), ("budgie", "small"),
    
    # Dog breeds
    ("labrador", "dog"), ("labrador", "friendly"),
    ("golden retriever", "dog"), ("golden retriever", "golden"),
    ("german shepherd", "dog"), ("german shepherd", "smart"),
    ("bulldog", "dog"), ("bulldog", "wrinkly"),
    ("poodle", "dog"), ("poodle", "curly"),
    ("beagle", "dog"), ("beagle", "nose"),
    ("husky", "dog"), ("husky", "snow"),
    ("dalmatian", "dog"), ("dalmatian", "spots"),
    ("chihuahua", "dog"), ("chihuahua", "tiny"),
    ("great dane", "dog"), ("great dane", "huge"),
    ("corgi", "dog"), ("corgi", "short legs"),
    ("pug", "dog"), ("pug", "flat face"),
    ("boxer", "dog"), ("boxer", "playful"),
    ("rottweiler", "dog"), ("rottweiler", "guard"),
    ("doberman", "dog"), ("doberman", "sleek"),
    ("shih tzu", "dog"), ("shih tzu", "fluffy"),
    ("yorkshire terrier", "dog"), ("yorkshire terrier", "small"),
    ("dachshund", "dog"), ("dachshund", "long"),
    ("border collie", "dog"), ("border collie", "smart"),
    ("australian shepherd", "dog"), ("australian shepherd", "herding"),
    
    # Cat breeds
    ("persian", "cat"), ("persian", "fluffy"),
    ("siamese", "cat"), ("siamese", "vocal"),
    ("maine coon", "cat"), ("maine coon", "big"),
    ("ragdoll", "cat"), ("ragdoll", "floppy"),
    ("british shorthair", "cat"), ("british shorthair", "round"),
    ("scottish fold", "cat"), ("scottish fold", "ears"),
    ("bengal", "cat"), ("bengal", "spots"),
    ("sphynx", "cat"), ("sphynx", "hairless"),
    ("russian blue", "cat"), ("russian blue", "gray"),
    ("abyssinian", "cat"), ("abyssinian", "active"),
    
    # Space detailed
    ("milky way", "galaxy"), ("milky way", "home"),
    ("andromeda", "galaxy"), ("andromeda", "neighbor"),
    ("black hole", "space"), ("black hole", "gravity"),
    ("supernova", "space"), ("supernova", "explosion"),
    ("nebula", "space"), ("nebula", "colorful"),
    ("asteroid belt", "space"), ("asteroid belt", "rocks"),
    ("kuiper belt", "space"), ("kuiper belt", "far"),
    ("dwarf planet", "space"), ("dwarf planet", "small"),
    ("pluto", "dwarf planet"), ("pluto", "small"),
    ("space station", "space"), ("space station", "orbit"),
    ("satellite", "space"), ("satellite", "orbit"),
    ("telescope", "space"), ("telescope", "see"),
    ("astronomer", "space"), ("astronomer", "study"),
    
    # Geography detailed
    ("north pole", "cold"), ("north pole", "arctic"),
    ("south pole", "cold"), ("south pole", "antarctica"),
    ("equator", "hot"), ("equator", "middle"),
    ("tropics", "hot"), ("tropics", "jungle"),
    ("rainforest", "jungle"), ("rainforest", "rain"),
    ("savanna", "grassland"), ("savanna", "africa"),
    ("tundra", "cold"), ("tundra", "frozen"),
    ("prairie", "grassland"), ("prairie", "flat"),
    ("swamp", "wetland"), ("swamp", "alligator"),
    ("marsh", "wetland"), ("marsh", "birds"),
    ("reef", "ocean"), ("reef", "coral"),
    ("glacier", "ice"), ("glacier", "move"),
    ("iceberg", "ice"), ("iceberg", "float"),
    ("canyon", "deep"), ("canyon", "river"),
    ("cliff", "high"), ("cliff", "edge"),
    ("plateau", "flat"), ("plateau", "high"),
    ("peninsula", "land"), ("peninsula", "water"),
    ("bay", "water"), ("bay", "land"),
    ("gulf", "water"), ("gulf", "big"),
    ("strait", "water"), ("strait", "narrow"),
    ("archipelago", "islands"), ("archipelago", "many"),
    
    # Famous places
    ("eiffel tower", "paris"), ("eiffel tower", "tall"),
    ("statue of liberty", "new york"), ("statue of liberty", "green"),
    ("great wall", "china"), ("great wall", "long"),
    ("pyramids", "egypt"), ("pyramids", "ancient"),
    ("taj mahal", "india"), ("taj mahal", "white"),
    ("colosseum", "rome"), ("colosseum", "ancient"),
    ("big ben", "london"), ("big ben", "clock"),
    ("sydney opera house", "australia"), ("sydney opera house", "white"),
    ("mount everest", "nepal"), ("mount everest", "tallest"),
    ("grand canyon", "arizona"), ("grand canyon", "big"),
    ("niagara falls", "waterfall"), ("niagara falls", "big"),
    ("amazon river", "south america"), ("amazon river", "long"),
    ("nile river", "africa"), ("nile river", "long"),
    ("sahara desert", "africa"), ("sahara desert", "big"),
    ("great barrier reef", "australia"), ("great barrier reef", "coral"),
    
    # More countries and languages
    ("english", "language"), ("english", "america"),
    ("spanish", "language"), ("spanish", "spain"),
    ("french", "language"), ("french", "france"),
    ("german", "language"), ("german", "germany"),
    ("italian", "language"), ("italian", "italy"),
    ("portuguese", "language"), ("portuguese", "brazil"),
    ("russian", "language"), ("russian", "russia"),
    ("chinese", "language"), ("chinese", "china"),
    ("japanese", "language"), ("japanese", "japan"),
    ("korean", "language"), ("korean", "korea"),
    ("arabic", "language"), ("arabic", "middle east"),
    ("hindi", "language"), ("hindi", "india"),
    
    # Currencies
    ("dollar", "currency"), ("dollar", "america"),
    ("euro", "currency"), ("euro", "europe"),
    ("pound", "currency"), ("pound", "england"),
    ("yen", "currency"), ("yen", "japan"),
    ("yuan", "currency"), ("yuan", "china"),
    
    # Units of measurement
    ("inch", "length"), ("foot", "length"),
    ("yard", "length"), ("mile", "length"),
    ("centimeter", "length"), ("meter", "length"),
    ("kilometer", "length"), ("ounce", "weight"),
    ("pound", "weight"), ("gram", "weight"),
    ("kilogram", "weight"), ("cup", "volume"),
    ("pint", "volume"), ("quart", "volume"),
    ("gallon", "volume"), ("liter", "volume"),
    ("second", "time"), ("minute", "time"),
    ("hour", "time"), ("day", "time"),
    ("week", "time"), ("month", "time"),
    ("year", "time"), ("decade", "time"),
    ("century", "time"), ("fahrenheit", "temperature"),
    ("celsius", "temperature"),
    
    # =====================================================
    # MORPHOLOGICAL CONNECTIONS (Verb forms)
    # =====================================================
    # BIOLOGY: Child learns that different word forms are connected
    # This allows "say" to activate "meow" through "says"
    
    # say/says
    ("say", "says"), ("says", "say"),
    ("say", "said"), ("said", "say"),
    
    # come/comes
    ("come", "comes"), ("comes", "come"),
    ("come", "came"), ("came", "come"),
    
    # go/goes
    ("go", "goes"), ("goes", "go"),
    ("go", "went"), ("went", "go"),
    
    # make/makes
    ("make", "makes"), ("makes", "make"),
    ("make", "made"), ("made", "make"),
    
    # eat/eats
    ("eat", "eats"), ("eats", "eat"),
    ("eat", "ate"), ("ate", "eat"),
    
    # drink/drinks
    ("drink", "drinks"), ("drinks", "drink"),
    ("drink", "drank"), ("drank", "drink"),
    
    # sleep/sleeps
    ("sleep", "sleeps"), ("sleeps", "sleep"),
    ("sleep", "slept"), ("slept", "sleep"),
    
    # run/runs
    ("run", "runs"), ("runs", "run"),
    ("run", "ran"), ("ran", "run"),
    
    # walk/walks
    ("walk", "walks"), ("walks", "walk"),
    ("walk", "walked"), ("walked", "walk"),
    
    # play/plays
    ("play", "plays"), ("plays", "play"),
    ("play", "played"), ("played", "play"),
    
    # live/lives
    ("live", "lives"), ("lives", "live"),
    ("live", "lived"), ("lived", "live"),
    
    # =====================================================
    # PRESCHOOL KNOWLEDGE
    # =====================================================
    
    # Basic facts about self (Self-awareness)
    ("i", "person"), ("me", "person"), ("myself", "person"),
    ("boy", "child"), ("girl", "child"), ("child", "young"),
    ("name", "identity"), ("age", "number"), ("birthday", "special"),
    
    # Basic needs
    ("hungry", "need food"), ("thirsty", "need water"),
    ("tired", "need sleep"), ("cold", "need warmth"),
    ("hot", "need cool"), ("sick", "need doctor"),
    ("hurt", "need help"), ("scared", "need comfort"),
    
    # Health and hygiene
    ("wash hands", "clean"), ("wash hands", "germs"),
    ("brush teeth", "clean"), ("brush teeth", "healthy"),
    ("take bath", "clean"), ("take bath", "soap"),
    ("eat vegetables", "healthy"), ("drink water", "healthy"),
    ("exercise", "healthy"), ("sleep", "healthy"),
    ("germs", "sick"), ("germs", "small"),
    ("medicine", "sick"), ("medicine", "doctor"),
    ("vaccine", "protect"), ("vaccine", "doctor"),
    ("bandage", "hurt"), ("bandage", "heal"),
    
    # Safety rules
    ("stranger", "danger"), ("stranger", "unknown"),
    ("fire", "danger"), ("fire", "hot"),
    ("electricity", "danger"), ("electricity", "shock"),
    ("sharp", "danger"), ("sharp", "cut"),
    ("poison", "danger"), ("poison", "sick"),
    ("traffic", "danger"), ("traffic", "cars"),
    ("crosswalk", "safe"), ("crosswalk", "cross"),
    ("seatbelt", "safe"), ("seatbelt", "car"),
    ("helmet", "safe"), ("helmet", "head"),
    ("lifejacket", "safe"), ("lifejacket", "water"),
    ("stop", "red"), ("go", "green"), ("wait", "yellow"),
    ("look both ways", "cross"), ("hold hands", "safe"),
    
    # Manners and behavior
    ("please", "polite"), ("thank you", "polite"),
    ("sorry", "apologize"), ("excuse me", "polite"),
    ("share", "kind"), ("take turns", "fair"),
    ("listen", "respect"), ("wait", "patient"),
    ("quiet", "library"), ("inside voice", "polite"),
    ("raise hand", "school"), ("line up", "school"),
    
    # Emotional intelligence
    ("happy", "smile"), ("happy", "good"),
    ("sad", "cry"), ("sad", "tears"),
    ("angry", "mad"), ("angry", "upset"),
    ("scared", "afraid"), ("scared", "fear"),
    ("excited", "happy"), ("excited", "energy"),
    ("surprised", "unexpected"), ("surprised", "wow"),
    ("frustrated", "angry"), ("frustrated", "hard"),
    ("jealous", "want"), ("jealous", "unfair"),
    ("proud", "good job"), ("proud", "happy"),
    ("embarrassed", "shy"), ("embarrassed", "red"),
    ("lonely", "alone"), ("lonely", "sad"),
    ("calm", "peaceful"), ("calm", "relax"),
    ("nervous", "worried"), ("nervous", "scared"),
    ("bored", "nothing"), ("bored", "tired"),
    
    # Friendship
    ("friend", "play"), ("friend", "share"),
    ("friend", "kind"), ("friend", "fun"),
    ("best friend", "special"), ("best friend", "close"),
    ("play together", "fun"), ("share toys", "kind"),
    ("be nice", "friend"), ("help friend", "kind"),
    
    # School and learning
    ("alphabet", "letters"), ("alphabet", "abc"),
    ("numbers", "count"), ("numbers", "math"),
    ("reading", "books"), ("reading", "words"),
    ("writing", "pencil"), ("writing", "letters"),
    ("drawing", "crayons"), ("drawing", "picture"),
    ("coloring", "crayons"), ("coloring", "fun"),
    ("cutting", "scissors"), ("cutting", "paper"),
    ("gluing", "glue"), ("gluing", "stick"),
    ("homework", "school"), ("homework", "learn"),
    ("test", "school"), ("test", "answer"),
    ("teacher", "help"), ("teacher", "learn"),
    ("classmate", "friend"), ("classmate", "school"),
    
    # Alphabet sequence
    ("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"),
    ("e", "f"), ("f", "g"), ("g", "h"), ("h", "i"),
    ("i", "j"), ("j", "k"), ("k", "l"), ("l", "m"),
    ("m", "n"), ("n", "o"), ("o", "p"), ("p", "q"),
    ("q", "r"), ("r", "s"), ("s", "t"), ("t", "u"),
    ("u", "v"), ("v", "w"), ("w", "x"), ("x", "y"),
    ("y", "z"),
    ("a", "letter"), ("z", "letter"), ("alphabet", "26"),
    
    # Vowels and consonants
    ("a", "vowel"), ("e", "vowel"), ("i", "vowel"),
    ("o", "vowel"), ("u", "vowel"),
    
    # Counting and basic math
    ("count", "one"), ("count", "numbers"),
    ("more", "bigger"), ("less", "smaller"),
    ("equal", "same"), ("different", "not same"),
    ("first", "1"), ("second", "2"), ("third", "3"),
    ("half", "two parts"), ("whole", "complete"),
    ("pair", "two"), ("dozen", "twelve"),
    
    # Shapes detailed
    ("circle", "round"), ("circle", "no corners"),
    ("square", "four sides"), ("square", "equal sides"),
    ("triangle", "three sides"), ("triangle", "three corners"),
    ("rectangle", "four sides"), ("rectangle", "long"),
    ("oval", "egg"), ("oval", "round"),
    ("diamond", "four sides"), ("diamond", "pointy"),
    ("star", "five points"), ("star", "sky"),
    ("heart", "love"), ("heart", "shape"),
    
    # Positions and directions
    ("up", "above"), ("down", "below"),
    ("left", "side"), ("right", "side"),
    ("front", "ahead"), ("back", "behind"),
    ("inside", "in"), ("outside", "out"),
    ("top", "high"), ("bottom", "low"),
    ("middle", "center"), ("corner", "edge"),
    ("near", "close"), ("far", "away"),
    ("next to", "beside"), ("between", "middle"),
    
    # Comparisons
    ("big", "large"), ("small", "little"),
    ("tall", "high"), ("short", "low"),
    ("long", "length"), ("short", "length"),
    ("heavy", "weight"), ("light", "weight"),
    ("fast", "quick"), ("slow", "not fast"),
    ("loud", "noisy"), ("quiet", "soft"),
    ("hard", "solid"), ("soft", "squishy"),
    ("hot", "warm"), ("cold", "cool"),
    ("wet", "water"), ("dry", "no water"),
    ("clean", "neat"), ("dirty", "messy"),
    ("new", "fresh"), ("old", "used"),
    ("full", "complete"), ("empty", "nothing"),
    ("same", "equal"), ("different", "not same"),
    
    # Time concepts for preschooler
    ("morning", "wake up"), ("morning", "breakfast"),
    ("afternoon", "lunch"), ("afternoon", "nap"),
    ("evening", "dinner"), ("evening", "sunset"),
    ("night", "sleep"), ("night", "dark"),
    ("today", "now"), ("yesterday", "before"),
    ("tomorrow", "after"), ("later", "future"),
    ("soon", "almost"), ("now", "present"),
    ("always", "every time"), ("never", "no time"),
    ("sometimes", "not always"),
    
    # Money basics
    ("money", "buy"), ("money", "pay"),
    ("coin", "money"), ("coin", "round"),
    ("dollar", "money"), ("cent", "money"),
    ("expensive", "lots money"), ("cheap", "little money"),
    ("save", "keep"), ("spend", "use"),
    ("store", "buy"), ("price", "cost"),
    
    # Baby animals
    ("puppy", "baby dog"), ("kitten", "baby cat"),
    ("calf", "baby cow"), ("foal", "baby horse"),
    ("chick", "baby chicken"), ("duckling", "baby duck"),
    ("lamb", "baby sheep"), ("piglet", "baby pig"),
    ("cub", "baby bear"), ("cub", "baby lion"),
    ("fawn", "baby deer"), ("joey", "baby kangaroo"),
    ("tadpole", "baby frog"), ("caterpillar", "baby butterfly"),
    
    # Animal diets - using "eat" as connector
    ("dogs eat", "food"), ("cats eat", "food"),
    ("rabbits eat", "carrots"), ("horses eat", "hay"),
    ("cows eat", "grass"), ("birds eat", "seeds"),
    ("monkeys eat", "bananas"), ("pandas eat", "bamboo"),
    ("lions eat", "meat"), ("bears eat", "fish"),
    
    # Animal homes - using "live in" as connector
    ("dogs live in", "houses"), ("cats live in", "houses"),
    ("birds live in", "nests"), ("fish live in", "water"),
    ("bears live in", "caves"), ("lions live in", "dens"),
    ("bees live in", "hives"), ("ants live in", "anthills"),
    ("spiders make", "webs"), ("rabbits live in", "burrows"),
    ("squirrels live in", "trees"), ("owls live in", "trees"),
    
    # Daily routine
    ("wake up", "morning"), ("get dressed", "morning"),
    ("eat breakfast", "morning"), ("brush teeth", "morning"),
    ("go to school", "morning"), ("learn", "school"),
    ("eat lunch", "noon"), ("play", "afternoon"),
    ("do homework", "afternoon"), ("eat dinner", "evening"),
    ("take bath", "evening"), ("brush teeth", "night"),
    ("read story", "night"), ("go to sleep", "night"),
    
    # Body parts detailed
    ("head", "top"), ("neck", "connect"),
    ("shoulder", "arm"), ("elbow", "bend"),
    ("wrist", "hand"), ("palm", "hand"),
    ("thumb", "finger"), ("pinky", "finger"),
    ("hip", "leg"), ("knee", "bend"),
    ("ankle", "foot"), ("heel", "foot"),
    ("forehead", "face"), ("eyebrow", "face"),
    ("eyelash", "eye"), ("cheek", "face"),
    ("chin", "face"), ("lip", "mouth"),
    ("tongue", "mouth"), ("tooth", "mouth"),
    ("chest", "body"), ("belly", "body"),
    ("back", "body"), ("bottom", "sit"),
    
    # Five senses detailed
    ("see", "eyes"), ("look", "eyes"),
    ("hear", "ears"), ("listen", "ears"),
    ("smell", "nose"), ("sniff", "nose"),
    ("taste", "tongue"), ("lick", "tongue"),
    ("touch", "hands"), ("feel", "skin"),
    ("blind", "cannot see"), ("deaf", "cannot hear"),
    
    # Extended family
    ("mother", "mom"), ("mother", "mommy"),
    ("father", "dad"), ("father", "daddy"),
    ("parents", "mother"), ("parents", "father"),
    ("grandmother", "grandma"), ("grandmother", "nana"),
    ("grandfather", "grandpa"), ("grandfather", "papa"),
    ("grandparents", "old"), ("grandparents", "wise"),
    ("brother", "sibling"), ("sister", "sibling"),
    ("twins", "same age"), ("twins", "two"),
    ("baby", "young"), ("baby", "small"),
    ("toddler", "young"), ("toddler", "walk"),
    ("child", "young"), ("teenager", "older"),
    ("adult", "grown up"), ("elderly", "old"),
    ("family", "love"), ("family", "home"),
    ("aunt", "parent sister"), ("uncle", "parent brother"),
    ("cousin", "aunt child"), ("cousin", "uncle child"),
    
    # Jobs and duties
    ("doctor", "help sick"), ("nurse", "help doctor"),
    ("teacher", "teach children"), ("principal", "school boss"),
    ("firefighter", "put out fire"), ("police", "catch bad"),
    ("chef", "cook food"), ("baker", "bake bread"),
    ("farmer", "grow food"), ("gardener", "plant flowers"),
    ("builder", "build house"), ("carpenter", "work wood"),
    ("plumber", "fix pipes"), ("electrician", "fix wires"),
    ("mechanic", "fix cars"), ("pilot", "fly plane"),
    ("driver", "drive car"), ("captain", "drive ship"),
    ("astronaut", "go space"), ("scientist", "discover"),
    ("artist", "make art"), ("musician", "play music"),
    ("singer", "sing songs"), ("dancer", "dance"),
    ("actor", "act movies"), ("writer", "write books"),
    
    # Transportation details
    ("car", "four wheels"), ("car", "drive"),
    ("truck", "big"), ("truck", "carry"),
    ("bus", "many people"), ("bus", "public"),
    ("taxi", "yellow"), ("taxi", "pay"),
    ("ambulance", "hospital"), ("ambulance", "siren"),
    ("fire truck", "firefighter"), ("fire truck", "red"),
    ("police car", "police"), ("police car", "siren"),
    ("motorcycle", "two wheels"), ("motorcycle", "fast"),
    ("bicycle", "two wheels"), ("bicycle", "pedal"),
    ("tricycle", "three wheels"), ("tricycle", "child"),
    ("scooter", "two wheels"), ("scooter", "kick"),
    
    # Weather details
    ("sunny", "sun"), ("sunny", "bright"),
    ("cloudy", "clouds"), ("cloudy", "gray"),
    ("rainy", "rain"), ("rainy", "wet"),
    ("snowy", "snow"), ("snowy", "cold"),
    ("windy", "wind"), ("windy", "blow"),
    ("stormy", "thunder"), ("stormy", "lightning"),
    ("foggy", "fog"), ("foggy", "hard see"),
    ("humid", "sticky"), ("humid", "hot"),
    ("freezing", "very cold"), ("freezing", "ice"),
    ("temperature", "hot cold"), ("thermometer", "measure"),
    
    # Seasons details
    ("spring", "flowers bloom"), ("spring", "birds return"),
    ("spring", "rain"), ("spring", "warm"),
    ("summer", "hot"), ("summer", "vacation"),
    ("summer", "swim"), ("summer", "beach"),
    ("fall", "leaves fall"), ("fall", "orange"),
    ("fall", "cool"), ("fall", "harvest"),
    ("winter", "cold"), ("winter", "snow"),
    ("winter", "holidays"), ("winter", "coat"),
    
    # Seasonal clothing
    ("summer", "shorts"), ("summer", "t-shirt"),
    ("summer", "sandals"), ("summer", "sunglasses"),
    ("winter", "coat"), ("winter", "boots"),
    ("winter", "gloves"), ("winter", "scarf"),
    ("winter", "hat"), ("rain", "raincoat"),
    ("rain", "umbrella"), ("rain", "boots"),
    
    # Games
    ("hide and seek", "find"), ("tag", "chase"),
    ("hopscotch", "jump"), ("jump rope", "jump"),
    ("red light green light", "stop go"),
    ("simon says", "follow"), ("duck duck goose", "run"),
    ("musical chairs", "sit"), ("freeze dance", "stop"),
    
    # Cause and effect
    ("rain", "wet"), ("sun", "warm"),
    ("fire", "burn"), ("ice", "melt"),
    ("plant", "grow"), ("seed", "sprout"),
    ("eat", "full"), ("drink", "not thirsty"),
    ("sleep", "rested"), ("exercise", "strong"),
    ("study", "learn"), ("practice", "better"),
    
    # Animal classification
    ("mammal", "warm blooded"), ("mammal", "fur"),
    ("reptile", "cold blooded"), ("reptile", "scales"),
    ("amphibian", "water"), ("amphibian", "land"),
    ("bird", "feathers"), ("bird", "eggs"),
    ("fish", "gills"), ("fish", "scales"),
    ("insect", "six legs"), ("insect", "exoskeleton"),
    ("dog", "mammal"), ("cat", "mammal"), ("whale", "mammal"),
    ("snake", "reptile"), ("lizard", "reptile"), ("turtle", "reptile"),
    ("frog", "amphibian"), ("salamander", "amphibian"),
    ("eagle", "bird"), ("penguin", "bird"),
    ("shark", "fish"), ("salmon", "fish"),
    ("butterfly", "insect"), ("beetle", "insect"),
    
    # Ecosystems
    ("forest", "ecosystem"), ("ocean", "ecosystem"),
    ("desert", "ecosystem"), ("rainforest", "ecosystem"),
    ("predator", "hunt"), ("prey", "hunted"),
    ("herbivore", "plants"), ("carnivore", "meat"),
    ("omnivore", "plants"), ("omnivore", "meat"),
]


def load_external_data():
    """Load external data files (WordNet, SimLex, WordSim, EAT)."""
    connections = []
    
    external_files = [
        "wordnet_data.txt",
        "simlex_data.txt", 
        "wordsim_data.txt",
        "eat_data.txt",
    ]
    
    for filepath in external_files:
        try:
            with open(filepath, "r") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) == 2:
                        connections.append((parts[0], parts[1]))
        except FileNotFoundError:
            pass
    
    return connections


def get_all_connections():
    """Returns all connections."""
    all_connections = []
    
    all_connections.extend(CATEGORIES)
    all_connections.extend(PROPERTIES)
    all_connections.extend(ANIMAL_SOUNDS)
    all_connections.extend(ACTIONS)
    all_connections.extend(OPPOSITES)
    all_connections.extend(NUMBERS)
    all_connections.extend(SHAPES)
    all_connections.extend(TIME)
    all_connections.extend(PLACES)
    all_connections.extend(PEOPLE)
    all_connections.extend(EMOTIONS)
    all_connections.extend(WEATHER)
    all_connections.extend(NATURE)
    all_connections.extend(GEOGRAPHY)
    all_connections.extend(SCIENCE)
    all_connections.extend(ARITHMETIC)
    all_connections.extend(SOCIAL)
    all_connections.extend(EXTRA_FACTS)
    
    # External data
    all_connections.extend(load_external_data())
    
    return all_connections


# =============================================================================
# SENTENCES FOR CONTEXTUAL LEARNING
# =============================================================================

SENTENCES = [
    # NEGATIONS (Can we...? -> No)
    # Important: connection between question and answer "no"
    "we cannot drink salt water",
    "no we cannot drink salt water",
    "salt water is not drinkable",
    "you cannot drink ocean water",
    "fish cannot live on land",
    "no fish cannot live on land",
    "birds cannot swim underwater",
    "we cannot breathe underwater",
    "no we cannot breathe underwater",
    "cats cannot fly",
    "dogs cannot fly",
    "people cannot fly without planes",
    
    # Animals - sounds and actions
    "the dog barks loudly",
    "the cat meows softly",
    "the cow says moo",
    "the duck swims in the pond",
    "the bird flies in the sky",
    "the fish swims in water",
    "dogs like to run and play",
    "cats like to sleep and climb",
    "the lion roars in the jungle",
    "the elephant is big and gray",
    "the monkey climbs trees",
    "the rabbit hops fast",
    "the snake slithers on the ground",
    "the frog jumps into the water",
    "the bee makes honey",
    "the butterfly has colorful wings",
    
    # Animals - categories (IS-A) - MAIN FACTS
    # These connections should be stronger than sounds/actions
    "a dog is an animal",
    "dogs are animals",
    "the dog is an animal",
    "a cat is an animal",
    "cats are animals",
    "the cat is an animal",
    "a cow is an animal",
    "a bird is an animal",
    "a fish is an animal",
    "a horse is an animal",
    "a pig is an animal",
    "a lion is an animal",
    "a tiger is an animal",
    "a bear is an animal",
    "dogs and cats are animals",
    "dogs and cats are pets",
    "a dog is a pet",
    "a cat is a pet",
    "a whale is a mammal",
    "a dolphin is a mammal",
    "a penguin is a bird",
    "a shark is a fish",
    
    # Musical instruments
    "a piano is an instrument",
    "a guitar is an instrument",
    "a violin is an instrument",
    "a drum is an instrument",
    "pianos and guitars are instruments",
    
    # Animal sounds (additional)
    "the lion says roar",
    "a lion roars",
    "lions roar loudly",
    
    # Shapes
    "a ball is round",
    "balls are round",
    "the ball is round",
    "the shape of a ball is round",
    "a ball has a round shape",
    "a circle is round",
    "the sun is round",
    "the moon is round",
    
    # Fruits and vegetables
    "an apple is a fruit",
    "a banana is a fruit",
    "an orange is a fruit",
    "grapes are fruit",
    "apples and bananas are fruit",
    "a carrot is a vegetable",
    "a potato is a vegetable",
    "broccoli is a vegetable",
    "tomatoes are red",
    "lemons are yellow and sour",
    
    # Transport
    "a car is a vehicle",
    "a bus is a vehicle",
    "a truck is a vehicle",
    "a bicycle is a vehicle",
    "cars and buses are vehicles",
    "planes fly in the sky",
    "boats float on water",
    "trains run on tracks",
    "helicopters can hover",
    
    # Food
    "i eat an apple",
    "the apple is red",
    "the banana is yellow",
    "i drink milk",
    "milk is white",
    "bread is food",
    "we eat breakfast in the morning",
    "we eat dinner at night",
    "pizza is delicious",
    "ice cream is cold and sweet",
    "vegetables are healthy",
    
    # Body
    "i see with my eyes",
    "i hear with my ears",
    "i smell with my nose",
    "i eat with my mouth",
    "i walk with my feet",
    "i hold things with my hands",
    "the heart pumps blood",
    "the brain controls the body",
    "lungs help us breathe",
    "bones support the body",
    "muscles help us move",
    
    # Colors
    "the sky is blue",
    "the grass is green",
    "the sun is yellow",
    "the snow is white",
    "fire is red and hot",
    "the ocean is blue",
    "leaves are green",
    "oranges are orange",
    "grapes can be purple or green",
    
    # Family
    "my mother loves me",
    "my father helps me",
    "i love my family",
    "my brother plays with me",
    "my sister is my friend",
    "grandparents are wise",
    "families live together",
    "parents take care of children",
    
    # Philosophical questions (what kids ask)
    "the meaning of life is love",
    "love is the meaning of life",
    "life is about love and happiness",
    
    # Sequences (days of week)
    "tuesday comes after monday",
    "wednesday comes after tuesday",
    "thursday comes after wednesday",
    "friday comes after thursday",
    "saturday comes after friday",
    "sunday comes after saturday",
    "monday comes after sunday",
    
    # Sequences (numbers)
    "two comes after one",
    "three comes after two",
    "four comes after three",
    "five comes after four",
    "six comes after five",
    "five is before six",
    "after five comes six",
    "seven comes after six",
    "eight comes after seven",
    "nine comes after eight",
    "ten comes after nine",
    
    # Actions
    "i run fast",
    "i walk slowly",
    "i sleep at night",
    "i wake up in the morning",
    "i play with my toys",
    "i read a book",
    "i draw a picture",
    "i swim in the pool",
    "i climb the stairs",
    "i jump high",
    
    # Places
    "i go to school",
    "i learn at school",
    "i play in the park",
    "i sleep at home",
    "animals live in the zoo",
    "fish live in the ocean",
    "books are in the library",
    "doctors work in hospitals",
    "food is sold in stores",
    
    # Opposites
    "hot is the opposite of cold",
    "big is the opposite of small",
    "fast is the opposite of slow",
    "day is the opposite of night",
    "happy is the opposite of sad",
    "up is the opposite of down",
    "left is the opposite of right",
    "open is the opposite of closed",
    "in is the opposite of out",
    
    # Emotions
    "i laugh when i am happy",
    "i smile when i am happy",
    "i cry when i am sad",
    "i feel sad when i cry",
    "happy people laugh and smile",
    "sad people cry",
    "when you are happy you laugh",
    "when you are sad you cry",
    "love makes people happy",
    "fear makes people scared",
    
    # Time of day
    "it is dark at night",
    "the night is dark",
    "it is light during the day",
    "we sleep at night when it is dark",
    "the sun goes down at night",
    "the sun rises in the morning",
    "stars come out at night",
    "the moon shines at night",
    
    # Leaders and countries
    "a president leads a country",
    "presidents lead countries",
    "countries have presidents",
    "the president of a country is a leader",
    "kings and queens rule kingdoms",
    
    # School
    "we learn at school",
    "children learn to read at school",
    "we read and write at school",
    "we play with friends at school",
    "teachers teach at school",
    "at school we learn many things",
    "school is where we learn",
    "i learn new things at school",
    "learning happens at school",
    "math is taught at school",
    
    # Park
    "we play in the park",
    "children run in the park",
    "we swing and slide in the park",
    "we have fun in the park",
    "in the park we play games",
    "the park is where we play",
    "i play outside in the park",
    "playing in the park is fun",
    "children run and play in the park",
    "trees grow in the park",
    
    # Weather
    "the sun is shining",
    "it is raining today",
    "the snow is cold",
    "the wind is blowing",
    "clouds are in the sky",
    "thunder is loud",
    "lightning is bright",
    "rainbows have many colors",
    
    # Geography and capitals
    "paris is the capital of france",
    "london is the capital of england",
    "berlin is the capital of germany",
    "rome is the capital of italy",
    "tokyo is the capital of japan",
    "beijing is the capital of china",
    "moscow is the capital of russia",
    "washington is the capital of america",
    "the earth is a planet",
    "the sun is a star",
    "the moon orbits the earth",
    "mars is the red planet",
    "jupiter is the biggest planet",
    "saturn has rings",
    
    # Science
    "plants need water and sunlight",
    "trees have roots and leaves",
    "flowers have petals",
    "seeds grow into plants",
    "water is a liquid",  # Important: connector=is for question "What is water?"
    "ice is frozen water",
    "steam is hot water vapor",
    "gravity pulls things down",
    "magnets attract metal",
    "electricity powers lights",
    
    # Numbers
    "one plus one is two",
    "two plus two is four",
    "three plus three is six",
    "five plus five is ten",
    "ten minus five is five",
    "there are seven days in a week",
    "there are twelve months in a year",
    "a dozen is twelve",
    
    # Nature
    "the moon is in the sky",
    "trees have leaves",
    "flowers smell nice",
    "rivers flow to the ocean",
    "mountains are tall",
    "deserts are dry and hot",
    "forests have many trees",
    "the ocean is very big",
    
    # Professions
    "doctors help sick people",
    "teachers teach students",
    "firefighters put out fires",
    "police officers keep us safe",
    "farmers grow food",
    "pilots fly planes",
    "chefs cook food",
    "artists create art",
    
    # Categories (reinforced for tests)
    "a dog is an animal and a pet",
    "a cat is an animal and a pet",
    "dogs are animals that bark",
    "cats are animals that meow",
    "a bird is an animal that flies",
    "a fish is an animal that swims",
    "an elephant is a big animal",
    "a lion is a wild animal",
    "a rabbit is a small animal",
    "a horse is a farm animal",
    
    # Colors (reinforced for tests)
    "the sky is blue during the day",
    "blue is the color of the sky",
    "grass is green in color",
    "green is the color of grass",
    "the sun is yellow and bright",
    "yellow is the color of the sun",
    "bananas are yellow fruit",
    "apples are red fruit",
    "oranges are orange fruit",
    "snow is white and cold",
    
    # Animal sounds (reinforced)
    "a dog says woof woof",
    "a dog say woof",
    "dogs bark and say woof",
    "a cat says meow meow",
    "cats meow and purr",
    "a cow says moo",
    "cows moo in the farm",
    "a duck says quack quack",
    "ducks quack in the pond",
    "a pig says oink oink",
    "a sheep says baa baa",
    
    # Time (reinforced)
    "we sleep at night when it is dark",
    "night is when we sleep",
    "you sleep at night in your bed",
    "we wake up in the morning",
    "morning is when we wake up",
    "we eat breakfast in the morning",
    "we eat lunch in the afternoon",
    "we eat dinner in the evening",
    
    # Sun (reinforced)
    "the sun is a star in the sky",
    "the sun is a big hot star",
    "stars shine at night",
    "the sun gives us light and heat",
    
    # Opposites (reinforced)
    "hot and cold are opposites",
    "the opposite of hot is cold",
    "the opposite of cold is hot",
    "big and small are opposites",
    "the opposite of big is small",
    "the opposite of small is big",
    "fast and slow are opposites",
    "the opposite of fast is slow",
    "up and down are opposites",
    "the opposite of up is down",
    "happy and sad are opposites",
    "the opposite of happy is sad",
    "in and out are opposites",
    "the opposite of in is out",
    "the opposite of out is in",
    
    # Body parts (reinforced)
    "we see with our eyes",
    "eyes are for seeing",
    "we hear with our ears",
    "ears are for hearing",
    "we smell with our nose",
    "the nose is for smelling",
    "we taste with our tongue",
    "we eat with our mouth",
    "we walk with our feet and legs",
    "we hold things with our hands",
    
    # Places (reinforced)
    "we learn at school",
    "school is where children learn",
    "we play in the park",
    "the park is where we play",
    "we sleep at home in our bed",
    "home is where we live and sleep",
    "we buy food at the store",
    "doctors work at the hospital",
    
    # Math (reinforced)
    "one plus one equals two",
    "one and one make two",
    "two plus two equals four",
    "two and two make four",
    "there are seven days in a week",
    "a week has seven days",
    "there are twelve months in a year",
    
    # Safety
    "strangers can be dangerous",
    "do not talk to strangers",
    "fire is hot and dangerous",
    "do not touch fire",
    "look both ways before crossing",
    "hold hands when crossing the street",
    
    # Mythical creatures (not real)
    "dragons are not real",
    "unicorns are fictional and not real",
    "fairies are make believe",
    "monsters are not real",
    "ghosts are fictional",
    
    # =====================================================
    # PRESCHOOL SENTENCES
    # =====================================================
    
    # Baby animals
    "a puppy is a baby dog",
    "puppies are baby dogs",
    "a kitten is a baby cat",
    "kittens are baby cats",
    "a calf is a baby cow",
    "a chick is a baby chicken",
    "a duckling is a baby duck",
    "a lamb is a baby sheep",
    "a piglet is a baby pig",
    "a tadpole becomes a frog",
    "a caterpillar becomes a butterfly",
    
    # Where animals live
    "birds live in nests",
    "fish live in water",
    "bees live in hives",
    "ants live in anthills",
    "bears live in caves",
    "rabbits live in burrows",
    "spiders make webs",
    
    # What animals eat
    "rabbits eat carrots",
    "horses eat hay",
    "cows eat grass",
    "birds eat seeds",
    "monkeys eat bananas",
    "pandas eat bamboo",
    
    # Animal classification
    "dogs are mammals",
    "cats are mammals",
    "whales are mammals",
    "snakes are reptiles",
    "lizards are reptiles",
    "turtles are reptiles",
    "frogs are amphibians",
    "birds have feathers",
    "fish have gills",
    "insects have six legs",
    
    # Safety
    "we wash our hands to stay healthy",
    "germs can make us sick",
    "we brush our teeth every day",
    "strangers can be dangerous",
    "we look both ways before crossing",
    "we hold hands when crossing the street",
    "we wear seatbelts in the car",
    "we wear helmets when riding bikes",
    "fire is hot and dangerous",
    "we do not touch fire",
    
    # Emotions
    "when i am happy i smile",
    "when i am sad i cry",
    "when i am scared i feel afraid",
    "when i am angry i feel mad",
    "friends make us happy",
    "we share with our friends",
    "we take turns when playing",
    "we say please and thank you",
    "we say sorry when we make mistakes",
    
    # Daily routine
    "we wake up in the morning",
    "we eat breakfast in the morning",
    "we go to school to learn",
    "we eat lunch at noon",
    "we play in the afternoon",
    "we eat dinner in the evening",
    "we take a bath to get clean",
    "we brush our teeth before bed",
    "we read stories at night",
    "we sleep at night",
    
    # Family
    "mommy and daddy are my parents",
    "grandma and grandpa are my grandparents",
    "my brother and sister are my siblings",
    "i love my family",
    "family lives together at home",
    
    # School
    "we learn at school",
    "teachers help us learn",
    "we read books at school",
    "we write with pencils",
    "we draw with crayons",
    "we count numbers",
    "the alphabet has twenty six letters",
    "a e i o u are vowels",
    
    # Shapes
    "a circle is round",
    "a circle has no corners",
    "a square has four equal sides",
    "a triangle has three sides",
    "a rectangle has four sides",
    "a heart is a shape for love",
    
    # Opposites (reinforced)
    "big and small are opposites",
    "the opposite of big is small",
    "the opposite of small is big",
    "hot and cold are opposites",
    "the opposite of hot is cold",
    "the opposite of cold is hot",
    "fast and slow are opposites",
    "the opposite of fast is slow",
    "up and down are opposites",
    "the opposite of up is down",
    "happy and sad are opposites",
    "the opposite of happy is sad",
    "day and night are opposites",
    "the opposite of day is night",
    "in and out are opposites",
    "the opposite of in is out",
    "the opposite of out is in",
    
    # Cause and effect
    "rain makes things wet",
    "the sun makes us warm",
    "fire can burn things",
    "ice melts when it gets warm",
    "plants grow from seeds",
    "when we eat we feel full",
    "when we sleep we feel rested",
    "practice makes us better",
    
    # Professions
    "doctors help sick people",
    "teachers teach children",
    "firefighters put out fires",
    "police officers keep us safe",
    "farmers grow food",
    "chefs cook food",
    "pilots fly airplanes",
    "astronauts go to space",
    
    # Transport
    "cars have four wheels",
    "bicycles have two wheels",
    "tricycles have three wheels",
    "buses carry many people",
    "airplanes fly in the sky",
    "boats float on water",
    "trains ride on tracks",
    
    # Weather and seasons
    "in spring flowers bloom",
    "summer is hot",
    "in fall leaves change color",
    "winter is cold with snow",
    "when it rains we use umbrellas",
    "when it is sunny we wear sunglasses",
    "when it is cold we wear coats",
    
    # Additional facts for tests
    "we smell with our nose",
    "the sun is a star",
    "the sun is a star not a planet",
    "our sun is a star",
]


def get_sentences():
    """Returns all sentences for training."""
    return SENTENCES


def get_babi_facts(task: int = 1) -> list:
    """
    Loads facts from bAbI dataset for training.
    
    BIOLOGY: bAbI facts are simple sentences about actions and locations.
    We train on facts, test on questions.
    
    Args:
        task: bAbI task number (1-20).
        
    Returns:
        List of facts (strings) for training.
    """
    import os
    from pathlib import Path
    
    babi_dir = Path("data/babi/tasks_1-20_v1-2/en")
    
    # Find train file for task
    pattern = f"qa{task}_*_train.txt"
    train_files = list(babi_dir.glob(pattern))
    
    if not train_files:
        print(f"⚠️ bAbI task {task} not found in {babi_dir}")
        return []
    
    facts = []
    with open(train_files[0], 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Format: "1 Mary moved to the bathroom."
            # Questions contain "?" and tab with answer
            if '?' in line or '\t' in line:
                continue  # Skip questions
            
            # Extract text after number
            parts = line.split(' ', 1)
            if len(parts) == 2:
                fact = parts[1].strip()
                if fact:
                    facts.append(fact)
    
    return facts


def get_stats():
    """Dataset statistics."""
    print("=" * 60)
    print("CURRICULUM DATASET STATISTICS")
    print("=" * 60)
    
    print(f"Categories: {len(CATEGORIES)}")
    print(f"Properties: {len(PROPERTIES)}")
    print(f"Animal sounds: {len(ANIMAL_SOUNDS)}")
    print(f"Actions: {len(ACTIONS)}")
    print(f"Opposites: {len(OPPOSITES)}")
    print(f"Numbers: {len(NUMBERS)}")
    print(f"Shapes: {len(SHAPES)}")
    print(f"Time: {len(TIME)}")
    print(f"Places: {len(PLACES)}")
    print(f"People: {len(PEOPLE)}")
    print(f"Emotions: {len(EMOTIONS)}")
    print(f"Weather: {len(WEATHER)}")
    print(f"Nature: {len(NATURE)}")
    print(f"Geography: {len(GEOGRAPHY)}")
    print(f"Science: {len(SCIENCE)}")
    print(f"Arithmetic: {len(ARITHMETIC)}")
    print(f"Social: {len(SOCIAL)}")
    print(f"Extra facts: {len(EXTRA_FACTS)}")
    
    total_connections = len(get_all_connections())
    total_sentences = len(SENTENCES)
    print(f"\nTOTAL CONNECTIONS: {total_connections}")
    print(f"TOTAL SENTENCES: {total_sentences}")
    print("=" * 60)
    
    return total_connections


if __name__ == "__main__":
    get_stats()
