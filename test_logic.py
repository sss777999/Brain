import re

def has_location(words):
    location_preps = {'in', 'at', 'on', 'inside', 'outside', 'to', 'into', 'hallway', 'kitchen', 'bedroom', 'garden', 'office', 'bathroom'}
    return any(w in location_preps for w in words)

print(has_location({'daniel', 'travelled', 'to', 'the', 'bedroom'}))
print(has_location({'daniel', 'picked', 'up', 'the', 'apple', 'there'}))

