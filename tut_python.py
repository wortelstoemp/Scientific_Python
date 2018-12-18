# ------------------------------------------------------------------------------
# Python cheatsheet
# ------------------------------------------------------------------------------

# Numbers
x = 3                           # int
y = 2.5                         # float
print(x ** 2)                   # Exponentiation; prints "9"
x += 1                          # no x++, ++x, x--, --x !!!
print(y, y + 1, y * 2, y ** 2)  # Prints "2.5 3.5 5.0 6.25"


# Boolean logic
t = True
f = False
print(t and f) # AND
print(t or f)  # OR
print(not t)   # NOT
print(t ^ f)  # XOR


# Strings
hello = "hello"
world  = 'world'
hw = hello + " " + world 
print(hw)
print("%s %d %s" % (hello, 16, world))

s = "hello"
print(s.capitalize())           # "Hello"
print(s.upper())                # "HELLO"
print(s.rjust(7))               # "  hello"
print(s.center(7))              # " hello "
print(s.replace("l", "(ell)"))  # replace ... by ...
print("  world ".strip())       # Strip leading and trailing whitespace; prints "world"


# Tuples
t = (5, 6)
print(t)
print(t[0], t[1])


# Functions
# note that lists, dictionariesm sets and objects are mutable parameters!!!
def sign(x=0):
    if x > 0:
        return 'pos'
    elif x < 0:
        return 'neg'
    else:
        return 'zero'

for i in [-1, 0, 1]:
    print(sign(i))


# Classes
class Greeter(object):

    # Constructor
    def __init__(self, name):
        self.name = name

    # Method
    def greet(self, loud=False):
        if loud:
            print('HELLO, %s!' % self.name.upper())
        else:
            print('Hello, %s' % self.name)

g = Greeter('Tom')  # Construct an instance of the Greeter class
g.greet()            # Call an instance method; prints "Hello, Tom"
g.greet(loud=True)   # Call an instance method; prints "HELLO, TOM!"


# Containers: lists
xs = [3, 1, 2]      # Create a list
print(xs)           # Prints "[3, 1, 2]"

nums = list(range(5))
print(nums)         # Prints "[0, 1, 2, 3, 4]"

print(xs[2])        # 2 (indexes start from 0)
print(xs[-1])       # Negative indices count from the end of the list, starting from -1
xs[2] = "foo"       # Lists can mix types
print(xs)           # Prints "[3, 1, 'foo']"
xs.append("bar")    # Append to list
print(xs)           # Prints "[3, 1, 'foo', 'bar']"
x = xs.pop()        # Remove and return the last element of the list
print(xs, x)        # Prints "[3, 1, 'foo'] bar"

# Slicing
print(nums[2:4])            # indexes [2, 4[
print(nums[2:])             # indexes [2, end[
print(nums[:2])             # indexes [start, 2[
print(nums[:])              # all indexes
print(nums[:-1])            # everything except last one
nums[2:4] = [8, 9]          # replace a sublist
print(nums)                 # prints "[0, 1, 8, 9, 4]"


# Loops
nums = [0, 1, 2, 3, 4]
squares = []
for x in nums:
    squares.append(x ** 2)
print(squares)              # Prints [0, 1, 4, 9, 16]


# List comprehensions 
nums = [0, 1, 2, 3, 4]
even_squares = [x ** 2 for x in nums if (x % 2 == 0)]
print(even_squares)
# (can be used for dictionary as well {x: x ** 2 for x in nums if x % 2 == 0})


# Dictionaries
dic = {"cat": "cute", "dog": "furry"}
print(dic)
dic["dog"] = "woofy"
print(dic["dog"])
print("cat" in dic)           # Check if a dictionary has a given key; prints "True"
del dic["cat"]
print(dic.get("fish", 'N/A'))   # Get an element with a default; prints "N/A"

dic = {'human': 2, 'cat': 4, 'spider': 8}
for animal in dic:
    legs = dic[animal]
    print('A %s has %d legs' % (animal, legs))


# Sets
# TODO: http://cs231n.github.io/python-numpy-tutorial/#python-sets