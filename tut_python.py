# ------------------------------------------------------------------------------
# Python 3 cheatsheet
# ------------------------------------------------------------------------------

# ---------------------
# Python environment
# ---------------------
# Python 3.5.6 :: Anaconda, Inc.
# Numpy, Scipy (optimizers, regression, interpolation,...), 
# Matplotlib (2D visualization), Mayavi (3D visualization),
# scikit-learn (machine learning), scikit-image (image processing),
# sympy (Maple stuff),
# pandas (statistics)

# Interactive Python: type ipython in Anaconda console
# Run script: python my_script.py
# Run script in ipython: %run my_script.py


# ----------------------
# Code
# ----------------------

# Long lines (if more than 80)
long_line = "Here is a very very long line \
that we break in two parts."

# Print type of variable: print(type(a))
# Data types can be changed!

# Numbers
x = 3                           # int
z = 2
y = 2.5                         # float
print(x ** 2)                   # Exponentiation; prints "9"
x += 1                          # no x++, ++x, x--, --x !!!
print(y, y + 1, y * 2, y ** 2)  # Prints "2.5 3.5 5.0 6.25"
u = x / z                       # float division => 1.5
v = u // z                      # integer division => 1

# Complex numbers
a = 1.5 + 0.5j
print(a.real)
print(a.imag)


# Boolean logic
t = True
f = False
print(t and f) # AND
print(t or f)  # OR
print(not t)   # NOT
print(t ^ f)  # XOR


# Strings (are immutable)
hello = "hello"
world  = 'world'
hw = hello + " " + world 
print(hw)
print("%s %d %s" % (hello, 16, world))
s = "hello"
print(len(s))                   # length of string
print(s.capitalize())           # "Hello"
print(s.upper())                # "HELLO"
print(s.rjust(7))               # "  hello"
print(s.center(7))              # " hello "
print(s.replace("l", "(ell)"))  # replace ... by ...
print("  world ".strip())       # Strip leading and trailing whitespace; prints "world"
message = "Hello how are you?"
sp = message.split()
print(sp)


# Tuples (immutable lists)
t = (5, 6)
print(t)
print(t[0], t[1])


# Functions
# note that lists, dictionaries, sets and objects are mutable parameters!!!
# Functions are first-class objects!!! => function pointerish stuff possible!
def sign(x = 0):
    if x > 0:
        return 'pos'
    elif x < 0:
        return 'neg'
    else:
        return 'zero'


# Loops
for i in range(4): # ints in range [0, 4[
    print(i)

for i in [-1, 0, 1]:
    print(sign(i))

nums = [0, 1, 2, 3, 4]
squares = []
for x in nums:
    squares.append(x ** 2)
print(squares)              # Prints [0, 1, 4, 9, 16]

z = 1 + 1j
while abs(z) < 100:
    if z.imag == 0:
        break
    z = z**2 + 1


# Classes
class Greeter(object):  # inherits from object (can be another class too!)
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
g.greet()           # Call an instance method; prints "Hello, Tom"
print(g.name)
g.greet(loud=True)   # Call an instance method; prints "HELLO, TOM!"


# Modules
#
# Example: write all code for module demo in demo.py, 
# import demo / from demo import foo, bar
# demo.foo() 
#
# Modules are cached: if you modify demo.py 
# and re-import it in the old session, you will get the old one.
# Solution: importlib.reload(demo)


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


# Slicing [start:stop:stride]
print(nums[2:4])            # indexes [2, 4[
print(nums[2:])             # indexes [2, end[
print(nums[:2])             # indexes [start, 2[
print(nums[:])              # all indexes
print(nums[:-1])            # everything except last one
nums[2:4] = [8, 9]          # replace a sublist
nums[::-1]                  # reverse a list (by negative stride or reverse())
nums.reverse()
nums * 2                    # repeat list a few times
sorted(nums)                # return new sorted list
nums.sort()                 # in-place sorting of list              
print(nums)                 # prints "[0, 1, 8, 9, 4]"


# List comprehensions 
nums = [0, 1, 2, 3, 4]
even_squares = [x ** 2 for x in nums if (x % 2 == 0)]
print(even_squares)
# (can be used for dictionary as well {x: x ** 2 for x in nums if x % 2 == 0})


# Dictionaries (unordered)
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


# Sets (unordered unique items)
# https://docs.python.org/3.5/library/stdtypes.html#set
s = set(('a', 'b', 'c', 'a'))
s.add('d')
s.remove('d')
print(s)
print('c' not in s)
# difference, union, intersection, isdisjoint, issubset, issuperset,...
sd = s.difference(('a', 'b'))
print(sd)


# Loop over set
animals = {'cat', 'dog', 'fish'}
for idx, animal in enumerate(animals):
    print('#%d: %s' % (idx + 1, animal))


# I/O: Write to file (write: w, append: a, read/write: r+, binary: b)
# f = open("file.txt", 'w')
# f.write("This is a test \nand another test")
# f.close()


# I/O: Read from file (read: r)
# f = open("file.txt", 'r')
# s = f.read()
# f.close()
#
# f = open("file.txt", 'r')
# for line in f:
#   print(line)
# f.close()


# Exception handling
y = "ooh"
try:
    x = int(y)
except ValueError:
    print("That was no valid number.")
finally:
    # close files etc.
    print("Thanks!")