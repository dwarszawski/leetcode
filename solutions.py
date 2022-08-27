## cheatsheet


# max and min values to be used
a = float("inf")
b = float("-inf")

longest = max(a, b)
shortest = max(a, b)

# reversed ordering
for col in reversed(range(len(arr))):
    pass

## reverse string with slicing
txt = "Hello World"[::-1]
## copy of the list:
copied = currentCombination[:]

# reverse list
list(reversed(sequence))
# sum all elements in the list
sum(array)

## nested for loop
nums = [[1, 2], [3, 4], [5, 6]]
result = [[col for col in row] for row in nums]
# Out[4]: [[1, 2], [3, 4], [5, 6]]
result[0]
# Out[6]: [1, 2]

## item getter

from operator import itemgetter

def findPositions(arr, x):
    # Write your code here

    q = [(value, idx) for idx, value in enumerate(arr, start=1)]
    output = []

    for _ in range(x):
        tmp = q[:x]
        maxItem = max(tmp, key=itemgetter(0))
        output.append(maxItem[1])
        tmp.remove(maxItem)

        tmp = [(v -1, idx) if v > 0 else (0, idx) for v, idx in tmp]
        q = q[x:] + tmp

    return output


## is alphanumeric char:
'1'.isalpha()
# Out[30]: False
'a'.isalnum()
# Out[31]: True
'a'.isalpha()
# Out[32]: True
'1'.isnumeric()
# Out[19]: True

'a'.islower()
# Out[33]: True
'a'.isupper()
# Out[34]: False


## index of element in list
["foo", "bar", "baz"].index("bar")
"adasd".index('a')
# Out[2]: 0
"adasd".index('d')
# Out[3]: 1

"dsadvfd dfdsf fdfd".split(" ")
# Out[4]: ['dsadvfd', 'dfdsf', 'fdfd']


## filtering list
tokens = filter(isImportantToken, path.split("/ "))

carry = value // 10
leastSignificantDigit = value % 10
offset = abs(k) % listLength

## Python priority queue - min heap by default
from queue import PriorityQueue

q = PriorityQueue()

q.put(4)
q.put(2)
q.put(5)
q.put(1)
q.put(3)

while not q.empty():
    next_item = q.get()
    print(next_item)

## Python queue
# Python program to
# demonstrate queue implementation
# using collections.dequeue


from collections import deque

# Initializing a queue
q = deque()

# Adding elements to a queue
q.append('a')
q.append('b')
q.append('c')

print("Initial queue")
print(q)

# Removing elements from a queue
print("\nElements dequeued from the queue")
print(q.popleft())
print(q.popleft())
print(q.popleft())

print("\nQueue after removing elements")
print(q)

## Sorting list
times.sort(key=lambda x: x[0])  # custom sorting

a = [4, 1, 3, 4]
a.sort()  # ascending by default
a == [1, 3, 4, 4]
a.sort(reverse=True)  # descending
a == [4, 4, 3, 1]


## Translation
## exponent - wykładnik potęgowania
## exponential - wykładnicza


# items = "FDFFDFDD", return 2
# "FD" creates the first balanced meal.
# "FFDFDD" creates the second balanced meal.


def balancedMeals(items):


    foodCounter = 0
drinkCounter = 0
mealsCounter = 0

for item in items:
    if item == "F":
        foodCounter += 1
    else:
        drinkCounter += 1

    if foodCounter == drinkCounter:
        mealsCounter += 1

return mealsCounter

balancedMeals("FDFFDFDD")
balancedMeals("FDFDFD")



#A company is booking flights to send its employees to its two satellite offices A and B.
# The cost of sending the ith employee to office A and office B is given by prices[i][0] and prices[i][1] respectively.
# Given that half the employees must be sent to office A and half the employees must be sent to office B,
# return the minimum cost the company must pay for all their employees’ flights.

# prices = [[40,30],[300,200],[50,50],[30,60]], return 310
# prices = [[30,60], [40,30], [50,50], [300,200],,], return 310
# Fly the first personn to office B.
# Fly the second person to office B.
# Fly the third person to office A.
# Fly the fourth person to office A.

def minimumCost(prices):
    minCost = 0
    officeThreshold = len(prices) / 2
    prices.sort(key=lambda x: x[0] - x[1])
    print(prices)
    for i in range(len(prices)):
        if i < officeThreshold:
            minCost += prices[i][0]
        else:
            minCost += prices[i][1]
    return minCost

assert minimumCost([[40, 30], [300, 200], [50, 50], [30, 60]]) == 310


# bricks = [1000, 1000, 1000, 2000], return 3.

def minBricks(bricks):


    bricks.sort()
maxWeight = 5000
bricksCounter = 0
totalWeight = 0
for i in range(len(bricks)):
    currentBrick = bricks[i]
if totalWeight + currentBrick < maxWeight:
    totalWeight += currentBrick
bricksCounter += 1
else:
break
return bricksCounter

minBricks([1000, 1000, 1000, 2000])
minBricks([1000, 200, 150, 200])

base62vals = []
myBase = 62
while num > 0:
    reminder = num % myBase
num = num / myBase
base62vals.insert(0, reminder)


# This question is asked by Amazon. Given a string s and a list of words representing a dictionary, return whether or not the entirety of s can be segmented into dictionary words.
# Note: You may assume all characters in s and the dictionary are lowercase.
#
#
# s = "thedailybyte", dictionary = ["the", "daily", "byte"], return true.
#    0 1 2 3
# "t h e d a i l y b y t e"
# [T F F F T F F F F T F F F]
#                    j     i
#  0 1 2 3 4
# s = "pizzaplanet", dictionary = ["plane", "pizza"], return false.

def dictionaryWords(s, words):


    prefixes = [False for i in range(len(s) + 1)]
prefixes[0] = True

for i in range(len(s) + 1):
    for j in range(i):
        if prefixes[j] and s[j:i] in words:
            prefixes[i] = True
            break

return prefixes[len(s)]

dictionaryWords("thedailybyte", ["the", "daily", "byte"])
dictionaryWords("pizzaplanet", ["plane", "pizza"])


# costs = [[1, 3, 5],[2, 4, 6],[5, 4, 3]], return 8.
# Paint the first house red, paint the second house blue, and paint the third house green.
# The cost of painting the ith house red, blue or green, is given by costs[i][0], costs[i][1], and costs[i][2]

def minCost(costs):
    # [1, 2]
    # [R, R]
    # [1,3,5] [2,4,6]
    for i in range(1, len(costs)):
        costs[i][0] += min(costs[i - 1][1], costs[i - 1][2])
        costs[i][1] += min(costs[i - 1][0], costs[i - 1][2])
        costs[i][2] += min(costs[i - 1][0], costs[i - 1][1])

    return min(costs[-1][0], costs[-1][1], costs[-1][2])

minCost([[1, 3, 5], [2, 4, 6], [5, 4, 3]])


# W = 10, weights = [4, 1, 3], values = [4, 2, 7], return 13.
# W = 5, weights = [2, 4, 3], values = [3, 7, 2], return 7.
# W = 7, weights = [1, 3, 4], values = [3, 5, 6], return 11.
#
#  1
#  3
#  4

class Art:


    def __init__(self, weight, value):


    self.weight = weight
self.value = value


def maxStolenValue(W, weights, values):


# [[0,3], [0, 3, 5, 8], [0, 3, 5, 6, 9, 11]]
currentValues = []
currentWeights = []

if len(weights) > 0:
    currentValues.extend([0, values[0]])
    currentWeights.extend([0, weights[0]])

for i in range(1, len(weights)):
    currentLength = len(currentValues)
    for j in range(currentLength):
        currentValue = currentValues[j]
        currentWeight = currentWeights[j]
        if currentWeight + weights[i] <= W:
            currentValues.append(currentValue + values[i])
            currentWeights.append(currentWeight + weights[i])

maxValue = 0
for i in range(len(currentValues)):
    maxValue = max(maxValue, currentValues[i])
return maxValue


def maxStolenValue2(W, weights, values):


    dp = [[0 for _ in range(W + 1)] for _ in range(len(values) + 1)]

for i in range(0, len(dp)):
    for j in range(0, len(dp[i])):
        if i == 0 or j == 0:
            dp[i][j] = 0
        elif j < weights[i - 1]:
            dp[i][j] = dp[i - 1][j]
        else:
            dp[i][j] = max(values[i - 1] + dp[i - 1][j - weights[i - 1]], dp[i - 1][j])

return dp

maxStolenValue2(W=10, weights=[4, 1, 3], values=[4, 2, 7])
maxStolenValue2(W=5, weights=[2, 4, 3], values=[3, 7, 2])
maxStolenValue(W=7, weights=[1, 3, 4], values=[3, 5, 6])


#        i
# [2, 4, 3]
# [3, 7, 2]
#       j
#   0 1 2 3 4 5    W
# 0 0 0 0 0 0 0
# 1 0 0 3 3 3 3
# 2 0 0 3 3 7 7
# 3 0 0 3 3
# V


# N = 5 and release number four is the release your bug was shipped in...
# isBadRelease(3) // returns false.
# isBadRelease(5) // returns true.
# isBadRelease(4) // returns true.
# i       i
# 0 1 2 3 4 5
# F F F F T T
def findBug(N):


    left = 0
right = N
while left <= right:
    mid = (left + right) // 2

    if mid == 0:
        return mid
    if isBadRelease(mid) and not isBadRelease(mid - 1):
        return mid
    elif isBadRelease(mid):
        right = mid - 1
    else:
        left = mid + 1

return -1


def findBadRelease(N):


    return findBadReleaseHelper(0, N)


def findBadReleaseHelper(start, end):


    middle = (start + end) // 2

if middle == 0:
    return 0

if isBadRelease(middle) and not isBadRelease(middle - 1):
    return middle
elif isBadRelease(middle) and isBadRelease(middle - 1):
    return findBadReleaseHelper(start, middle)
else:
    return findBadReleaseHelper(middle + 1, end)


def isBadRelease(i):


    if i >= 0:
    return True
else:
return False

import math
def isPrime(n):


    for i in range(2, int(math.sqrt(n) + 1)):
    if n % i == 0:
    return False
return True
isPrime(3)
isPrime(4)
isPrime(5)


def numberOfPrimeFactors(n):


    factors = []
for i in range(2, int(math.sqrt(n)) + 1):
    if n % i == 0:
    factors.append(i)
n = n / i
return factors

numberOfPrimeFactors(3)  # int(math.sqrt(3)) - number of distinct factors(not only prime numbers) including 1 and n
numberOfPrimeFactors(24)  # int(math.sqrt(24)) - number of factors (not only prime numbers) including 1 and n
numberOfPrimeFactors(25)  # int(math.sqrt(25)) - number of factors (not only prime numbers) including 1 and n


# This question is asked by Facebook.
# In a gym hallway there are N lockers.
# You walk back and forth down the hallway opening and closing lockers.
# On your first pass you open all the lockers.
# On your second pass, you close every other locker.
# On your third pass you open every third locker.
# After walking the hallway N times opening/closing lockers in the previously described manner,
# how many locker are left open?

#
# N =1
# N =2
# N=3
# N=4
#
# O C O
def gymLockers(n):


    return int(math.sqrt(n))  # equals to number of distinct factors

# N =1 # O O O
# N =2 # O C O
# N =3 #O C O
gymLockers(3)  # equals 1

# N =1 #O O O O
# N =2 #O C O C
# N =3 #O C O C
# N =4 #O C O C
gymLockers(4)  # equals 2


#
# nums = [1, 4, 2, 0], return 3.
# 3 is the only number missing in the array between 0 and 4.
# Ex: Given the following array nums…
#
# nums = [6, 3, 1, 2, 0, 5], return 4.
# 4 is the only number missing in the array between 0 and 6.
def missingValue(nums):


    maxValue = 0
a = {}
for num in nums:
    maxValue = max(maxValue, num)
a[num] = True

for i in range(maxValue):
    if i not in a:
        return i
return 0


def guassian(nums):
    sum = 0
    for num in nums:
        sum += num
    n = len(nums)
    return int((n * (n + 1) / 2) - sum)


missingValue([1, 4, 2, 0])
guassian([1, 4, 2, 0])
missingValue([6, 3, 1, 2, 0, 5])
guassian([6, 3, 1, 2, 0, 5])


## complementary number
# number = 27, return 4.
# 27 in binary (not zero extended) is 11011.
# Therefore, the complementary binary is 00100 which is 4.

# Runtime: O(logN) where N is the origin number we’re given.
# Space complexity: O(1) or constant.
def complementaryNumber(num):
    power = 1
    result = 0
    while num > 0:
        result += power * ((num % 2) ^ 1)  # num % 2 check the last bit is 1 or zero (odd or even) # ((num % 2) ^ 1) xor to flip the bits # power * ((num % 2) ^ 1) include the the index of the bit by simulating next powers of two
    num >>= 1
    power <<= 1
    return result


complementaryNumber(27)


#    11011
#    last bit xor with (^1)
#    multiply by next powers of 2 starting from 1,2,4,8
#    power <<= 1 will give us 1, 10, 100, 1000
#




#
# Finding friends

# This question is asked by Facebook. You are given a two dimensional matrix,friends, that represents relationships between coworkers in an office. If friends[i][j] = 1 then coworker i is friends with coworker j and coworker j is friends with coworker i. Similarly if friends[i][j] = 0 then coworker i is not friends with coworker j and coworker j is not friend with coworker i. Friendships in the office are transitive (i.e. if coworker one is friends with coworker two and coworker two is friends with coworker three, coworker one is also friends with coworker three). Given the friendships in the office defined by friends, return the total number of distinct friends groups in the office.
# Note: Each coworker is friends with themselves (i.e.matrix[i][j] = 1 for all values where i = j)
#
# Ex: Given the following matrix friends…
#
# friends = [
#               [1, 1, 0],
#               [1, 1, 0],
#               [0, 0, 1]
#           ], return 2.
# The 0th and 1st coworkers are friends with one another (first friend group).
# The 2nd coworker is friends with themself (second friend group).

# friends = [
# [1, 1, 0],
# [1, 1, 0],
# [0, 0, 1]
# ], return 2.
# The 0th and 1st coworkers are friends with one another (first friend group).
# The 2nd coworker is friends with themself (second friend group).


# DFS' time complexity is proportional to the total number of vertexes and edges of the graph visited. In that case, there are N*M vertexes and slightly less than 4*N*M edges, their sum is still O(N*M).
# Time complexity : O(M×N) where M is the number of rows and N is the number of columns.
# Space complexity : O(M,N)) because in worst case where the grid is filled with lands, the size of queue can grow up to O(M,N).

# Time complexity of DFS is O(N*M ) so in this case is O(N ^ 2)
# Space complexity
def findFriends(friends):
    groupCounter = 0

    for i in range(len(friends)):
        for j in range(len(friends[i])):
            if friends[i][j] == 1:
                groupCounter += 1
                exploreGroup(friends, i, j)
    return groupCounter


def exploreGroup(friends, i, j):
    if i < 0 or i >= len(friends) or j < 0 or j >= len(friends[i]) or friends[i][j] == 0:
        return
    friends[i][j] = 0

    exploreGroup(friends, i, j + 1)
    exploreGroup(friends, i + 1, j)
    exploreGroup(friends, i, j - 1)
    exploreGroup(friends, i - 1, j)


findFriends([[1, 1, 0], [1, 1, 0], [0, 0, 1]])


### find friends using Union Find data structure


# [1, 1, 0],
# [1, 1, 0],
# [0, 0, 1]

# [1,-1,-1]
# []

# Runtime: O(N^3) where N is the number of coworkers we’re given. This results from iterating through our N * N matrix and for each value that is equal to one, we can iterate through all of the N coworkers in the worst case.
# Space complexity: O(N) where N is the number of coworkers we’re given (i.e. friends.length). This results from creating our parent array that is of size N.
def findFriends(friends):
    parents = [-1 for _ in friends]

    for i in range(len(friends)):
        for j in range(len(friends[i])):
            if friends[i][j] == 1 and i != j:
                union(parents, i, j)

    friendCounter = 0
    for parent in parents:
        if parent == -1:
            friendCounter += 1

    return friendCounter


def union(parents, i, j):
    iGroup = find(parents, i)
    jGroup = find(parents, j)

    if (iGroup != jGroup):
        parents[iGroup] = jGroup


def find(parents, i):
    if parents[i] == -1:
        return i
    return find(parents, parents[i])


## count number of primes less than N
import math


# Naive Approach: Iterate from 2 to N, and check for prime. If it is a prime number, print the number.

# N = 7, return 3.
# 2, 3, and 5 are the only prime numbers less than 7.
# Time Complexity: O(N * N)
def countPrimes(N):
    primeCounter = 0

    for i in range(2, N):
        if isPrime(i):
            primeCounter += 1
    return primeCounter


def isPrime(N):
    for i in range(2, N):
        if N % i == 0:
            return False
    return True


countPrimes(7)  # should return 3
countPrimes(3)  # should return 1


## A better approach is based on the fact that one of the divisors must be smaller than or equal to √n. So we check for divisibility only till √n.
# Every composite number has at least one prime factor less than or equal to square root of itself.
# Time Complexity: O(N^3/2)


## the best solution is to use sieve of Eratosthenes
## count prime numbers using sieve of eratosthenes
# Time Complexity: O(n*log(log(n)))
# Auxiliary Space: O(n)
def countPrimes(N):
    prime = [True for _ in range(N)]
    p = 2

    while p * p < N:

        # If prime[p] is not changed, then it is a prime
        if prime[p] == True:
            # Update all multiples of p
            for i in range(2 * p, N, p):
                prime[i] = False
        p += 1

    primeCounter = 0
    for i in range(2, N):
        if prime[i]:
            primeCounter += 1
    return primeCounter


countPrimes(7)  # should return 3
countPrimes(3)  # should return 1


# ITERATIVE IN-ORDER TRAVERSAL

def inOrderTraversal(root):
    ordered = []
    inorder(root, ordered)
    return ordered


def inorder(root, ordered):
    if not root:
        return

    inorder(root.left)
    ordered.append(root.val)
    inorder(root.right)


#     2
#    / \
#   1   7
#  / \
# 4   8

#  8 2
#  4 1
# return [4, 1, 8, 2, 7]

# Runtime: O(N) where N is the number of nodes in our tree.
# Space complexity: O(N) where N is the number of nodes in our tree.
def iterativeTraversal(root):
    ordered = []
    stack = []
    current = root
    while current or len(stack) > 0
        while current:
            stack.append(current)
            current = current.left
        current = stack.pop()
        ordered.append(current.val)
        current = current.right

    return ordered


# This question is asked by Facebook. Given a string, reverse the vowels of it.
# Note: In this problem y is not considered a vowel.
#
# Ex: Given the following strings s…
#
# s = "computer", return "cemputor"
# Ex: Given the following strings s…
#
# s = "The Daily Byte", return "The Dialy Byte"

def reverseVowels(s):
    vowels = ['a', 'e', 'i', 'u', 'o', 'A', "E", "I", "U", "O"]
    i = 0
    j = len(s) - 1

    result = list(s)
    while i < j:
        while i < j and result[i] not in vowels:
            i += 1
        while i < j and result[j] not in vowels:
            j -= 1
        if i < j:
            tmp = result[i]
            result[i] = result[j]
            result[j] = tmp
            i += 1
            j -= 1

    return ''.join(result)


assert reverseVowels("computer") == "cemputor"  # , return "cemputor"
assert reverseVowels("cOmputer") == "cemputOr"  # , return "cemputor"
assert reverseVowels("The Daily Byte") == "The Dialy Byte"  # s = "The Daily Byte", return "The Dialy Byte"


# You are given two lists of integers and an integer representing a process id to kill. One of the lists represents a list of process ids and the other represents a list of each of the processes’ corresponding (by index) parent ids. When a process is killed, all of its children should also be killed. Return a list of all the process ids that are killed as a result of killing the requested process.
#
# Ex: Given the following…
#
# pid =  [2, 4, 3, 7]
# ppid = [0, 2, 2, 3]
# kill = 3
# the tree of processes can be represented as follows:
#        2
#       / \
#     4     3
#          /
#         7
# return [3, 7] as killing process 3 will also require killing its child (i.e. process 7).

def exited(pid, ppid, kill):
    next = kill
    result = []
    while next:
        for p in pid:
            if p == next:
                result.append(p)
                for idx, c in enumerate(ppid):
                    if c == p:
                        next = pid[idx]
                        break
                if p == next:
                    return result
        break
    return result


exited([2, 4, 3, 7], [0, 2, 2, 3], 3) == [3, 7]
exited([2, 4, 3, 7, 8], [0, 2, 2, 3, 7], 3) == [3, 7, 8]
exited([2, 4, 3, 7, 8], [0, 2, 2, 3, 7], 9) == []


def exited2(pid, ppid, kill):
    parentToChild = {}
    for idx, p in enumerate(ppid):
        if p in parentToChild.keys():
            parentToChild[p].append(pid[idx])
        else:
            parentToChild[p] = [pid[idx]]

    result = []
    next = [kill] if kill in pid else []
    while len(next) > 0:
        curr = next.pop(0)
        result.append(curr)
        if curr in parentToChild.keys():
            for e in parentToChild[curr]:
                next.append(e)
    return result


exited2([2, 4, 3, 7], [0, 2, 2, 3], 3) == [3, 7]
exited2([2, 4, 3, 7, 8], [0, 2, 2, 3, 7], 3) == [3, 7, 8]
exited2([2, 4, 3, 7, 8], [0, 2, 2, 3, 7], 9) == []


# This question is asked by Amazon. Given two strings, passage and text return whether or not the characters in text can be used to form the given passage.
# Note: Each character in text may only be used once and passage and text will only contain lowercase alphabetical characters.
#
# Ex: Given the following passage and text…
#
# passage = "bat", text = "cat", return false.
# Ex: Given the following passage and text…
#
# passage = "dog" text = "didnotgo", return true.

# Time complexity O(m + n)
# Space complexity O(1) since there are only 26 chars
def containsChar(passage, text):
    freq = {}
    for char in text:
        if char in freq.keys():
            freq[char] += 1
        else:
            freq[char] = 1
    for c in passage:
        if c in freq.keys() and freq[c] > 0:
            freq[c] -= 1
        else:
            return False
    return True


containsChar("dog", "didnotgo") == True
containsChar("bat", "cat") == False


# STRINGS
# This question is asked by Google. Given two strings s and t return whether or not s is a subsequence of t.
# Note: You may assume both s and t only consist of lowercase characters and both have a length of at least one.
#
# Ex: Given the following strings s and t…
#
# s = "abc", t = "aabbcc", return true.
# Ex: Given the following strings s and t…
#
# s = "cpu", t = "computer", return true.
# Ex: Given the following strings s and t…
#
# s = "xyz", t = "axbyc", return false.

#     i
# "abc"
#      j
# "aabbcc"

# Time complexity O(t)
# Space complexity O(1)
def isSubstring(s, t):
    i = 0
    j = 0
    while i < len(s) and j < len(t):
        if s[i] == t[j]:
            i += 1
        j += 1

    return i == len(s)


assert isSubstring("abc", "aabbcc") == True
assert isSubstring("xyz", "axbyc") == False
assert isSubstring("cpu", "computer") == True


# Given a 2D array of integers with ones representing land and zeroes representing water, return the number of islands in the grid. Note: an island is one or more ones surrounded by water connected either vertically or horizontally. Ex: Given the following grid…
#
# 11000
# 11010
# 11001
# return 3.
# Ex: Given the following grid…
#
# 00100
# 00010
# 00001
# 00001
# 00010
# return 4.

# Time complexity O(mxn)
# Space complexity O(mxn) - number of recursive calls
def countIslands(grid):
    counter = 0

    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] == 1:
                explore(grid, i, j)
                counter += 1
    return counter


def explore(grid, i, j):
    if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[i]) or grid[i][j] == 0:
        return
    grid[i][j] = 0
    explore(grid, i + 1, j)
    explore(grid, i - 1, j)
    explore(grid, i, j + 1)
    explore(grid, i, j - 1)


# 11000
# 11010
# 11001
countIslands([[1, 1, 0, 0, 0], [1, 1, 0, 1, 0], [1, 1, 0, 0, 1]]) == 3

# 00100
# 00010
# 00001
# 00001
# 00010
countIslands([[0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 0, 1, 0]]) == 4

# STRINGS
# COMPRESSION
# This question is asked by Facebook. Given a character array, compress it in place and return the new length of the array.
# Note: You should only compress the array if its compressed form will be at least as short as the length of its original form.
#
# Ex: Given the following character array chars…
#
# chars = ['a', 'a', 'a', 'a', 'a', 'a'], return 2.
# chars should be compressed to look like the following:
# chars = ['a', '6']
# Ex: Given the following character array chars…
#
# chars = ['a', 'a', 'b', 'b', 'c', 'c'], return 6.
# chars should be compressed to look like the following:
# chars = ['a', '2', 'b', '2', 'c', '2']
# Ex: Given the following character array chars…
#
# chars = ['a', 'b', 'c'], return 3.
# In this case we chose not to compress chars.

#   i                            j
# ['a', 'a', 'a', 'a', 'a', 'a']

#           i         j
# chars = ['a', 'a', 'b', 'b', 'c', 'c']
# ['a', '2', 'b', '2', 'c', '2']

# Time complexity O(N)
# Space complexity O(1)
def compress(chars):
    index = 0
    i = 0
    while i < len(chars):
        j = i

        '''
           i  j
        aaa -> 2
        ab -> 2
        abb -> 3
        '''
        while j < len(chars) and chars[i] == chars[j]:
            j+=1
        if j - i > 1:
            for _ in str(j-i):
                index+=1
        index+=1 # always increase by a given character
        i = j
        j+=1
    return min(index, len(chars))


assert compress(['a', 'a', 'b', 'b', 'c', 'c']) == 6
assert compress(['a', 'b', 'c']) == 3
assert compress(['a', 'a', 'a', 'a', 'a', 'a']) == 2
assert compress(['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a']) == 3
assert compress(['a']) == 1
assert compress([]) == 0


# including inplace update
# Time complexity O(N)
# Space complexity O(1)
def compress2(chars):
    charsLen = len(chars)
    index = 0
    i = 0
    while i < charsLen:
        j = i
        while j < charsLen and chars[i] == chars[j]:
            j += 1
        chars[index] == chars[i]
        index += 1
        if j - i > 1:
            count = str(j - i)
            for c in count:
                chars[index] = c
                index += 1
        i = j
    return index


compress2(['a', 'a', 'b', 'b', 'c', 'c']) == 6
compress2(['a', 'b', 'c']) == 3
compress2(['a', 'a', 'a', 'a', 'a', 'a']) == 2
compress2(['a']) == 1
compress2([]) == 0


# Given a 2D array containing only the following values: -1, 0, 1 where -1 represents an obstacle, 0represents a rabbit hole, and 1represents a rabbit, update every cell containing a rabbit with the distance to its closest rabbit hole.
#
# Note: multiple rabbit may occupy a single rabbit hole and you may assume every rabbit can reach a rabbit hole. A rabbit can only move up, down, left, or right in a single move. Ex: Given the following grid…
#
# -1  0  1
# 1  1 -1
# 1  1  0
# your grid should look like the following after running the function...
# -1  0  1
# 2  1 -1
# 2  1  0
#
# Ex: Given the following grid…
#
# 1  1  1
# 1 -1 -1
# 1  1  0
# your grid should look like the following after running the function...
# 4  5  6
# 3 -1 -1
# 2  1  0

# time complexity O(N^2)
# space complexity O(N^2) coming for recursion
def findShortestWay(grid):
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] == 1:
                grid[i][j] = float("inf")
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] == 0:
                dfs(i, j, 0, grid)

    return grid


def dfs(i, j, count, grid):
    if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[i]) or grid[i][j] == -1 or grid[i][j] < count:
        return

    grid[i][j] = count

    dfs(i + 1, j, count + 1, grid)
    dfs(i - 1, j, count + 1, grid)
    dfs(i, j + 1, count + 1, grid)
    dfs(i, j - 1, count + 1, grid)


#
#
#
grid = [[-1, 0, 1], [1, 1, -1], [1, 1, 0]]
findShortestWay([[-1, 0, 1], [1, 1, -1], [1, 1, 0]]) == [[-1, 0, 1], [2, 1, -1], [2, 1, 0]]


# This question is asked by Amazon. Given a valid IP address, defang it.
# Note: To defang an IP address, replace every ”.”, with ”[.]”.
#
# Ex: Given the following address…
#
# address = "127.0.0.1", return "127[.]0[.]0[.]1"

# O(n) time complexity
# O(n) space complexity
def defang(ip):
    result = []
    for c in ip:
        if c == '.':
            result.append("[")
            result.append(c)
            result.append("]")
        else:
            result.append(c)
    return ''.join(result)


defang("127.0.0.1") == "127[.]0[.]0[.]1"


# This question is asked by Google. Given an NxM matrix, grid, where each cell in the matrix represents the cost of stepping on the current cell, return the minimum cost to traverse from the top-left hand corner of the matrix to the bottom-right hand corner.
# Note: You may only move down or right while traversing the grid.
#
# Ex: Given the following grid…
#
# grid = [          y
#          x [1,1,3],
#            [2,3,1],
#          x [4,6,1]
#        ], return 7.
# The path that minimizes our cost is 1->1->3->1->1 which sums to 7.


# Time complexity = O(2 ^ K) exponential time where K is the total number of cells in our grid.
# Space complexity =
def minCost(grid):
    costs = []
    traverse(grid, 0, 0, 0, costs)

    minCost = float("inf")
    for cost in costs:
        minCost = min(cost, minCost)

    return minCost


def traverse(grid, x, y, cost, costs):    if


x >= len(grid) or y >= len(grid[0]):
return

currentCost = cost + grid[x][y]

if x == len(grid) - 1 and y == len(grid[0]) - 1:
    costs.append(currentCost)
    return

traverse(grid, x + 1, y, currentCost, costs)
traverse(grid, x, y + 1, currentCost, costs)

minCost([[1, 1, 3], [2, 3, 1], [4, 6, 1]]) == 7


# Improved version using DP
# Time complexity - O(M x N)
# Space complexity O(1)
def minCost2(grid):
    for i in range(1, len(grid)):
        grid[i][0] += grid[i - 1][0]
    for j in range(1, len(grid[0])):
        grid[0][j] += grid[0][j - 1]

    for i in range(1, len(grid)):
        for j in range(1, len(grid[i])):
            grid[i][j] = grid[i][j] + min(grid[i - 1][j], grid[i][j - 1])
    return grid[len(grid) - 1][len(grid[0]) - 1]


minCost2([[1, 1, 3], [2, 3, 1], [4, 6, 1]]) == 7


## Hamming distance
# x = 2, y = 4, return 2.
# 2 in binary is 0 0 1 0
# 4 in binary is 0 1 0 0
# therefore the number of positions in which the bits differ is two.


## Time complexity O(1) - all integers are 32 bits
## Space complexity O(1)
def hammingDistance(x, y):
    distance = 0
    while x > 0 or y > 0:
        distance += x % 2 ^ y % 2
        x >>= 1
        y >>= 1
    return distance


hammingDistance(2, 4) == 2


# This question is asked by Facebook. Given N points on a Cartesian plane, return the minimum time required to visit all points in the order that they’re given.
# Note: You start at the first point and can move one unit vertically, horizontally, or diagonally in a single second.
#
# Ex: Given the following points…
#
# points = [[0, 0], [1,1], [2,2]], return 2.
# In one second we can travel from [0, 0] to [1, 1]
# In another second we can travel from [1, 1,] to [2, 2]
# Ex: Given the following points…
#
# points = [[0, 1], [2, 3], [4, 0]], return 5.

# cartesian plane
# 1 0 0
# 0 1 0
# 0 0 1

# 0 1 0 0
# 0 0 0 0
# 0 0 0 1
# 1 0 0 0

# Time complexity O(n)
# Space complexity O(1)
def minTime(points):
    time = 0
    startX, startY = points.pop(0)
    while len(points) > 0:
        targetX, targetY = points.pop(0)
        xMove = abs(startX - targetX)
        yMove = abs(startY - targetY)
        howFarDiagonal = min(xMove, yMove)
        remainingHorizontalOrVertical = abs(yMove - xMove)
        time += howFarDiagonal + remainingHorizontalOrVertical

        startX = targetX
        startY = targetY

    return time


minTime([[0, 0], [1, 1], [2, 2]]) == 2
minTime([[0, 1], [2, 3], [4, 0]]) == 5


# This question is asked by Apple. Given an array of numbers, move all zeroes in the array to the end while maintaining the relative order of the other numbers.
# Note: You must modify the array you’re given (i.e. you cannot create a new array).
#
# Ex: Given the following array nums…
#
# nums = [3, 7, 0, 5, 0, 2], rearrange nums to look like the following [3,7,5,2,0,0]

#        i  j
# [3, 7, 5, 0, 0, 2]

# Time complexity O(N)
# Space complexity O(1)
def rearrange(nums):
    index = 0
    for i in range(len(nums)):
        if nums[i] != 0:
            nums[index] = nums[i]
            index += 1
    for i in range(index, len(nums)):
        nums[i] = 0


array = [3, 7, 0, 5, 0, 2]
rearrange(array)
array == [3, 7, 5, 2, 0, 0]


# Averages
# This question is asked by Facebook. Given a reference to the root of a binary tree, return a list containing the average value in each level of the tree.
#
# Ex: Given the following binary tree…
#
#      1
#     / \
#   6    8
#  / \
# 1   5
# return [1.0, 7.0, 3.0]

class Node:
    def __init__(self, value, left=None, right=None):
        self.left = left
        self.right = right
        self.value = value


# Time complexity: O(N)
# Space complexity: O(N)
def averages(root):
    averages = []
    queue = [root]
    while len(queue) > 0:
        currentLength = len(queue)
        currentSum = 0.0
        for _ in range(currentLength):
            current = queue.pop(0)
            if current.left:
                queue.append(current.left)
            if current.right:
                queue.append(current.right)
            currentSum += current.value
        averages.append(currentSum / currentLength)
    return averages


root = Node(1, Node(6, Node(1), Node(5)), Node(8))
averages(root) == [1.0, 7.0, 3.0]
root2 = Node(1, Node(6, Node(1), Node(5)), Node(8, Node(1), Node(1, Node(1))))
averages(root2) == [1.0, 7.0, 3.0]


# This question is asked by Amazon. Given an array of integers, nums, sort the array in any manner such that when i is even, nums[i] is even and when i is odd, nums[i] is odd.
# Note: It is guaranteed that a valid sorting of nums exists.
#
# Ex: Given the following array nums…
#
# nums = [1, 2, 3, 4], one possible way to sort the array is [2,1,4,3]

# Time complexity O(N)
# Space complexity O(1)
def sort(nums):
    for i in range(len(nums)):
        if i % 2 == 0:
            if nums[i] % 2 != 0:
                for j in range(i + 1, len(nums)):
                    if nums[j] % 2 == 0:
                        swap(nums, i, j)
        else:
            if nums[i] % 2 == 0:
                for j in range(i + 1, len(nums)):
                    if nums[j] % 2 != 0:
                        swap(nums, i, j)


# Time complexity O(N)
# Space complexity O(N)
def swap(arr, i, j):
    tmp = arr[i]
    arr[i] = arr[j]
    arr[j] = tmp


def sort2(nums):
    evenIdx = 0
    oddIdx = 1
    sorted = [-1 for _ in range(len(nums))]
    for i in range(len(nums)):
        if nums[i] % 2 == 0:
            sorted[evenIdx] = nums[i]
            evenIdx += 2
        else:
            sorted[oddIdx] = nums[i]
            oddIdx += 2
    return sorted


arr = [1, 2, 3, 4]
arr = [1, 2, 3, 4, 1, 6, 5, 8]
sort(arr)
arr == [2, 1, 4, 3]

sort2(arr)


# This question is asked by Facebook. Given an array nums, return whether or not its values are monotonically increasing or monotonically decreasing.
# Note: An array is monotonically increasing if for all values i <= j, nums[i] <= nums[j]. Similarly an array is monotonically decreasing if for all values i <= j, nums[i] >= nums[j].
#
# Ex: Given the following array nums…
#
# nums = [1, 2, 3, 4, 4, 5], return true.
# Ex: Given the following array nums…
#
# nums = [7, 6, 3], return true.
# Ex: Given the following array nums…
#
# nums = [8, 4, 6], return false.

def isMonotonic(nums):
    isIncreasing = True
    isDecreasing = True

    for i in range(1, len(nums)):
        if nums[i] < nums[i - 1]:
            isIncreasing = False
        if nums[i] > nums[i - 1]:
            isDecreasing = False
    return isIncreasing or isDecreasing


isMonotonic([1, 2, 3, 4, 4, 5]) == True
isMonotonic([7, 6, 3]) == True
isMonotonic([8, 4, 6]) == False


# Diving Deep
# This question is asked by Google. Given an N-ary tree, return its maximum depth.
# Note: An N-ary tree is a tree in which any node may have at most N children.
#
# Ex: Given the following tree…
#
#     4
#   / | \
#  3  9  2
# /        \
# 7          2
# return 3

class Node:
    def __init__(self, value, childrens=[]):
        self.value = value
        self.childrens = childrens


example1 = Node(4, [Node(3, [Node(7)]), Node(9), Node(2, [Node(2)])])
example2 = Node(4, [Node(3)])
example3 = Node(4)
example4 = Node(4, [Node(3, [Node(7)]), Node(9), Node(2, [Node(2, [Node(3)])])])


# Runtime: O(N) where N is the total number of nodes in our tree.
# Space complexity: O(N) where N is the total number of nodes in our tree. This extra space results from our recursion.
def maxDepth2(root):
    if not root:
        return 0

    currentMax = 0
    for c in root.childrens:
        currentMax = max(maxDepth2(c), currentMax)
    return currentMax + 1


def maxDepth(root):
    result = [0]
    maxDepthHelper(root, 1, result)
    return result[0]


def maxDepthHelper(root, current, result):
    if len(root.childrens) == 0:
        result[0] = max(current, result[0])
        return

    for c in root.childrens:
        maxDepthHelper(c, current + 1, result)


maxDepth(example1) == 3
maxDepth(example2) == 2
maxDepth(example3) == 1
maxDepth(example4) == 4

maxDepth2(example1) == 3
maxDepth2(None) == 0
maxDepth2(example2) == 2
maxDepth2(example3) == 1
maxDepth2(example4) == 4


# Reverse linked list
# Given a linked list, containing unique values, reverse it, and return the result.
#
# Ex: Given the following linked lists...
#
# 1->2->3->null, return a reference to the node that contains 3 which points to a list that looks like the following: 3->2->1->null
# 7->15->9->2->null, return a reference to the node that contains 2 which points to a list that looks like the following: 2->9->15->7->null
# 1->null, return a reference to the node that contains 1 which points to a list that looks like the following: 1->null

# p  c  n
# n<-c
#    1->2->3->null


def reverse(root):
    previous = None

    while root:
        next = root.next
        root.next = previous
        previous = root
        root = next

    return previous


class Node:
    def __init__(self, value, next=None):
        self.value = value
        self.next = next


#      p  r  n
# n <- 1->2->3->null

def reverse(root):
    prev = None
    while root:
        next = root.next
        root.next = prev
        prev = root
        root = next

    return prev


list1 = Node(1, Node(2, Node(3)))
reverse(list1)


# Given a list of words, return all the words that require only a single row of a keyboard to type.
# Note: You may assume that all words only contain lowercase alphabetical characters.
#
# Ex: Given the following list of words…
#
# words = ["two", "dad", "cat"], return ["two", "dad"].
# Ex: Given the following list of words…
#
# words = ["ufo", "xzy", "byte"], return [].


# Time complexity O(NxM)
# Space complexity O(N)
def keyboardRow(words):
    firstRow = "qwertyuiop"
    secondRow = "asdfghjkl"
    thirdRow = "zxcvbnm"
    map = {}

    for c in firstRow:
        map[c] = 0
    for c in secondRow:
        map[c] = 1
    for c in thirdRow:
        map[c] = 2

    result = []
    for w in words:
        row = map[w[0]]
        isValid = True
        for i in range(1, len(w)):
            if map[w[i]] != row:
                isValid = False
                break
        if isValid:
            result.append(w)

    return result


keyboardRow(["two", "dad", "cat"]) == ["two", "dad"]
keyboardRow(["ufo", "xzy", "byte"]) == []


# This question is asked by Microsoft. Design a class, MovingAverage, which contains a method, next that is responsible for returning the moving average from a stream of integers.
# Note: a moving average is the average of a subset of data at a given point in time.
#
#
# // i.e. the moving average has a capacity of 3.
# MovingAverage movingAverage = new MovingAverage(3);
# m.next(3) returns 3 because (3 / 1) = 3
# m.next(5) returns 4 because (3 + 5) / 2 = 4
# m.next(7) = returns 5 because (3 + 5 + 7) / 3 = 5

class MovingAverage():
    def __init__(self, size):
        self.queue = []
        self.sum = 0
        self.size = 0

    def next(self, value):
        self.queue.append(value)
        self.sum += value

        if self.size == len(self.queue):
            self.sum -= self.size.pop(0)
        else:
            self.size += 1
        return float(self.sum / self.size)


m = MovingAverage(3)
assert m.next(3) == 3
assert m.next(5) == 4
assert m.next(7) == 5

from collections import deque


# Runtime: All of our operations run in constant time except push() which is O(M) where M is the current number of elements in our QueueStack (because every time we add a new item we must iterate through our entire queue)
# Space complexity: O(N) where N is the total number of items we are allowed to hold in our queue.
class QueueStack:
    def __init__(self):
        self.queue = deque()

    def push(self, value):
        currentLength = len(self.queue)
        self.queue.append(value)

        for i in range(currentLength):
            self.queue.append(self.queue.popleft())

    def pop(self):
        if len(self.queue) == 0:
            return None
        return self.queue.popleft()

    def peek(self):
        if len(self.queue) == 0:
            return None
        return self.queue[0]


stack = QueueStack()

stack.push(3)
stack.push(4)

stack.peek() == None
stack.peek() == 4
stack.pop() == 4
stack.peek() == 3


# Alternating bits
# Given a positive integer N, return whether or not it has alternating bit values.
#
# Ex: Given the following value for N…
#
# N = 5, return true.
# 5 in binary is 101 which alternates bit values between 0 and 1.
# Ex: Given the following value for N…
#
# N = 8, return false
# 8 in binary is 1000 which does not alternate bit values between 0 and 1.


# Time complexity O(1) - always 32 bits
# Space complecity O(1)
def alternatingBits(num):
    previousBit = num % 2  # check if last bit
    num = num >> 1
    while num > 0:
        currentBit = num % 2
        if previousBit == currentBit:
            return False
        previousBit = currentBit
        num = num >> 1

    return True


alternatingBits(5) == True
alternatingBits(8) == False


# REVERSE NUMBER
# This question is asked by Apple. Given a 32 bit signed integer, reverse it and return the result.
# Note: You may assume that the reversed integer will always fit within the bounds of the integer data type.
#
# Ex: Given the following integer num…
#
# num = 550, return 55
# Ex: Given the following integer num…
#
# num = -37, return -73

# Time complexity O(1) - always 32 bits integer
# Space complexity O(1)
def reverseNumber(num):
    reversed = 0
    isNegative = num < 0
    num = abs(num)

    while num > 0:
        lastDigit = num % 10
        num //= 10
        reversed = (reversed * 10) + lastDigit
    return (-1) * reversed if isNegative else reversed


reverseNumber(550) == 55
reverseNumber(-37) == -73

# This question is asked by Facebook. Given a string, check if it can be modified such that no two adjacent characters are the same. If it is possible, return any string that satisfies this condition and if it is not possible return an empty string.
#
# Ex: Given the following string s…
#
# s = "abb", return "bab".

#  ij
# aab

# Ex: Given the following string s…
#
# s = "xxxy", return "" since it is not possible to modify s such that no two adjacent characters are the same.


# Greedy algorithm
# Runtime: O(NlogN) where N is the total number of elements in s. This results from adding our N elements to our heap and removing them all.
# Space complexity: O(N) where N is the total number of elements in s.
from queue import PriorityQueue


def noSameNeighbour(string):
    freqs = {}
    q = PriorityQueue()

    for c in string:
        if c in freqs:  # check if key exist in map
            freqs[c] += 1
        else:
            freqs[c] = 1

    for k, v in freqs.items():
        q.put(((-1) * v, k))

    result = []

    while q.qsize() > 1:
        mostFrequent = q.get()[1]
        print("current most frequent " + str(mostFrequent))
        secondMostFrequent = q.get()[1]
        print("current second most frequent " + str(secondMostFrequent))
        result.append(mostFrequent)
        result.append(secondMostFrequent)

        print("current freqs " + str(freqs))
        freqs[mostFrequent] = freqs[mostFrequent] - 1
        freqs[secondMostFrequent] = freqs[secondMostFrequent] - 1

        if freqs[mostFrequent] > 0:
            q.put(((-1) * freqs[mostFrequent], mostFrequent))
        if freqs[secondMostFrequent] > 0:
            q.put(((-1) * freqs[secondMostFrequent], secondMostFrequent))

    if q:
        last = q.get()[1]
        if freqs[last] > 1:
            return ""
        else:
            result.append(last)
    return "".join(result)


noSameNeighbour("abb") == "bab"
noSameNeighbour("xxxy") == ""
noSameNeighbour(None) == ""
noSameNeighbour("") == ""


# This question is asked by Facebook. Given the root of a binary tree and two values low and high return the sum of all values in the tree that are within low and high.
#
# Ex: Given the following tree where low = 3 and high = 5…
#
#          1
#         / \
#        7   5
#       /   / \
#     4    3   9
# return 12 (3, 4, and 5 are the only values within low and high and they sum to 12)


class Node:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right


# Time Complexity O(N)
# Space Complexity O(d) coming from recursion - O(N) in the worst case
def sumWithinBounds(root, low, high):
    if not root:
        return 0

    current = root.value if root.value >= low and root.value <= high else 0
    left = sumWithinBounds(root.left, low, high)
    right = sumWithinBounds(root.right, low, high)
    return left + right + current


root = Node(1, Node(7, Node(4)), Node(5, Node(3), Node(9)))
sumWithinBounds(root, 3, 5) == 12


# This question is asked by Google. Given an array, nums, and an integer k, return whether or not two unique indices exist such that nums[i] = nums[j] and the two indices i and jj are at most k elements apart. Ex: Given the following array nums and value k…
#         i
# nums = [1, 2, 1], k = 1, return false.
# Ex: Given the following array nums and value k…
#
# nums = [2, 3, 2], k = 2, return true.

# Time complexity O (n ^ 2 )
# Space complexity O(1)
def hasIdenticalElements(nums, k):
    for i in range(len(nums)):
        for j in range(i - k, i + k + 1):
            if j >= 0 and j < len(nums) and i != j and nums[i] == nums[j]:
                print("answer is:" + str(nums[i]) + " " + str(i) + " and " + str(nums[j]) + " " + str(j))
                return True
    return False


hasIdenticalElements([1, 2, 1], 1) == False
hasIdenticalElements([2, 3, 2], 2) == True


# O(N) time complexity
# O(N) space complexity
def hasIdenticalElements2(nums, k):
    dict = {}
    for i in range(len(nums)):
        if nums[i] in dict and i - dict[nums[i]] <= k:
            return True

        dict[nums[i]] = i
    return False


hasIdenticalElements2([1, 2, 1], 1) == False
hasIdenticalElements2([2, 3, 2], 2) == True


# Greedy algorithm
def numberOfRescueBoats(weights, limit):
    weights.sort()

    count = 0

    i = 0
    j = len(weights) - 1

    while i <= j:
        if weights[i] + weights[j] <= limit:
            i += 1
            j -= 1
        else:
            j -= 1
        count += 1

    return count


numberOfRescueBoats([1, 3, 5, 2], 5) == 3
numberOfRescueBoats([1, 2], 3) == 1
numberOfRescueBoats([4, 2, 3, 3], 5) == 3

##
# This question is asked by Amazon. You are given a group of stones, all of which have a positive weight. At each turn, we select the heaviest two stones and smash them together. When smashing these two stones together, one of two things can happen:
#
# If the stones are both the same weight, both stones are destroyed
# If the weights of the stones are not equal, the smaller of the two stones is destroyed and the remaining stone’s weight is updated to the difference (i.e. if we smash two stones together of weight 3 and weight 5 the stone with weight 3 is destroyed and the stone with original weight 5 now has weight 2).
# Continue smashing stones together until there is at most one stone left and return the weight of the remaining stone. If not stones remain, return zero.
from queue import PriorityQueue


# Time complexity O(N log N) This results from removing N items from our heap where each of the N removals costs O(logN) time (i.e. the height of our heap).
# Spece complexity O(N) This results from storing N elements in our max heap.
def throwingStones(stones):
    maxHeap = PriorityQueue()

    for stone in stones:
        maxHeap.put(stone * (-1))

    while maxHeap.qsize() > 1:
        firstStone = maxHeap.get() * (-1)
        secondStone = maxHeap.get() * (-1)

        if firstStone != secondStone:
            maxHeap.put((firstStone - secondStone) * (-1))

    return 0 if maxHeap.qsize() == 0 else maxHeap.get() * (-1)


throwingStones([2, 4, 3, 8]) == 1
throwingStones([1, 2, 3, 4]) == 0


# Memoization

## Find the winner
# This question is asked by Amazon. Given an integer array, two players take turns picking the largest number from the ends of the array. First, player one picks a number (either the left end or right end of the array) followed by player two. Each time a player picks a particular numbers, it is no longer available to the other player. This picking continues until all numbers in the array have been chosen. Once all numbers have been picked, the player with the larger score wins. Return whether or not player one will win.
# Note: You may assume that each player is playing to win (i.e. both players will always choose the maximum of the two numbers each turn) and that there will always be a winner.
#
# Ex: Given the following integer array...
#
# nums = [1, 2, 3], return true
# Player one takes 3
# Player two takes 2
# Player one takes 1
# 3 + 1 > 2 and therefore player one wins

# O(2 ^ N) without memoization
# O(N) with memoization

# Runtime: O(N ^ 2) where N is the total number of elements in nums.
# Space complexity: O(N ^ 2) where N is the total number of elements in nums. This results from our 2D matrix (which is size N x N).
def findWinner(nums):
    cache = [[0 for _ in range(len(nums))] for _ in range(len(nums))]

    return play(nums, 0, len(nums) - 1, cache) > 0


def play(nums, start, end, cache):
    if start == end:
        return nums[start]

    if cache[start][end] != 0:
        return cache[start][end]

    left = nums[start] - play(nums, start + 1, end, cache)
    right = nums[end] - play(nums, start, end - 1, cache)
    cache[start][end] = max(left, right)
    return cache[start][end]


findWinner([1, 2, 3]) == True


# Stairmaster
# This question is asked by Apple. Given a staircase where the ith step has a non-negative cost associated with it given by cost[i], return the minimum cost of climbing to the top of the staircase. You may climb one or two steps at a time and you may start climbing from either the first or second step.
#
# Ex: Given the following cost array…
#
# cost = [5, 10, 20], return 10.
#
# Ex: Given the following cost array…
#
# cost = [1, 5, 10, 3, 7, 2], return 10.
# Top Down approach
# Runtime: O(N) where N is the total number of elements in cost.
# Space complexity: O(N) where N is the total number of elements in cost. This extra space results in our memoize array.
def minCostStairs(costs):
    cache = [0 for _ in range(len(costs))]
    return climbSteps(len(costs) - 1, costs, cache)


def climbSteps(step, costs, cache):
    if step == 0 or step == 1:
        return costs[step]
    elif cache[step] != 0:
        return cache[step]
    else:
        cache[step] = costs[step] + min(climbSteps(step - 1, costs, cache), climbSteps(step - 2, costs, cache))
        return cache[step]


minCostStairs([1, 5, 10, 3, 7, 2]) == 10

# This question is asked by Amazon. Given two trees s and t return whether or not t is a subtree of s.
# Note: For t to be a subtree of s not only must each node’s value in t match its corresponding node’s value in s, but t must also exhibit the exact same structure as s. You may assume both trees s and t exist.
#
# Ex: Given the following trees s and t…
#
# s = 1
#    / \
#   3   8
#
# t = 1
#     \
#      8
# return false
s1 = Node(1, Node(3), Node(8))
t1 = Node(1, None, Node(8))

#
# Ex: Given the following trees s and t…
#
# s = 7
#    / \
#   8   3
#
# t = 7
#    / \
#   8   3
# return true
#

s2 = Node(7, Node(8), Node(3))
t2 = Node(7, Node(8), Node(3))

# Ex: Given the following trees s and t…
#
# s = 7
#    / \
#    8   3
#
# t = 7
#    / \
#   8   3
#       /
#      1
# return false

s3 = Node(7, Node(8), Node(3))
t3 = Node(7, Node(8), Node(3, Node(1)))

# Ex: Given the following trees s and t…
#
# s = 1
#    / \
#   8   7
#      / \
#      8  3
#
# t = 7
#    / \
#    8   3

s4 = Node(1, Node(8), Node(7, Node(8), Node(3)))
t4 = Node(7, Node(8), Node(3))


class Node:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right


# Time complexity O(N*M)
# Space complexity O(N) where N is the number of nodes in s because in the worst case our recursion can go N levels deep.
def isSubTree(s, t):
    if not s:
        return False
    if s.value == t.value:
        return isSameTree(s, t)
    else:
        return isSubTree(s.left, t) or isSubTree(s.right, t)


def isSameTree(s, t):
    if not s and not t:
        return True
    if not s or not t:
        return False
    if s.value != t.value:
        return False
    left = isSameTree(s.left, t.left)
    right = isSameTree(s.right, t.right)
    return left and right


s1 = Node(1, Node(3), Node(8))
t1 = Node(1, None, Node(8))

isSubTree(s1, t1) == False

s2 = Node(7, Node(8), Node(3))
t2 = Node(7, Node(8), Node(3))
isSubTree(s2, t2) == True

s3 = Node(7, Node(8), Node(3))
t3 = Node(7, Node(8), Node(3, Node(1)))
isSubTree(s3, t3) == False

s4 = Node(1, Node(8), Node(7, Node(8), Node(3)))
t4 = Node(7, Node(8), Node(3))
isSubTree(s4, t4) == True


# This question is asked by Amazon. A frog is attempting to cross a river to reach the other side. Within the river, there are stones located at different positions given by a stones array (this array is in sorted order). Starting on the first stone (i.e. stones[0]), the frog makes a jump of size one potentially landing on the next stone. If the frog’s last jump was of size x, the frog’s next jump may be of size x - 1, x, or x + 1. Given these following conditions return whether or not the frog can reach the other side.
# Note: The frog may only jump in the forward direction.
#
# Ex: Given the following stones…
#
# stones = [0, 1, 10], return false.
# The frog can jump from stone 0 to stone 1, but then the gap is too far to jump to the last stone (i.e. the stone at position 10)
# Ex: Given the following stones…
#
# stones = [0, 1, 2, 4], return true.
# The frog can jump from stone 0, to stone 1, to stone 2, to stone 4.
# Runtime: O(N2) where N is the number of stones we’re given.
# Space complexity: O(N2) where N is the number of stones we’re given (this results from our memoize matrix).
def riverCrossing(stones):
    cache = [[-1 for _ in range(len(stones))] for _ in range(len(stones))]

    return canCross(stones, 0, 0, cache) == 1


def canCross(stones, start, jump, cache):
    if cache[start][jump] != -1:
        return cache[start][jump]

    for i in range(start + 1, len(stones)):
        distance = stones[i] - stones[start]
        if distance >= jump - 1 and distance <= jump + 1 and canCross(stones, i, distance, cache):
            cache[start][jump] = 1
            return 1

    cache[start][jump] = 1 if start == len(stones) - 1 else 0
    return cache[start][jump]


riverCrossing([0, 1, 10]) == False
riverCrossing([0, 1, 2, 4]) == True


# Longest common subsequence
def longestCommonSubsequence(text1, text2):
    """
    :type text1: str
    :type text2: str
    :rtype: int
    """
    #   - a c  e
    # - - - -  -
    # a - a a  a
    # b - a a  a
    # c - a ac ac
    # d - a ac ac
    # e - a ac ace

    #   - a c  c
    # - - - -  -
    # a - a a  a
    # b - a a  a
    # c - a ac ac
    # d - a ac
    # e - a ac

    longestCommonString = [[0 for _ in range(len(text1) + 1)] for _ in range(len(text2) + 1)]

    for i in range(len(text2)):
        for j in range(len(text1)):
            if text1[j] == text2[i]:
                longestCommonString[i + 1][j + 1] = longestCommonString[i][j] + 1
            else:
                longestCommonString[i + 1][j + 1] = max(longestCommonString[i][j + 1], longestCommonString[i + 1][j])
    return longestCommonString[len(text2)][len(text1)]


# Dynamic Programming

# Word segmentation
# This question is asked by Amazon. Given a string s and a list of words representing a dictionary, return whether or not the entirety of s can be segmented into dictionary words.
# Note: You may assume all characters in s and the dictionary are lowercase.
#
# Ex: Given the following string s and dictionary…
#      j   i
# s = "thedailybyte", dictionary = ["the", "daily", "byte"], return true.
#    [TFFT]
# Ex: Given the following string s and dictionary…
#
# s = "pizzaplanet", dictionary = ["plane", "pizza"], return false.


# Painting Houses
# This question is asked by Apple. You are tasked with painting a row of houses in your neighborhood such that each house is painted either red, blue, or green. The cost of painting the ith house red, blue or green, is given by costs[i][0], costs[i][1], and costs[i][2] respectively. Given that you are required to paint each house and no two adjacent houses may be the same color, return the minimum cost to paint all the houses.
#
# Ex: Given the following costs array…
#
# costs = [[1, 3, 5],[2, 4, 6],[5, 4, 3]], return 8.
# Paint the first house red, paint the second house blue, and paint the third house green.

#  R B G
# [1,3,5]
# [2,4,6]
# [5,4,3]

# Runtime: O(N) where N is the total number of houses we are given (i.e. costs.length).
# Space complexity: O(1) or constant since we are reusing the costs matrix we are given in the problem.
def minCost(costs):
    for i in range(1, len(costs)):
        costs[i][0] += min(costs[i - 1][1], costs[i - 1][2])
        costs[i][1] += min(costs[i - 1][0], costs[i - 1][2])
        costs[i][2] += min(costs[i - 1][0], costs[i - 1][1])

    return min(costs[len(costs) - 1][0], costs[len(costs) - 1][1], costs[len(costs) - 1][2])


minCost([[1, 3, 5], [2, 4, 6], [5, 4, 3]]) == 8


## Art thief
# You’ve broken into an art gallery and want to maximize the value of the paintings you steal. All the paintings you steal you place in your bag which can hold at most W pounds. Given that the weight and value of the ith painting is given by weights[i] and values[i] respectively, return the maximum value you can steal.
#
# Ex: Given the following W, weights array and values array…
#
# W = 10, weights = [4, 1, 3], values = [4, 2, 7], return 13.
#
# Ex: Given the following W, weights array and values array…
#
# W = 5, weights = [2, 4, 3], values = [3, 7, 2], return 7.
#
# Ex: Given the following W, weights array and values array…
#
# W = 7, weights = [1, 3, 4], values = [3, 5, 6], return 11.

# NOT OPTIMAL
# One way to solve our problem would be to simulate choosing and not choosing each painting. By doing so we could tally up the total value each different set of possibilities results in and return the result that yields the largest value such that the weight of all the paintings we steal is at most W. While this would work it’d result in a rather slow runtime of O(2N) where N is the total number of paintings (i.e. weights.length/values.length). This results from having two choices for each of our N paintings: steal the current painting or don’t steal the current painting.

# OPTIMAL WITH DP

# row represents max value up to j-th painting
# column represents current weight of bag from 0 to N

# artThief(W = 5, weights = [2, 4, 3], values = [3, 7, 2]) == 7

#   0 1 2 3
# 0 0 0 0 0
# 1 0 0 0 0
# 2 0 3 3 3
# 3 0 3 3 3
# 4 0 3 7 7
# 5 0 3 7 7


# Runtime: O(N * M) where N is the length of weights and M is the length of values.
# Space complexity: O(N * M) where N is the length of weights and M is the length of values. This results from our dp array.
def artThief(W, weights, values):
    dp = [[0 for _ in range(len(values) + 1)] for _ in range(W + 1)]

    for i in range(len(dp)):
        for j in range(len(dp[i])):
            if i == 0 or j == 0:
                dp[i][j] = 0
            elif i < weights[j - 1]:
                dp[i][j] = dp[i][j - 1]
            else:
                dp[i][j] = max(values[j - 1] + dp[i - weights[j - 1]][j - 1], dp[i][j - 1])

    return dp[W][len(values)]


artThief(W=10, weights=[4, 1, 3], values=[4, 2, 7]) == 13
artThief(W=5, weights=[2, 4, 3], values=[3, 7, 2]) == 7
artThief(W=7, weights=[1, 3, 4], values=[3, 5, 6]) == 11


## Birthday cake
# This question is asked by Amazon. You are at a birthday party and are asked to distribute cake to your guests. Each guess is only satisfied if the size of the piece of cake they’re given, matches their appetite (i.e. is greater than or equal to their appetite). Given two arrays, appetite and cake where the ithelement of appetite represents the ith guest’s appetite, and the elements of cake represents the sizes of cake you have to distribute, return the maximum number of guests that you can satisfy.
#
# Ex: Given the following arrays appetite and cake…
#
# appetite = [1, 2, 3], cake = [1, 2, 3], return 3.
# Ex: Given the following arrays appetite and cake…
#
# appetite = [3, 4, 5], cake = [2], return 0.

# Runtime: O(NlogN) where N is the total number of guests we need to serve (i.e. appetite.length). This results from sorting both our cake and our guests’ appetite (which in the worst case will be the same length).
# Space complexity: O(1) or constant since the arrays that we sort are already given to us as parameters.
def countSatisfied(appetatite, cake):
    appetatite.sort()
    cake.sort()

    counter = 0
    i = 0
    j = 0
    while j < len(cake) and i < len(appetatite):
        currentCake = cake[j]
        if currentCake >= appetatite[i]:
            counter += 1
            i += 1
        j += 1

    return counter


countSatisfied([1, 2, 3], [1, 2, 3]) == 3
countSatisfied([3, 4, 5], [2]) == 0


## Max sum increasing subsequence


# ## Divisible digits
# Given an integer N, return the total number self divisible numbers that are strictly less than N (starting from one).
# Note: A self divisible number if a number that is divisible by all of its digits.
#
# Ex: Given the following value of N…
#
# N = 17, return 12 because 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 15 are all self divisible numbers.

# Time complexity: O(N)
# Space complexity: O(1)
def divisibleDigits(N):
    counter = 0
    for i in range(1, N):
        if hasDivisibleDigits(i):
            counter += 1
    return counter


def hasDivisibleDigits(num):
    currentNum = num
    while currentNum != 0:
        lastDigit = currentNum % 10
        if lastDigit == 0 or num % lastDigit != 0:
            return False
        currentNum = currentNum // 10
    return True


divisibleDigits(17) == 12


## Implement Trie
# This question is asked by Microsoft. Implement a trie class that supports insertion and search functionalities.
# Note: You may assume only lowercase alphabetical characters will added to your trie.
#
# Ex: Given the following operations on your trie…
#
# Trie trie = new Trie()
# trie.insert("programming");
# trie.search("computer") // returns false.
# trie.search("programming") // returns true.


# Runtime of insert and search: O(N) where N is the length of the string we’re inserting or searching for.
# Space complexity of insert and search: O(N) where N is the length of the string and O(1) respectively.
class TrieNode:
    def __init__(self, char):
        self.char = char
        self.isEnd = False
        self.children = {}


class Trie:
    def __init__(self):
        self.root = TrieNode("")

    def insert(self, word):
        node = self.root

        for c in word:
            if c in node.children:
                node = node.children[c]
            else:
                newNode = TrieNode(c)
                node.children[c] = newNode
                node = newNode
        node.isEnd = True

    def search(self, word):
        current = self.root
        for c in word:
            if c not in current.children:
                return False
            current = current.children[c]
        return current.isEnd


trie = Trie()
trie.insert("programming")
trie.search("computer") == False
trie.search("programming") == True
trie.search("programmingg") == False


# # Pond size
# You are given two-dimensional matrix that represents a plot of land. Within the matrix there exist two values: ones which represent land and zeroes which represent water within a pond. Given that parts of a pond can be connected both horizontally and vertically (but not diagonally),
# return the largest pond size.
# Note: You may assume that each zero within a given pond contributes a value of one to the total size of the pond.
#
# Ex: Given the following plot of land…
#
# land = [
#            [1,1,1],
#            [1,0,1],
#            [1,1,1]
#        ], return 1.
# Ex: Given the following plot of land…
#
# land = [
#            [1,0,1],
#            [0,0,0],
#            [1,0,1]
#        ], return 5.

# DFS
# Time complexity O(M x N) or O(N) where N is total number of cells
# Space complexity O(M x N) or O(N)  where N is total number of cells
def maxPond(grid):
    largestPondSize = 0

    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] == 0:
                size = explorePond(grid, i, j)
                largestPondSize = max(largestPondSize, size)

    return largestPondSize


def explorePond(grid, i, j):
    if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[i]) or grid[i][j] == 1:
        return 0

    grid[i][j] = 1
    up = explorePond(grid, i - 1, j)
    right = explorePond(grid, i, j + 1)
    down = explorePond(grid, i + 1, j)
    left = explorePond(grid, i, j - 1)
    return up + right + down + left + 1


maxPond([[1, 1, 1], [1, 0, 1], [1, 1, 1]]) == 1
maxPond([[1, 0, 1], [0, 0, 0], [1, 0, 1]]) == 5


'''
Time complexity of classic dijkstra : O(E log V)

Time: O((|V| + |E|) log V)
Space: O(|V| + |E|)
However, (E >= V - 1) so |V| + |E| ==> |E|. But usually we use both V and E
'''
def findShortestPath(start, edges):
    from queue import PriorityQueue

    minHeap = PriorityQueue()
    minHeap.put((0, start))
    minDistances = [float("inf") for _ in range(len(edges))]
    minDistances[start] = 0

    while minHeap.qsize() > 0:
        distance, v = minHeap.get()

        for edge in edges[v]:
            destination, weight = edge

            currentDistance = minDistances[destination]
            newDistance = distance + weight
            if newDistance < currentDistance:
                minDistances[destination] = newDistance
                minHeap.put((newDistance ,destination))

    return [-1 if x == float("inf") else x for x in minDistances]

start = 0
edges = [[[1, 7]], [[2, 6], [3, 20], [4, 3]], [[3, 14]], [[4, 2]], [], []]
findShortestPath(start, edges) == [0, 7, 13, 27, 10, -1]

# K-th largest number
# Given an array of integers, nums, and a value k, return the kth largest element.
#
# Ex: Given the following array nums…
#
#
# [1, 2, 3], k = 1, return 3.
# Ex: Given the following array nums…
#
# [9, 2, 1, 7, 3, 2], k = 5, return 2.


# Time complexity O(n log(n))
# Space complexity O(n)
# One approach t solve this problem would simply be to sort our array and return the kth to last element in the array (assuming our array is sorted in ascending order).
# This is a valid solution and it’d run in O(NlogN) time where N is the total number of elements in nums (this overhead results from sorting our array nums). While this solution works, it can be optimized slightly (in the average case).

# Runtime: O(NlogN) where N is the total number of elements in nums (our optimization will help us knock the average case down to O(Nlogk) but in the worse case if k is equal to N our runtime degrades back to O(NlogN).
# Space complexity: O(N). This results from storing all N elements in our heap in the worst case (i.e. when k is equal to N).

# [1,2,3], k=1
# [1] add first element
# [2] remove min add next
# [3] remove min add next
# return
from queue import PriorityQueue


def kLargestNumber(nums, k):
    minHeap = PriorityQueue()
    for num in nums:
        if minHeap.qsize() < k:
            minHeap.put(num)
        else:
            minHeap.get()
            minHeap.put(num)
    return minHeap.get()


kLargestNumber([9, 2, 1, 7, 3, 2], k=5) == 2
kLargestNumber([1, 2, 3], k=1) == 3


## Topological sort - DFS appraoch
## Time complexity O(V+E)
## Space complexity O(V+E)
def topologicalSort(jobs, deps):
    visited = [False for _ in range(len(jobs) + 1)]
    visiting = [False for _ in range(len(jobs) + 1)]
    prereq = {}

    for dep in deps:
        first, second = dep
        if second not in prereq:
            prereq[second] = [first]
        else:
            prereq[second].append(first)

    orderedJobs = []
    for node in jobs:
        containsCycle = traverse(node, visited, visiting, orderedJobs, prereq)
        if containsCycle:
            return []
    return orderedJobs


def traverse(node, visited, visiting, orderedJobs, prereq):
    if visited[node] == True:
        return False
    if visiting[node] == True:
        return True

    visiting[node] = True
    if node in prereq:
        for p in prereq[node]:
            containsCycle = traverse(p, visited, visiting, orderedJobs, prereq)
            if containsCycle:
                return True
    visiting[node] = False
    visited[node] = True
    orderedJobs.append(node)
    return False


jobs = [1, 2, 3, 4]
deps = [[1, 2], [1, 3], [3, 2], [4, 2], [4, 3]]

topologicalSort(jobs, deps) == [1, 4, 3, 2] or topologicalSort(jobs, deps) == [4, 1, 3, 2]

jobs = [1, 2, 3, 4]
deps = [[1, 2], [1, 3], [3, 2], [4, 2], [4, 3], [3, 4]]  # contains cycle
topologicalSort(jobs, deps) == []


# STRINGS

## Reverse string
def reverseString(value):
    chars = []
    for idx in range(len(value) - 1, -1, -1):
        chars.append(value[idx])
    return "".join(chars)
    # return value[::-1]


reverseString("Cat") == "taC"
reverseString("The Daily Byte") == "etyB yliaD ehT"


## Ceasar cipher encryptor
# time complexity O(N)
# space complexity O(N)
def encrypt(string, key):
    newKey = key % 26
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    encrypted = []
    for c in string:
        newChar = shift(c, key, alphabet)
        encrypted.append(newChar)

    return "".join(encrypted)


def shift(c, key, alphabet):
    charIdx = alphabet.index(c)
    newIdx = (charIdx + key) % 26
    return alphabet[newIdx]


encrypt("xyz", 2) == "zab"


#           i
#               j
# "The Daily Byte"
## Swap words


# Time complexity O(N)
# Space complexity (O(N))
def swapWords(string):
    swapped = []
    findWords(string, swapped)
    return " ".join(swapped)


def findWords(string, swapped):
    i = len(string) - 1
    j = len(string)

    while i >= 0:
        if string[i] == " " or i == 0:
            word = string[i + 1:j] if i != 0 else string[i:j]
            if len(word) > 0:
                swapped.append(word)
            j = i
            i = j - 1
        else:
            i -= 1


assert swapWords("The Daily Byte") == "Byte Daily The"


# non-constructible change
def nonConstructibleChange(coins):
    currentChange = 0
    coins.sort()
    for coin in coins:
        if coin > currentChange + 1:
            return currentChange + 1
        currentChange += coin

    return currentChange + 1


#     i
# 1,1,2,3,5, 7, 22
# 1 2 4 7 12,19    1
assert nonConstructibleChange([5, 7, 1, 1, 2, 3, 22]) == 20


# three number sum

# Time complexity O(n^2)
# Space complexity O(N)
def threeNumSum(array, target):
    triplets = []
    array.sort()
    for i in range(len(array) - 2):
        left = i + 1
        right = len(array) - 1
        while left < right:
            currentSum = array[i] + array[left] + array[right]
            if currentSum == target:
                triplets.append([array[i], array[left], array[right]])
                left += 1
                right -= 1
            elif currentSum > target:
                right -= 1
            elif currentSum < target:
                left += 1
    return triplets

    return triplets


#   c  i            j
# [-8,-6,1,2,3,5,6,12]
assert threeNumSum([12, 3, 1, 2, -6, 5, -8, 6], 0) == [[-8, 2, 6], [-8, 3, 5], [-6, 1, 5]]


# Time complexity O(NlogN + MlogM)
# Space complexity O(1)
def smallestDifference(arrOne, arrTwo):
    arrOne.sort()
    arrTwo.sort()
    idxOne = 0
    idxTwo = 0
    smallestDiff = float("inf")
    current = float("inf")
    smallestPair = []
    while idxOne < len(arrOne) and idxTwo < len(arrTwo):
        firstNum = arrOne[idxOne]
        secondNum = arrTwo[idxTwo]
        if firstNum > secondNum:
            current = firstNum - secondNum
            idxTwo += 1
        elif secondNum > firstNum:
            current = secondNum - firstNum
            idxOne += 1
        else:
            return [firstNum, secondNum]

        if current < smallestDiff:
            smallestDiff = current
            smallestPair = [firstNum, secondNum]

    return smallestPair


#        i
# [-1 , 3, 5, 10, 20, 28]
# [15, 17, 26, 134, 134]
#   j
arrOne = [-1, 5, 10, 20, 28, 3]
arrTwo = [26, 134, 135, 15, 17]

assert smallestDifference(arrOne, arrTwo) == [28, 26]


# Time complextity O(n)
# Space complexity O(1)
def isMonotonic(nums):
    isIncreasing = True
    isDecreasing = True
    for i in range(1, len(nums)):
        if nums[i] > nums[i - 1]:
            isDecreasing = False
        if nums[i] < nums[i - 1]:
            isIncreasing = False

        if not isIncreasing and not isDecreasing:
            return False

    return isIncreasing or isDecreasing


assert isMonotonic([-1, -5, -10, -1100, -1100, -1101, -1102, -9001]) == True


def longestPeak(nums):
    maxPeak = 0
    for i in range(1, len(nums) - 1):
        left = i - 1
        right = i + 1

        if nums[i] > nums[left] and nums[right] > nums[i]:
            currentPeak = explore(nums, left, right)
            if currentPeak > maxPeak:
                maxPeak = currentPeak

    return maxPeak


def explore(nums, left, right):
    while left > 0 and nums[left - 1] < nums[left]:
        left -= 1
    while right < len(nums) - 1 and nums[right + 1] > nums[right]:
        right += 1

    return right - left + 1


assert longestPeak([1, 2, 3, 3, 4, 0, 10, 6, 5, -1, -3, 2, 3]) == 6


# Product of values
# Given an integer array nums, return an array where each element i represents the product of all values in nums excluding nums[i].
# Note: You may assume the product of all numbers within nums can safely fit within an integer range.
#
# Ex: Given the following array nums…
#
# nums = [1, 2, 3], return [6,3,2].
# 6 = 3 * 2 (we exclude 1)
# 3 = 3 * 1 (we exclude 2)
# 2 = 2 * 1 (we exclude 3)

# [1 1 2]
# 1 2 3

# Time complexity O(N^2)
# Space complexity O(N)
def products(nums):
    products = [1 for _ in range(len(nums))]
    for i in range(len(nums)):
        for j in range(len(nums)):
            if i != j:
                products[i] *= nums[j]

    return products


# Time complexity O(N)
# Space complexity O(N)
def products(nums):
    products = [1 for _ in range(len(nums))]

    leftRunning = 1
    for i in range(len(nums)):
        products[i] = leftRunning
        leftRunning *= nums[i]

    rightRunning = 1
    for i in reversed(range(len(nums))):
        products[i] *= rightRunning
        rightRunning *= nums[i]
    return products


assert products([1, 2, 3]) == [6, 3, 2]


# Given the reference to the root of a binary tree and a value k, return the number of paths in the tree such that the sum of the nodes along the path equals k.
# Note: The path does not need to start at the root of the tree, but must move downward.
#
# Ex: Given the following binary tree and value k…
#
#        3
#      /  \
#     1   8
# k = 11, return 1 (3 -> 8).
# Input : k = 5
# Root of below binary tree:
#         1
#        / \
#      3    -1
#    /  \    /  \
#   2   1   4    5
#       /  / \   \
#      1  1  2    6
#
# Output :
# 3 2
# 3 1 1
# 1 3 1
# 4 1
# 1 -1 4 1
# -1 4 2
# 5
# 1 -1 5


# Time complexity: O(4 ^ N)
# Space complexity: O(N)
def countPaths(root, k):
    counter = [0]
    seqs = []
    exploreTree(root, k, 0, counter, [], seqs)
    print(seqs)
    return counter[0]


def exploreTree(root, k, currentSum, counter, currentSeq, seqs):
    currentSum = currentSum + root.value
    currentSeq.append(root.value)

    if currentSum == k:
        seqs.append(currentSeq)
        counter[0] += 1

    if root.left:
        exploreTree(root.left, k, currentSum, counter, currentSeq[:], seqs)
        if len(currentSeq) < 2:
            exploreTree(root.left, k, 0, counter, [], seqs)
    if root.right:
        exploreTree(root.right, k, currentSum, counter, currentSeq[:], seqs)
        if len(currentSeq) < 2:
            exploreTree(root.right, k, 0, counter, [], seqs)


root = Node(3, Node(1), Node(8))
k = 11
assert countPaths(root, k) == 1

root = Node(1, Node(3, Node(2), Node(1, Node(1))), Node(-1, Node(4, Node(1), Node(2)), Node(5, None, Node(6))))
k = 5
assert countPaths(root, k) == 8


## Sum of Linked Lists

class Node:
    def __init__(self, value, next=None):
        self.value = value
        self.next = next


def sumOfLinkedLists(rootOne, rootTwo):
    resultNode = Node(-1)
    resultPointer = resultNode
    remainder = 0
    while rootOne or rootTwo:
        firstValue = rootOne.value if rootOne else 0
        secondValue = rootTwo.value if rootTwo else 0
        result = firstValue + secondValue + remainder

        current = result % 10
        currentNode = Node(current)

        resultPointer.next = currentNode
        resultPointer = currentNode

        remainder = result // 10
        rootOne = rootOne.next if rootOne else None
        rootTwo = rootTwo.next if rootTwo else None

    return resultNode.next


# 1 7 4 2
#   5 4 9
#       1
rootOne = Node(2, Node(4, Node(7, Node(1))))
rootTwo = Node(9, Node(4, Node(5)))

result = Node(1, Node(9, Node(2, Node(2))))

assert sumOfLinkedLists(rootOne, rootTwo) == result


## Bottom of the Barell
# Given a binary tree, return the bottom-left most value.
# Note: You may assume each value in the tree is unique.
# Ex: Given the following binary tree…
#
#        1
#       / \
#     2   3
#    /
#   4
# return 4.
# Ex: Given the following binary tree…
#
#    8
#   / \
#  1   4
#     /
#    2
# return 2.

class Node:
    def __init__(self, value, left=None, right=None):
        self.left = left
        self.right = right
        self.value = value


# Time complexity O(N)
# Space complexity O(N)
def bottomLeft(root):
    queue = [root]
    bottomLeftValue = root.value

    while len(queue) > 0:
        currentLength = len(queue)

        for i in range(currentLength):
            currentNode = queue.pop(0)
            if i == 0:
                bottomLeftValue = currentNode.value
            if currentNode.left:
                queue.append(currentNode.left)
            if currentNode.right:
                queue.append(currentNode.right)
    return bottomLeftValue


assert bottomLeft(Node(1, Node(2, Node(4)), Node(3))) == 4
assert bottomLeft(Node(8, Node(1, Node(2)), Node(4))) == 2

# String Repetition
# Given a string s, return all of its repeated 10 character substrings.
# Note: You may assume s only contains uppercase alphabetical characters.
#
# Ex: Given the following string s…
#
# s = "BBBBBBBBBBB", return ["BBBBBBBBBB"].
# Ex: Given the following string s…
#      0        1           21
# s = "ABABABABABABBBBBBBBBBB", return ["ABABABABAB","BBBBBBBBBB"].

# 22 -13 = 9
# Time complexity O(N)
# Space complexity O(N)
def findRepetitions(input):
    if not input or len(input) < 10:
        return []

    repetitions = []
    seen = {}
    for i in range(len(input) - 9):
        currentSubstring = input[i:i + 10]
        if currentSubstring in seen and seen[currentSubstring] == 1:
            repetitions.append(currentSubstring)
            seen[currentSubstring] += 1
        else:
            seen[currentSubstring] = 1
    return repetitions


assert findRepetitions("BBBBBBBBBBB") == ["BBBBBBBBBB"]
assert findRepetitions("ABABABABABABBBBBBBBBBB") == ["ABABABABAB", "BBBBBBBBBB"]


# Next greater element
# Given two arrays of numbers, where the first array is a subset of the second array, return an array containing all the next greater elements for each element in the first array, in the second array. If there is no greater element for any element, output -1 for that number.
#
# Ex: Given the following arrays…
#
# nums1 = [4,1,2], nums2 = [1,3,4,2], return [-1, 3, -1] because no element in nums2 is greater than 4, 3 is the first number in nums2 greater than 1, and no element in nums2 is greater than 2.
# nums1 = [2,4], nums2 = [1,2,3,4], return [3, -1] because 3 is the first greater element that occurs in nums2 after 2 and

# Naive approach
# Time complexity O(M*N)

def nextGreaterElement(nums1, nums2):
    nextGreaterElements = []
    for i in range(len(nums1)):
        foundElement = False
        for j in range(len(nums2)):
            if nums1[i] == nums2[j]:
                for k in range(j + 1, len(nums2)):
                    if nums2[k] > nums2[j]:
                        nextGreaterElements.append(nums2[k])
                        foundElement = True
                        break
        if not foundElement:
            nextGreaterElements.append(-1)

    return nextGreaterElements


# Time complexity O(N+M)
# Space complexity O(N +M)
def nextGreaterElement(nums1, nums2):
    stack = []
    nextGreater = {}
    result = []
    for num in nums2:
        if len(stack) > 0 and stack[-1] < num:
            nextGreater[stack.pop()] = num
        else:
            stack.append(num)

    while (len(stack) > 0):
        nextGreater[stack.pop()] = -1

    for num in nums1:
        result.append(nextGreater[num])
    return result


assert nextGreaterElement(nums1=[4, 1, 2], nums2=[1, 3, 4, 2]) == [-1, 3, -1]


# Given the reference to a binary search tree and a value to insert, return a reference to the root of the tree after the value has been inserted in a position that adheres to the invariants of a binary search tree.
# Note: It is guaranteed that each value in the tree, including the value to be inserted, is unique.
#
# Ex: Given the following tree and value…
#
#       2
#      / \
#     1   3
# value = 4, return the following tree...
#       2
#      / \
#     1   3
#          \
#           4

class Node:
    def __init__(self, value, left=None, right=None):
        self.left = left
        self.right = right
        self.value = value


def nextValue(root, value):
    if not root:
        return Node(value)
    elif root.value > value:
        root.left = nextValue(root.left, value)
    elif root.value < value:
        root.right = nextValue(root.right, value)
    else:
        left = root.left
        root.left = Node(value)
        if left:
            root.left.left = left.left
            root.left.right = left.right
    return root


root = Node(2, Node(1), Node(3))
value = 4
assert nextValue(root, value).right.right.value == 4

root = Node(2, Node(1), Node(3))
value = 3
assert nextValue(root, value).right.left.value == 3


# Setting sail
# You’re about to set sail off a pier and first want to count the number of ships that are already in the harbor. The harbor is deemed safe to sail in if the number of boats in the harbor is strictly less than limit. Given a 2D array that presents the harbor, where O represents water and S represents a ship, return whether or not it’s safe for you to set sail.
# Note: All ships in the harbor can only lie entirely vertically or entirely horizontally and cannot be connected to another ship.
#
# Ex: Given the following 2D array harbor and value limit…
#
# harbor = [
#              [O, O, S],
#              [S, O, O],
#              [O, O, S]
#          ], limit = 5, return true. You setting sail would cause there to be 4 ships in the harbor which is under the limit of 5.
#
# Ex: Given the following 2D array harbor and value limit…
#
# harbor = [
#              [O, O, O],
#              [S, O, S],
#              [O, O, S]
#          ], limit = 3, return false. The harbor is not safe to sail in since you setting sail would cause the number of boats in the harbor to reach the limit.


# Time complexity - O(M * N) - number of vertices + number of edges
# Space complexity - O(M * N)
def settingSail(harbor, limit):
    boatCounter = 0

    for i in range(len(harbor)):
        for j in range(len(harbor[i])):
            if harbor[i][j] == 'S':
                explore(harbor, i, j)
                boatCounter += 1
    return boatCounter + 1 < limit


def explore(harbor, i, j):
    if i < 0 or i > len(harbor) - 1 or j < 0 or j > len(harbor[i]) - 1 or harbor[i][j] == 'O':
        return
    harbor[i][j] = 'O'
    explore(harbor, i + 1, j)
    explore(harbor, i - 1, j)
    explore(harbor, i, j - 1)
    explore(harbor, i, j + 1)


assert settingSail([['O', 'O', 'S'], ['S', 'O', 'O'], ['O', 'O', 'S']], 5) == True
assert settingSail([['O', 'O', 'O'], ['S', 'O', 'S'], ['O', 'O', 'S']], 3) == False
assert settingSail([['O']], 1) == False
assert settingSail([['S']], 1) == False


# String permutations
# This question is asked by Amazon. Given a string s consisting of only letters and digits, where we are allowed to transform any letter to uppercase or lowercase, return a list containing all possible permutations of the string.
#
# Ex: Given the following string…
#
# S = "c7w2", return ["c7w2", "c7W2", "C7w2", "C7W2"]

# Time complexity O(2^N)
# Space complexity O(2^N) - result from storing all permutations
def permutations(input):
    permutations = []

    findPermutations(input, 0, "", permutations)
    return permutations


def findPermutations(input, idx, current, permutations):
    if len(current) == len(input):
        permutations.append(current)
        return

    currentChar = input[idx]
    if currentChar.isnumeric():
        findPermutations(input, idx + 1, current + currentChar, permutations)
    else:
        findPermutations(input, idx + 1, current + currentChar, permutations)
        findPermutations(input, idx + 1, current + currentChar.upper(), permutations)


assert permutations("c7w2") == ["c7w2", "c7W2", "C7w2", "C7W2"]
assert permutations("") == []


# WORD SEARCH
#
# This question is asked by Amazon. Given a 2D board that represents a word search puzzle and a string word, return whether or the given word can be formed in the puzzle by only connecting cells horizontally and vertically.
#
# Ex: Given the following board and words…

# board =
# [
#     ['C','A','T','F'],
#     ['B','G','E','S'],
#     ['I','T','A','E']
# ]
# word = "CAT", return true
# word = "TEA", return true
# word = "SEAT", return true
# word = "BAT", return false
def wordSearch(puzzle, word):
    for i in range(len(puzzle)):
        for j in range(len(puzzle[i])):
            if puzzle[i][j] == word[0]:
                if explore(puzzle, word, 0, i, j):
                    return True

    return False


# Time complexity O(N * 4 ^M)
# Space Complexity O(M) where M is the length of the word
def explore(puzzle, word, idx, i, j):
    if idx == len(word):
        return True
    if i < 0 or i > len(puzzle) - 1 or j < 0 or j > len(puzzle[i]) - 1 or puzzle[i][j] != word[idx]:
        return False

    tmp = puzzle[i][j]
    puzzle[i][j] = ' '
    up = explore(puzzle, word, idx + 1, i - 1, j)
    down = explore(puzzle, word, idx + 1, i + 1, j)
    left = explore(puzzle, word, idx + 1, i, j - 1)
    right = explore(puzzle, word, idx + 1, i, j + 1)

    found = up or down or left or right
    puzzle[i][j] == tmp
    return found


assert wordSearch([['C', 'A', 'T', 'F'], ['B', 'G', 'E', 'S'], ['I', 'T', 'A', 'E']], "CAT")
assert wordSearch([['C', 'A', 'T', 'F'], ['B', 'G', 'E', 'S'], ['I', 'T', 'A', 'E']], "TEA")
assert wordSearch([['C', 'A', 'T', 'F'], ['B', 'G', 'E', 'S'], ['I', 'T', 'A', 'E']], "SEAT")
assert wordSearch([['C', 'A', 'T', 'F'], ['B', 'G', 'E', 'S'], ['I', 'T', 'A', 'E']], "BAT") == False


# TWO UNIQUE CHARACTERS
# Given a string s, return the length of the longest substring containing at most two distinct characters.
# Note: You may assume that s only contains lowercase alphabetical characters.
#
# Ex: Given the following value of s…
#
# s = "aba", return 3.
# Ex: Given the following value of s…
#      012
# s = "abca", return 2.
# Time complexity O(N)
# Space complexity O(1)
def longestSubstring(s):
    uniqueChars = {}
    lenOfRunningSubstring = 0
    lenOfLongestSubstring = 0
    idx = 0
    while idx < len(s):
        currentChar = s[idx]
        if (len(uniqueChars) < 3 and currentChar in uniqueChars) or len(uniqueChars) < 2:
            lenOfRunningSubstring += 1
        else:
            latestUniqueCharIdx = 0
            latestUniqueChar = None
            for k, v in uniqueChars.items():
                if v > latestUniqueCharIdx:
                    latestUniqueCharIdx = v
                    latestUniqueChar = k

            uniqueChars = {}
            uniqueChars[latestUniqueChar] = latestUniqueCharIdx
            lenOfRunningSubstring = idx - latestUniqueCharIdx + 1

        uniqueChars[currentChar] = idx
        lenOfLongestSubstring = max(lenOfRunningSubstring, lenOfLongestSubstring)
        idx += 1

    return lenOfLongestSubstring


assert longestSubstring("aba") == 3
assert longestSubstring("abca") == 2
assert longestSubstring("abcaaaaa") == 6
assert longestSubstring("abbbbbbbabcaaaaa") == 10
assert longestSubstring("") == 0

from queue import PriorityQueue


# Time complexity O(Nlog(k)) - keeping heap invariant
# Space complexity O(k) - at most k length of heap
def kSortedArray(arr, k):
    minHeap = PriorityQueue()
    for i in range(min(k + 1, len(arr))):
        minHeap.put(arr[i])

    indexToFill = 0

    for idx in range(k + 1, len(arr)):
        minElement = minHeap.get()
        arr[indexToFill] = minElement
        indexToFill += 1

        minHeap.put(arr[idx])

    while not minHeap.empty():
        minElement = minHeap.get()
        arr[indexToFill] = minElement
        indexToFill += 1
    return arr


assert kSortedArray([3, 2, 1, 5, 4, 7, 6, 5], 3) == [1, 2, 3, 4, 5, 5, 6, 7]


# Given a list of points, return the k closest points to the origin (0, 0).
#
# Ex: Given the following points and value of k…
#
# points = [[1,1],[-2,-2]], k = 1, return [[1, 1,]].


import math
from queue import PriorityQueue
#Runtime: O(Nlogk) where N is the total number of points we’re given.
#Space complexity: O(k)
def kClosestPoints(points, k):
    minHeap = PriorityQueue()

    for point in points:
        x,y = point
        minHeap.put((math.sqrt(x**2 + y**2) * (-1), point))
        while minHeap.qsize() > k:
            minHeap.get()


    result = []
    while minHeap.qsize() > 0:
        result.append(minHeap.get()[1])
    return result





assert kClosestPoints([[1, 1], [-2, -2]], 1) == [[1, 1]]
assert kClosestPoints([[1, 1], [-2, -2], [-1,-1]], 2) == [[-1,-1],[1, 1]]

# LIFE RAFTS
# This question is asked by Amazon. A ship is about to set sail and you are responsible for its safety precautions. More specifically, you are responsible
# for determining how many life rafts to carry onboard. You are given a list of all the passengers’ weights and are informed that a single life raft has a maximum capacity of limit and can hold at most two people.
# Return the minimum number of life rafts you must take onboard to ensure the safety of all your passengers. Note: You may assume that a the maximum weight of any individual is at most limit.
#
# Ex: Given the following passenger weights and limit…
#
# weights = [1, 3, 5, 2] and limit = 5, return 3
# weights = [1, 2] and limit = 3, return 1
# weights = [4, 2, 3, 3] and limit = 5 return 3


# [1,3,5,2] -> [1,2,3,5]
def countRafts(weights, limit):
    raftCounter = 0

    weights.sort()

    while len(weights) > 0:
        heaviest = weights[-1]
        slimmest = weights[0]
        if heaviest + slimmest <= limit and len(weights) > 1:
            weights.pop(0)

        weights.pop()
        raftCounter+=1

    return raftCounter

assert countRafts([1,3,5,2], 5) == 3
assert countRafts([1,2], 3) == 1
assert countRafts([4,2, 3, 3], 5) == 3
assert countRafts([5], 5) == 1


# Task assignment

def assignTask(tasks, k):
    tasksToIndices = mapTaskToIndices(tasks)

    sortedTasks = sorted(tasks)

    pairedTasks = []
    for idx in range(k):
        firstTaskDuration = sortedTasks[idx]
        idxOfFirstTask = tasksToIndices[firstTaskDuration].pop()

        secondIdx = len(tasks) -1 -idx
        secondTaskDuration = sortedTasks[secondIdx]
        idxOfSecondTask = tasksToIndices[secondTaskDuration].pop()

        pairedTasks.append([idxOfFirstTask, idxOfSecondTask])

    return pairedTasks
def mapTaskToIndices(tasks):
    taskToIndices = {}
    for idx, duration in enumerate(tasks):
        if duration in taskToIndices:
            taskToIndices[duration].append(idx)
        else:
            taskToIndices[duration] = [idxc]
    return taskToIndices


assert assignTask([1,3,5,3,1,4], 3) == [[0,2], [4,5], [1,3]]



# Linked List Sum
# Given two linked lists that represent two numbers, return the sum of the numbers also represented as a list.
#
# Ex: Given the two linked lists…
#
# a = 1->2, b = 1->3, return a list that looks as follows: 2->5
# Ex: Given the two linked lists…
#
# a = 1->9, b = 1, return a list that looks as follows: 2->0

class Node():
    def __init__(self, value, next = None):
        self.value = value
        self.next = next

def listSum(firstHead, secondHead):
    reversedFirstList = reverse(firstHead)
    reversedSecondList = reverse(secondHead)

    remainder = 0
    result = Node(-1)
    current = result

    while reversedFirstList or reversedSecondList:
        firstListDigit = reversedFirstList.value if reversedFirstList else 0
        secondListDigit = reversedSecondList.value if reversedSecondList else 0
        currentSum = firstListDigit + secondListDigit + remainder
        currentDigit = currentSum % 10
        remainder = currentSum // 10
        currentNode = Node(currentDigit)
        current.next = currentNode
        current = currentNode

        reversedFirstList = reversedFirstList.next if reversedFirstList else None
        reversedSecondList = reversedSecondList.next if reversedSecondList else None

    if reversedFirstList:
        result.next = reversedFirstList
    elif reversedSecondList:
        result.next = reversedSecondList


    return reverse(result.next)

assert listSum(Node(1, Node(2)), Node(1, Node(3))).value == 2
assert listSum(Node(1, Node(2)), Node(1, Node(3))).next.value == 5


def galtonBoard(m, n):
    cache = [[0 for _ in range(n)] for _ in range(m)]
    return explore(0,0, cache)

def explore(row, col, cache):
    if row >= len(cache) or col >= len(cache[row]):
        return 0
    if row == len(cache) -1 or col == len(cache[row]) -1 : # if reach the right end or bottom end there is only way to reach the bottom right cornet
        return 1

    return explore(row+1, col, cache) + explore(row, col+1, cache)

assert galtonBoard(2, 2) == 2
assert galtonBoard(4, 3) == 10


# Longest Increasing Subsequence
# This question is asked by Facebook. Given an array of unsorted integers, return the length of its longest increasing subsequence.
# Note: You may assume the array will only contain positive numbers.
#
# Ex: Given the following array nums…
#
# nums = [1, 9, 7, 4, 7, 13], return 4.
# The longest increasing subsequence is 1, 4, 7, 13.
#    j     i
# 1, 9, 7, 4, 7, 13
#[1, 2]
def longestIncreasingSubsequence(nums):
    dp = [1 for _ in range(len(nums))]

    for i in range(len(nums)):
        for j in range(i):
            if nums[i] > nums[j] and dp[i] < dp[j] +1:
                dp[i] = dp[j] +1

    return dp[-1]

assert longestIncreasingSubsequence([1, 9, 7, 4, 7, 13]) == 4

# CLEAN TREE
# This question is asked by Amazon. Given the root of a binary tree where every node’s value is either 0 or 1 remove every subtree that does not have a 1 in it.
# Ex: Given the following binary tree…
#
#         1
#       /  \
#     1      0
#


# Return the tree such that it’s been modified to look as follows…
#    1
#   /
#  1
# Ex: Given the following binary tree…
#
#       1
#      / \
#     1   1
# Return the tree such that it’s been modified to look as follows…
#       1
#      / \
#     1   1
# (No modifications are necessary)

class Node:
    def __init__(self, value, left = None, right =  None):
        self.value = value
        self.left = left
        self.right = right


#
#         1
#       /  \
#     1      0
#          /   \
#         1     0
#                \
#                 0


# Runtime: O(N) where N is the number of nodes in the tree.
# Space complexity: O(H) where H is the height of the tree.
def cleanTree(root):
    if not root:
        return None

    root.left = cleanTree(root.left)
    root.right = cleanTree(root.right)

    if root.value == 0 and not root.left and not root.right:
        return None

    return root


root = Node(1, Node(1), Node(0))

assert cleanTree(root).right == None
assert cleanTree(root).left.value == 1


# Given a string, s, return the length of the last word.
# Note: You may not use any sort of split() method.
#
# Ex: Given the following string…
#
# s = "The Daily Byte", return 4 (because "Byte" is four characters long).
#           j
# "The Daily Byte"

# The Daily Byte
# Time complexity O(N)
# Space complexity O(N) - due to creating new string with rstrip() method
def lenghOfLastWord(input):
    if not input:
        return 0

    input = input.rstrip()
    for i in reversed(range(len(input))):
        if input[i] == " ":
            return len(input) - (i+1)
    return 0


assert lenghOfLastWord("The Daily Byte") == 4
assert lenghOfLastWord("The Daily Byte ") == 4
assert lenghOfLastWord(" ") == 0
assert lenghOfLastWord(None) == 0


# num of ways to make a change

# return 6 having [5, 1]
#  0 1 2 3 4 5 6
# [1,1,1,1,1,2,2]

def numOfWays(n, denoms):
    numOfWays = [0 for i in range(n+1)]
    numOfWays[0] = 1
    for d in denoms:
        for i in range(1, n+1):
            if d <= i:
                numOfWays[i] += numOfWays[i - d]
    return numOfWays[-1]


assert numOfWays(6, [1,5]) == 2
assert numOfWays(2, [1,5]) == 1
assert numOfWays(10, [1,5]) == 3

# Given an array nums, return the third largest (distinct) value.
# Note: If the third largest number does not exist, return the largest value.
#
#
# Ex: Given the following array nums…
#
# nums = [1, 4, 2, 3, 5], return 3.
# Ex: Given the following array nums…
#
# nums = [2, 3, 3, 5], return 2.
# Ex: Given the following array nums…
#
# nums = [9, 5], return 9.


# Third Largest Distinct Value
# Given an array nums, return the third largest (distinct) value.
# Note: If the third largest number does not exist, return the largest value.
#
#
# Ex: Given the following array nums…
#
# nums = [1, 4, 2, 3, 5], return 3.
# Ex: Given the following array nums…
#
# nums = [2, 3, 3, 5], return 2.
# Ex: Given the following array nums…
#
# nums = [9, 5], return 9.

# [1, 4, 2, 3, 5]
# 3 -> 4 -> 5


# 2 -> 3 -> 3
#[2, 3, 3, 5])


# Time complexity O(NLog(N))
# Space complexity O(N)
from queue import PriorityQueue
def thirdLargest(nums):
    minHeap = PriorityQueue()
    visited = {}

    for num in nums:
        if num not in visited:
            visited[num] = True
            minHeap.put(num)
        while minHeap.qsize() > 3:
            minHeap.get()


    while minHeap.qsize() < 3 and minHeap.qsize() > 1:
        minHeap.get()

    return minHeap.get()

assert thirdLargest([1, 4, 2, 3, 5]) == 3
assert thirdLargest([2, 3, 3, 5]) == 2
assert thirdLargest([9,5]) == 9


# Given an array of words, return the length of the longest phrase, containing only unique characters, that you can form by combining the given words together.
#
# Ex: Given the following words…
#
# words = ["the","dog","ran"], return 9 because you can combine all the words, i.e. "the dog ran" since all the characters in the phrase are unique.
# Ex: Given the following words…
#
# words = ["the","eagle","flew"], return 4 because "flew" is the longest phrase you can create without using duplicate characters.


# Runtime: O(2^N * N) where N is the total number of words we’re given. This results from having two choices at each of our N words (i.e. take the current word or don’t take the current word).
# Space complexity: O(N) where N is the total number of words we’re given. This results from having N levels of recursion.
def longestPhrase(words):
    if not words or len(words) == 0:
        return 0

    longestPhrase = [0]

    findLongestPhrase(words, "",  0, longestPhrase)
    return longestPhrase[0]

def findLongestPhrase(words, current, idx, longestPhrase):
    isUnique = hasUniqueChars(current)
    if isUnique:
        longestPhrase[0] = max(longestPhrase[0], len(current))
    if idx == len(words) or not isUnique:
        return

    for i in range(len(words)):
        findLongestPhrase(words, current  + words[i], idx + 1, longestPhrase)

def hasUniqueChars(phrase):
    visited = {}
    for c in phrase:
        if c in visited:
            return False
        visited[c] = True
    return True

assert longestPhrase(["the","dog","ran"]) == 9
assert longestPhrase(["the","eagle","flew"]) == 4

# Take two
# Given an array of integers, nums, each element in the array either appears once or twice. Return a list containing all the numbers that appear twice.
# Note: Each element in nums is between one and nums.length (inclusive).
#
# Ex: Given the following array nums…
#
# nums = [2, 3, 1, 5, 5], return [5].
# Ex: Given the following array nums…
#
# nums = [1, 2, 3], return [].
# Ex: Given the following array nums…
#
# nums = [7, 2, 2, 3, 3, 4, 4], return [2,3,4].

# naive approach
# O(NlogN) time complexity
# O(1) space complexity

def takeTwo(nums):
    result = []

    for i in range(0, len(nums)):
        idx = abs(nums[i]) - 1
        if nums[idx] < 0:
            result.append(idx + 1)
        nums[idx] = -1 * nums[idx]
    return result

assert takeTwo([2, 3, 1, 5, 5]) == [5]
assert takeTwo([1,2,3]) == []
assert takeTwo([7, 2, 2, 3, 3, 4, 4]) == [2,3,4]

# Bipartite graph
# This question is asked by Facebook. Given an undirected graph determine whether it is bipartite.
# Note: A bipartite graph, also called a bigraph, is a set of graph vertices decomposed into two disjoint sets such that no two graph vertices within the same set are adjacent.
#
# Ex: Given the following graph...
#
# graph = [[1, 3], [0, 2], [1, 3], [0, 2]]
# 0----1
# |    |
# |    |
# 3----2
# return true.
# Ex: Given the following graph...
#
# graph = [[1, 3], [0, 2, 3], [1, 3], [0, 1, 2]]
# 0----1
# |  / |
# | /  |
# 3----2
# return false.

# Runtime: O(N + E) where N is the number of node in the graph and E is the number of Edges. This results from us iterating through both of them entirely.
# Space complexity: O(N) which is the space used to store the colors array + plus number of recursive calls
def twoGroups(graph):
    groups = [-1 for i in range(len(graph))]
    return explore(0, 1, graph, groups)

def explore(currentNode, nextGroup, graph, groups):

    if groups[currentNode] != -1  and groups[currentNode] != nextGroup:
        return False

    groups[currentNode] = nextGroup

    for child in graph[currentNode]:
        if groups[child] == -1:
            group = 1 if nextGroup == 0 else 0
            isPossible = explore(child, group, graph, groups)
            if not isPossible:
                return False

    return True

assert twoGroups([[1, 3], [0, 2], [1, 3], [0, 2]])
assert twoGroups([[1,3],[0,2],[1,3],[0,2]])
assert  not twoGroups([[1,2,3],[0,2],[0,1,3],[0,2]])
assert not twoGroups( [[1, 3], [0, 2, 3], [1, 3], [0, 1, 2]])

# K-th smallest element in the BST
# Given the reference to a binary search tree, return the kth smallest value in the tree.
#
# Ex: Given the following binary search tree and value k…
#
#      3
#     / \
#    2   4, k = 1, return 2.
# Ex: Given the following binary search tree and value k…
#
#     7
#    /  \
#    3   9
#     \
#      5, k = 3, return 7

class Node:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right



# Runtime: O(N) where N is the total number of elements in our tree.
# Space complexity: O(N) where N is the total number of elements in our tree.
def kthSmallest(root, k):
    result = [0, k]
    traverse(root, result)

    return result[0]

def traverse(root, result):
    if not root:
        return

    traverse(root.left, result)

    result[1] -=1
    if result[1] == 0:
        result[0] = root.value
        return


    traverse(root.right, result)

assert kthSmallest(Node(7, Node(3, None, Node(5)), Node(9)), 3) == 7
assert kthSmallest(Node(3, Node(2), Node(4)), 1) == 2


# SWITCHER
# Given a 2D matrix nums, return the matrix transposed.
# Note: The transpose of a matrix is an operation that flips each value in the matrix across its main diagonal.
#
# Ex: Given the following matrix nums…
#
# nums = [
#     [1, 2],
#     [3, 4]
# ]
# return a matrix that looks as follows...
#      0 1
# [
#0    [1,3],
#1    [2,4]
# ]


#      0 1
# [
#0    [1,3,2],
#1    [2,4,1]
#     [5,6,8]
# ]

def transpose(matrix):
    for i in range(len(matrix)):
        for j in range(i, len(matrix[i])):
            if i != j:
                print("swap " +  str(matrix[i][j]) +  " and " + str(matrix[j][i]))
                tmp = matrix[i][j]
                matrix[i][j] = matrix[j][i]
                matrix[j][i] = tmp

matrix = [[1,2], [3,4]]
transpose(matrix)
assert matrix == [[1,3], [2,4]]


# [
#0    [1,3,2],
#1    [2,4,1]
#     [5,6,8]

#0    [1,2,5],
#1    [3,4,6]
#     [2,1,8]

matrix = [[1,3,2], [2,4,1], [5,6,8]]
transpose(matrix)
assert matrix == [[1,2,5], [3,4,6],[2,1,8]]

# Infection
# Given a 2D array each cells can have one of three values. Zero represents an empty cell, one represents a healthy person, and two represents a sick person. Each minute that passes, any healthy person who is adjacent to a sick person becomes sick. Return the total number of minutes that must elapse until every person is sick.
# Note: If it is not possible for each person to become sick, return -1.
#
# Ex: Given the following 2D array grid…
#
# grid = [
#            [1,1,1],
#            [1,1,0],
#            [0,1,2]
#        ], return 4.
# [2, 1] becomes sick at minute 1.
# [1, 1] becomes sick at minute 2.
# [1, 0] and [0, 1] become sick at minute 3.
# [0, 0] and [0, 2] become sick at minute 4.
# Ex: Given the following 2D array grid…
#
# grid = [
#            [1,1,1],
#            [0,0,0],
#            [2,0,1]
#        ], return -1.


# Time complexity O(V+E) = O(N) - total number of cells
# Space complexity O(V+E) = O(N) - total number of cells
from collections import deque

def totalMinutes(grid):
    sickCounter = 0
    numberOfPersons = 0
    minutesElapsed = 0

    queue = deque()

    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 2:
                numberOfPersons += 1
                sickCounter +=1
                queue.append((i,j))
            if grid[i][j] == 1:
                numberOfPersons+=1

    while len(queue) > 0:
        if numberOfPersons == sickCounter:
            return minutesElapsed
        currentSize = len(queue)

        for i in range(currentSize):
            xCoord, yCoord = queue.popleft()
            healthyNeighbours = explore(xCoord, yCoord, grid)
            for n in healthyNeighbours:
                x, y = n
                grid[x][y] = 2
                sickCounter+=1
                queue.append((x,y))
        minutesElapsed+=1

    return minutesElapsed if numberOfPersons == sickCounter else -1

def explore(x,y, grid):
    neighbours = []

    if x-1 >= 0 and x-1 < len(grid) and y >= 0 and y < len(grid[x-1]) and grid[x-1][y] == 1:
        neighbours.append((x-1, y))

    if x >= 0 and x < len(grid) and y-1 >= 0 and y-1 < len(grid[x]) and grid[x][y-1] == 1:
        neighbours.append((x, y-1))

    if x+1 >= 0 and x+1 < len(grid) and y >= 0 and y < len(grid[x+1]) and grid[x+1][y] == 1:
        neighbours.append((x+1, y))

    if x >= 0 and x < len(grid) and y+1 >= 0 and y+1 < len(grid[x]) and grid[x][y+1] == 1:
        neighbours.append((x, y+1))
    return neighbours



assert totalMinutes([[1,1,1],[1,1,0],[0,1,2]]) == 4
assert totalMinutes([[1,1,1],[0,0,0],[2,0,1]]) == -1

# Link up
# This question is asked by Facebook. Given a singly linked list, re-order and group its nodes in such a way that the nodes in odd positions come first and the nodes in even positions come last.
#
# Ex: Given the reference to the following linked list...
#
# 4->7->5->6->3->2->1->NULL, return 4->5->3->1->7->6->2->NULL
# Ex: Given the reference to the following linked list...
#
# 1->2->3->4->5->NULL, return 1->3->5->2->4->NULL


# i  j
# 4->7->5->6->3->2->1->NULL, return 4->5->3->1->7->6->2->NULL
#     i
# 4-> 5
# 7 -> 6

# Time complexity O(N)
# Space complexity O(1)
def oddFirst(root):
    if not root or not root.next :
        return root

    i = root
    j = root.next
    jHead = j

    while j and j.next:
        i.next = j.next
        i = i.next
        j.next = i.next
        j = j.next

    i.next = jHead
    return root

class Node:
    def __init__(self, value, next = None):
        self.value = value
        self.next = next

root = Node(4, Node(7, Node(5, Node(6, Node(3, Node(2, Node(1)))))))

result = oddFirst(root)
assert result.value == 4
assert result.next.value == 5
assert result.next.next.value == 3
assert result.next.next.next.value == 1
assert result.next.next.next.next.value == 7
assert result.next.next.next.next.next.value == 6
assert result.next.next.next.next.next.next.value == 2
assert result.next.next.next.next.next.next.next == None

# High school students are voting for their class president and you’re tasked with counting the votes. Each presidential candidates is represented by a unique integer and the candidate that should win the election is the candidate that has received more than half the votes. Given a list of integers, return the candidate that should become the class president.
# Note: You may assume there is always a candidate that has received more than half the votes.
#
# Ex: Given the following votes…
#
# votes = [1, 1, 2, 2, 1], return 1.
# Ex: Given the following votes…
#
# votes = [1, 3, 2, 3, 1, 2, 3, 3, 3], return 3.

# Boyer-Moore majority voting algorithm to use constant space
# Time complexity O(N)
# Space complexity O(1)
def mostFrequentElement(votes):
    currentCandidate = votes[0]
    currentVote = 1

    for i in range(1, len(votes)):
        if currentVote == 0:
            currentVote = 1
            currentCandidate = votes[i]
        elif votes[i] == currentCandidate:
            currentVote+=1
        else:
            currentVote-=1
    return currentCandidate

assert mostFrequentElement([1, 1, 2, 2, 1]) == 1
assert mostFrequentElement([1, 3, 2, 3, 1, 2, 3, 3, 3]) == 3

# This question is asked by Amazon. Given N distinct rooms that are locked we want to know if you can unlock and visit every room. Each room has a list of keys in it that allows you to unlock and visit other rooms. We start with room 0 being unlocked. Return whether or not we can visit every room.
# Ex: Given the following rooms…
#
# rooms = [[1], [2], []], return true
# Ex: Given the following rooms…
#
# rooms = [[1, 2], [2], [0], []], return false, we can’t enter room 3.

# Time complexity: O(N + E) where N is the number of rooms and E is the number of keys
# Space complexity O(N)
from collections import deque
def visit(rooms):
    visited = [False for _ in range(len(rooms))]
    visited[0] = True
    queue = deque()

    queue.append(rooms[0])

    while len(queue) > 0:
        currentKeys = queue.popleft()
        for key in currentKeys:
            if not visited[key]:
                visited[key] = True
                queue.append(rooms[key])

    for isVisited in visited:
        if not isVisited:
            return False

    return True

assert visit([[1], [2], []])
assert not visit([[1, 2], [2], [0], []])


# Partners
# Given an integer array, nums, return the total number of “partners” in the array.
# Note: Two numbers in our array are partners if they reside at different indices and both contain the same value.
#
# Ex: Given the following array nums…
#
# nums = [5, 5, 3, 1, 1, 3, 3], return 5.
# 5 (index 0) and 5 (index 1) are partners.
# 1 (index 3) and 1 (index 4) are partners.
# 3 (index 2) and 3 (index 5) are partners.
# 3 (index 2) and 3 (index 6) are partners.
# 3 (index 5) and 3 (index 6) are partners.

def numOfPartners(nums):
    positions = {}

    for idx, num in enumerate(nums):
        if num in positions:
            positions[num].append(idx)
        else:
            positions[num] = [idx]

    partnerCounter = 0
    for num, positions in positions.items():
        partnerCounter += (len(positions) * (len(positions) -1)) // 2

    return partnerCounter

assert numOfPartners([5, 5, 3, 1, 1, 3, 3]) == 5
assert numOfPartners([5, 3, 1, 1, 3, 3]) == 4
assert numOfPartners([2, 3]) == 0
assert numOfPartners([3, 3]) == 1

def numOfPartners(nums):
    pairCounter = 0
    seen = {}
    for n in nums:
        if n in seen:
            pairCounter += seen[n]
            seen[n] +=1
        else:
            seen[n] = 1
    return pairCounter

assert numOfPartners([5, 5, 3, 1, 1, 3, 3]) == 5
assert numOfPartners([5, 3, 1, 1, 3, 3]) == 4
assert numOfPartners([2, 3]) == 0
assert numOfPartners([3, 3]) == 1


# Tree Pair
# Given the reference to the root of a binary search tree and a target value, return whether or not two individual values within the tree can sum to the target.
#
# Ex: Given the following tree and target…
#
#       1
#      / \
#     2   3, target = 4, return true.
# Ex: Given the following tree and target…
#
#       1
#      / \
#     2   3, target = 7, return false.

# 1,2,3


class Node:
    def __init__(self, value, left = None, right = None):
        self.value = value
        self.left = left
        self.right = right

def traverse(root, inorder):
    if not root:
        return
    traverse(root.left, inorder)
    inorder.append(root.value)
    traverse(root.right, inorder)

# Assuming that we are working with BST
# Time complexity O(N)
# Space complexity O(N)
def treePair(root, target):
    inorder = []
    traverse(root, inorder)

    left = 0
    right = len(inorder) -1
    while left < right:
        currentSum = inorder[left] + inorder[right]

        if currentSum == target:
            return True
        elif currentSum > target:
            right-=1
        else:
            left+=1

    return False

assert treePair(Node(2, Node(1), Node(3)), 4)
assert not treePair(Node(2, Node(1), Node(3)), 7)

# Assuming that we are working with Binary Tree (not BST)
# Runtime: O(N) where N is the total number of nodes in our tree.
# Space complexity: O(N) where N is the total number of nodes in our tree.
def treePair(root, k):
    visited = set()

    return treePairHelper(root, k, visited)

def treePairHelper(root, k, visited):
    if not root:
        return False
    if (k - root.value) in visited:
        return True
    else:
        visited.add(root.value)
        return treePairHelper(root.left, k, visited) or treePairHelper(root.right, k, visited)

assert treePair(Node(1, Node(2), Node(3)), 4)
assert not treePair(Node(1, Node(2), Node(3)), 7)
assert not treePair(None, 1)
assert not treePair(Node(1), 1)


# Longest substring with unique characters
# Given a string s, return the length of the longest substring that contains only unique characters.
#
# Ex: Given the following string s…
#       ij
# s = "ababbc", return 2.
# Ex: Given the following string s…
#
# s = "abcdssa", return 5.


'''
 
 i j
"ababbc"

'''
# Time complexity  O(N)
# Space complexity O(N)
def longestUniqueSubstring(s):
    lastSeen = {}
    longestUnique = 0
    i = 0
    j = 0
    while j < len(s):
        if s[j] in lastSeen:
            i = lastSeen[s[j]] +1
            lastSeen[s[j]] = j
            j = i + 1
        else:
            lastSeen[s[j]] = j
            currentLength = j - i + 1
            longestUnique = max(longestUnique, currentLength)
            j+=1

    return longestUnique

assert longestUniqueSubstring("ababbc") == 2
assert longestUniqueSubstring("abcdssa") == 5

# LONGEST CONSECUTIVE PATH
# Given the reference to a binary tree, return the length of the longest path in the tree that contains consecutive values.
# Note: The path must move downward in the tree.
# Ex: Given the following binary tree…
#
# 1
#  \
#   2
#    \
#     3
# return 3.
# Ex: Given the following binary tree…
#
#       1
#      / \
#     2    2
#    / \   / \
#   4  3  5  8
#     /
#    4
# return 4.

class Node:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

# Time complexity - O(N)
# Space complexity - O(N)
def longestConsecutivePath(root):
    return max(longestPathHelper(root.left, 1, root.value), longestPathHelper(root.right, 1, root.value))

def longestPathHelper(root, currentLength, previousValue):
    if not root:
        return currentLength

    print(str(root.value))
    print(str(previousValue))
    print()

    if root.value != previousValue + 1:
        currentLength =1
    else:
        currentLength +=1
    leftLengthContinue = longestPathHelper(root.left, currentLength, root.value)
    rightLengthContinue = longestPathHelper(root.right, currentLength, root.value)


    return max(leftLengthContinue, rightLengthContinue)


#       1
#      / \
#     2    2
#    / \   / \
#   4  3  5  8
#     /
#    4
# return 4.
assert longestConsecutivePath(Node(1, None, Node(2, None, Node(3)))) == 3
assert longestConsecutivePath(Node(1, Node(2, Node(4), Node(3, Node(4))), Node(2, Node(5), Node(8)))) == 4
assert longestConsecutivePath(Node(1, Node(1, Node(4), Node(2, Node(3))), Node(2, Node(5), Node(8)))) == 3

# WRITING EMAILS
# This question is asked by Google. Email addresses are made up of local and domain names. For example in kevin@dailybyte.com, kevin is the local name and dailybye.com is the domain name.
# Email addresses may also contain '+' or '.' characters besides lowercase letters.
# If there is a '.' in the local name of the email address it is ignored.
# Everything after a '+' in the local name is also ignored. For example ke.vin+abc@dailybyte.com is equivalent to kevin@dailybyte.com
# Given a list of email addresses, find the number of unique addresses.
#
# Ex: Given the following emails...
#
# emails = [
#              "test.email+kevin@dailybyte.com",
#              "test.e.mail+john.smith@dailybyte.com",
#              "testemail+david@daily.byte.com"
#          ], return 2. "testemail@dailybyte.com" and "testemail@daily.byte.com" are unique since the first two email addresses in the list are equivalent.

def uniqueEmails(emails):
    seen = set()

    for email in emails:
        formattedEmail = formatEmail(email)
        seen.add(formattedEmail)
    return len(seen)

# Time complexity O(N * M)
# Space complexity O(N)
def formatEmail(email):
    local, domain = email.split("@")

    formatted = []
    for c in local:
        if c == "+":
            break
        elif c == ".":
            continue
        else:
            formatted.append(c)

    return "".join(formatted) + "@" + domain

assert uniqueEmails(["test.email+kevin@dailybyte.com", "test.e.mail+john.smith@dailybyte.com", "testemail+david@daily.byte.com"]) == 2
assert uniqueEmails([]) == 0
assert uniqueEmails(["test.+email+kevin@dailybyte.com", "test.email+kevin@dailybyte.com"]) == 2
assert uniqueEmails(["test.+email+kevin@dailybyte.com", "test+email+kevin@dailybyte.com"]) == 1

#
# Given an image represented as a 2D array of pixels, return the image rotation ninety degrees.
#
# Ex: Given the following image…
#
# image = [     0  1   2
#            0 [10, 11, 12],
#            1 [13, 14, 15],
#            2 [16, 17, 18]
#         ], return the image such that it's been modified as follows...
# [    0   1  2
#    0 [16,13,10],
#    1 [17,14,11],
#    2 [18,15,12]
# ]

# Time complexity O(N^2)
# Space complexity O(N^2)
def rotateImage(img):
    result = []
    rowIdx = len(img)
    columnIdx = 0

    while columnIdx < len(img):
        current = []
        for j in reversed(range(0, rowIdx)):
            current.append(img[j][columnIdx])

        result.append(current)
        columnIdx+=1

    return result

# Optimized space by doing change in place
# Transpose across main diagonal and the reverse rows
# Time complexity O(N^2)
# Space complexity O(1)
def rotateImage(img):
    for i in range(len(img)):
        for j in range(i, len(img[i])):
            if i != j:
                img[i][j], img[j][i] = img[j][i], img[i][j]

    for i in range(len(img)):
        for j in range(len(img[i])//2):
            img[i][j], img[i][len(img)- j - 1] = img[i][len(img)- j - 1], img[i][j]
    return img

assert rotateImage([[10, 11, 12], [13, 14, 15], [16,17,18]])  == [[16,13,10], [17,14,11], [18,15,12]]


# dinners
#       s
# 1 2 3 4 5 6 7 8 9 10
# [   ]     [   ]
import math
from typing import List
def getMaxAdditionalDinersCount(N: int, K: int, M: int, S: List[int]) -> int:
    S.sort()
    S.append(N + K +1) # adding imaginary meal to make sure we consider all cases till N
    start = 1
    result = 0
    for s in S:
        if s - start > K:
            result += math.ceil((s - start - K) / (K + 1))
        start = s + K  + 1
    return result

assert getMaxAdditionalDinersCount(10, 1, 2, [2,6]) == 3

# SHORTEST DISTANCE
# This question is asked by Apple. Given a string and a character, return an array of integers where each index is the shortest distance from the character.
# Ex: Given the following string s and character s...
#
# s = "dailybyte", c = 'y', return [4, 3, 2, 1, 0, 1, 0, 1, 2]

# Time complexity O(N)
# Space complexity O(N)
def shortestDistance(s, c):
    distances = [float("inf") for _ in range(len(s))]

    distances[0] = 0 if s[0] == c else distances[0]
    for i in range(len(s)):
        if s[i] == c:
            distances[i] = 0
        else:
            distances[i] = min(distances[i], distances[i-1]+1)

    for i in reversed(range(len(s)-1)):
        distances[i] = min(distances[i], distances[i+1] + 1)

    return distances

assert shortestDistance(s = "dailybyte", c = 'y') == [4, 3, 2, 1, 0, 1, 0, 1, 2]


##
def getArtisticPhotographCount(N: int, C: str, X: int, Y: int) -> int:
    # Write your code here
    validLocationCounter = 0

    for p in range(N):
        if C[p] == 'P':
            for a in range(p + X, min(N, p + X + Y +1)):
                if C[a] == 'A':
                    for b in range(a+X, min(N, a + X + Y + 1)):
                        if C[b] == 'B':
                            validLocationCounter+=1
    return validLocationCounter


# TWO UNIQUE CHARACTERS
# Given a string s, return the length of the longest substring containing at most two distinct characters.
# Note: You may assume that s only contains lowercase alphabetical characters.
#
# Ex: Given the following value of s…
# {a -> 0, b -> 1} a
# s = "aba", return 3.
# Ex: Given the following value of s…
#
# s = "abca", return 2.

def longestTwoCharsSubstring(s):
    currentSubstringChars = []
    currentLongest = 0

    for i in range(len(s)):
        currentChar = s[i]
        if len(currentSubstringChars) < 2:
            if len(currentSubstringChars) == 0 or currentChar != s[currentSubstringChars[0]]: # empty or contain only one char
                currentSubstringChars.append(i) # add only unique chars
        elif currentChar not in [s[currentSubstringChars[0]], s[currentSubstringChars[1]]]:
            currentSubstringChars[0] = currentSubstringChars[1]
            currentSubstringChars[1] = i
        currentLongest = max(currentLongest, i - currentSubstringChars[0] +1)

    return currentLongest

assert longestTwoCharsSubstring("aba") == 3
assert longestTwoCharsSubstring("abca") == 2
assert longestTwoCharsSubstring("") == 0
assert longestTwoCharsSubstring("aabaca") == 4
assert longestTwoCharsSubstring("aababaca") == 6

# Given a list of words, return the top k frequently occurring words.
# Note: If two words occur with the same frequency, then the alphabetically smaller word should come first.
# Ex: Given the following words and value k…
#
# words = ["the", "daily", "byte", "byte"], k = 1, return ["byte"].
# Ex: Given the following words and value k…
#
# words = ["coding", "is", "fun", "code", "coding", "fun"], k = 2, return ["coding","fun"].

# Time complexity O(N^2logN)
# Space complexity O(N)

from queue import PriorityQueue

# Time complexity O(NlogN)
# Space complexity O(N)
def mostFrequentWords(words, k):
    counter = {}
    for word in words:
        if word in counter:
            counter[word]+=1
        else:
            counter[word] = 1

    minHeap = PriorityQueue()

    for w, freq in counter.items():
        minHeap.put((freq, w))
        if minHeap.qsize() > k:
            minHeap.get()

    result = []
    while k > 0:
        result.append(minHeap.get()[1])
        k-=1
    return result


assert mostFrequentWords(["the", "daily", "byte", "byte"], 1) == ['byte']
assert mostFrequentWords(["coding", "is", "fun", "code", "coding", "fun"], 2) == ["coding","fun"]


# CUT STRING - each substring must contain unique set of letters which does not exists in other substrings
# Given a string s containing only lowercase characters, return a list of integers representing the size of each substring you can create such that each character in s only appears in one substring.
#
# Ex: Given the following string s…
#
# s = "abacdddecn", return [3, 6, 1]
# Ex: Given the following string s…
#
# s = "aba", return [3]
# {a -> 2, b -> 1, c -> 8, d-> 6, e -> 7 n -> 9}
# "abacdddecn"

# Time complexity O(N)
# Space complexity O(N)
def cutString(s):
    lastSeen = {}
    result = []
    for idx, char in enumerate(s):
        lastSeen[char] = idx

    i = 0
    while i < len(s):
        currentChar = s[i]
        currentCharLastOccurence = lastSeen[currentChar]
        endIdx = findEndIdx(i, currentCharLastOccurence, s, lastSeen)
        result.append(endIdx - i + 1)
        i = endIdx + 1
    return result

def findEndIdx(currentIdx, lastOccurence, s, lastSeen):
    for j in range(currentIdx+1, lastOccurence):
        nextChar = s[j]
        nextCharLastOccurence = lastSeen[nextChar]
        if nextCharLastOccurence > lastOccurence:
            return findEndIdx(currentIdx+1, nextCharLastOccurence, s, lastSeen) # cddcd - currencChar is c but last occurence of d > last occurence of c so look further
    return lastOccurence # aba - current char a - grater than last occurence of b so just return last occurence of a


assert cutString("abacdddecn") == [3,6,1]
assert cutString("aba") == [3]
assert cutString("a") == [1]
assert cutString("abacbddecn") == [9, 1]

# GOOD PAIR
# Given an integer array that is sorted in ascending order and a value target, return two unique indices (one based) such that the values at those indices sums to the given target.
# Note: If no two such indices exist, return null.
#
# Ex: Given the following nums and target…
#
# nums = [1, 2, 5, 7, 9], target = 10, return [1,5].
# Ex: Given the following nums and target…
#
# nums = [1, 3, 8], target = 13, return null.

# Time complexity O(N)
# Space complexity O(1)
def twoSum(nums, target):
    left = 0
    right = len(nums) - 1

    while left < right:
        currentSum = nums[left] + nums[right]
        if currentSum < target:
            left+=1
        elif currentSum > target:
            right-=1
        else:
            return [left+1, right+1]

    return None


assert twoSum([1, 2, 5, 7, 9], 10) == [1,5]
assert twoSum([1, 3, 8], 13) == None
assert twoSum([], 13) == None
assert twoSum([1,2], 0) == None

# INDEX OF
# Given two strings s and t, return the index of the first occurrence of t within s if it exists; otherwise, return -1.
#
# Ex: Given the following strings s and t…
#
# s = "abc", t = "a", return 0.
# Ex: Given the following strings s and t…
#
# s = "abc", t = "abcd", return -1.

# Time complexity O(S*T)
# Space complexity O(1)
def indexOf(s,t):
    if len(t) > len(s):
        return -1

    sIdx = 0
    while sIdx < len(s):
        startIdx = sIdx
        tIdx = 0
        while sIdx < len(s) and tIdx < len(t) and s[sIdx] == t[tIdx]:
            sIdx+=1
            tIdx+=1
        if tIdx == len(t):
            return startIdx
        else:
            sIdx = startIdx + 1
    return -1

assert indexOf("abc", "a") == 0
assert indexOf("abcdefgh", "fgh") == 5
assert indexOf("abcdefgh", "cde") == 2
assert indexOf("abc", "abcd") == -1

#
# Given a 2D matrix, return a list containing all of its element in spiral order.
#
# Ex: Given the following matrix...
#
# matrix = [    0  1
#              [1, 2, 3],
#          11  [4, 5, 6],
#              [7, 8, 9]
#          ], return [1, 2, 3, 6, 9, 8, 7, 4, 5].

# matrix = [
#              [1,  2,  3,  4],
#              [5,  6,  7,  8],
#              [9, 10, 11, 12]
# return [1,2,3,4,8, 12,11,10,9, 5, 6, 7]

# Time complexity O(N)
# Space complexity O(N)
def spiralOrder(matrix):
    startRowIdx = 0
    startColumnIdx = 0
    endRowIdx = len(matrix) -1
    endColumnIdx = len(matrix[0]) - 1

    result = []

    while startRowIdx <= endRowIdx and startColumnIdx <= endColumnIdx:
        for i in range(startColumnIdx, endColumnIdx + 1):
            result.append(matrix[startRowIdx][i])
        for j in range(startRowIdx+1, endRowIdx+1):
            result.append(matrix[j][endColumnIdx])
        for k in reversed(range(startColumnIdx, endColumnIdx)):
            if startColumnIdx < endColumnIdx:
                if startRowIdx == endRowIdx:
                    break
                result.append(matrix[endRowIdx][k])
        for l in reversed(range(startRowIdx+1, endRowIdx)):
            if startColumnIdx == endColumnIdx:
                break
            result.append(matrix[l][startColumnIdx])

        startRowIdx +=1
        startColumnIdx+=1
        endColumnIdx -=1
        endRowIdx-=1

    return result

assert spiralOrder([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == [1, 2, 3, 6, 9, 8, 7, 4, 5]
assert spiralOrder([[1,  2,  3,  4], [5,  6,  7,  8], [9, 10, 11, 12]]) == [1,2,3,4,8, 12,11,10,9, 5, 6, 7]


# COMPRESS/DECOMPRESS STRING
# In this exercise, you're going to decompress a compressed string.
#
# Your input is a compressed string of the format number[string] and the decompressed output form should be the string written number times. For example:
#
# The input
#
# 3[abc]4[ab]c
#
# Would be output as
#
# abcabcabcababababc
#
# Other rules
# Number can have more than one digit. For example, 10[a] is allowed, and just means aaaaaaaaaa
#
# One repetition can occur inside another. For example, 2[3[a]b] decompresses into aaabaaab
#
# Characters allowed as input include digits, small English letters and brackets [ ].
#
# Digits are only to represent amount of repetitions.
#
# Letters are just letters.
#
# Brackets are only part of syntax of writing repeated substring.
#
# Input is always valid, so no need to check its validity.
#
#       l
# 2[3[a]b]
# 2 3 a

def decompress(input):
    result = []
    leftIdx = 0
    while leftIdx < len(input):
        while leftIdx < len(input) and input[leftIdx].isalpha():
            result.append(input[leftIdx])
            leftIdx+=1


        repeatCounter = 0
        while leftIdx < len(input) and input[leftIdx].isnumeric():
            repeatCounter*=10
            repeatCounter += int(input[leftIdx])
            leftIdx+=1

        if leftIdx < len(input) and input[leftIdx] == "[":
            leftIdx +=1
            substring = decompress(input[leftIdx:])
            print('----------------------------------------------')
            print(substring)
            print(repeatCounter)
            print('----------------------------------------------')
            result.extend(substring * repeatCounter)
            leftIdx+=len(substring)
            leftIdx+=1

        if leftIdx < len(input) and input[leftIdx] == "]":
            return "".join(result)

    return "".join(result)

assert decompress("3[abc]4[ab]c") == "abcabcabcababababc"
assert decompress("c10[a]") == "caaaaaaaaaa"
assert decompress("2[3[a]b]") == "aaabaaab"
#FAIL                                 accaccaccaccaccacc'
assert decompress("2[3[a2[c]]b]") == "accaccaccbaccaccaccb" # still failing

#3[abc]4[ab]c
#2[3[a]b]
def decompress(input):
    startIdx = 0
    result = []
    if startIdx == len(input) or input[startIdx] == "]":
        return ""
    elif input[startIdx].isalpha():
        while startIdx < len(input) and input[startIdx].isalpha():
            result.append(input[startIdx])
            startIdx+=1
    elif input[startIdx].isnumeric():
        times = 0
        while startIdx < len(input)  and input[startIdx].isnumeric():
            times = times * 10 + int(input[startIdx])
            startIdx+=1
        startIdx+=1 # skip the opening bracket
        substring = decompress(input[startIdx:])
        print(times)
        print(substring)
        result.extend(times * substring)
        startIdx += len(substring) + 1

    print(startIdx)
    return "".join(result) + decompress(input[startIdx:])



decompress("3[abc]4[ab]c") == "abcabcabcababababc"
decompress("c10[a]") == "caaaaaaaaaa"
decompress("2[3[a]b]") == "aaabaaab"
decompress("2[3[a2[c]]b]") == "accaccaccbaccaccaccb" # still failing
# CRACK THE CODE
# Given a string s and a string code, return whether or not s could have been encrypted using the pattern represented in code.
#
# Ex: Given the following s and code...
#
# s = “the daily byte”, code = “abc”, return true
# Ex: Given the following s and code...
#
# s = “the daily byte curriculum”, code = “abcc”, return false because ‘c’ in code already maps to the word “byte”

# Time complexity O(W)
# Space complexity O(W)
def crackTheCode(s, code):
    mapping = {}
    words = s.split()

    if len(words) != len(code):
        return False

    for idx, word in enumerate(words):
        char = code[idx]
        if char not in mapping:
            mapping[char] = word
        else:
            if mapping[char] != word:
                return False
    return True


assert crackTheCode("the daily byte", "abc") == True
assert crackTheCode("the daily byte byte", "abc") == False
assert crackTheCode("the daily byte curriculum", "abcc") == False
assert crackTheCode("the daily byte byte", "abcc") == True

# MAX PATH SUM
# Given the reference to a binary tree, return the maximum path sum.
# Note: The path that creates the maximum sum does not need to pass through the root of the tree.
#
# Ex: Given the reference to the following binary tree...
#
#       1
#      / \
#     4   9, return 14.

#
#          1
#          / \
#(15, 23) 4   9, return 14.
#        / \
#       8   11
class Node:
    def __init__(self, value, left = None, right = None):
        self.value = value
        self.left = left
        self.right = right

def maxSum(root):
    result = [float("-inf")]
    maxPathSum(root, result)

    return result[0]


def maxPathSum(root, result):
    if not root:
        return 0

    left = max(maxPathSum(root.left, result), 0)
    right = max(maxPathSum(root.right, result), 0)

    result[0] = max(result[0], root.value + left + right)

    return root.value + max(left, right)


maxSum(Node(1, Node(4), Node(9))) == 14
maxSum(Node(1)) == 1
maxSum(None) == 0
maxSum(Node(1, Node(4, Node(8), Node(11)), Node(9))) == 25


# VOLUME OF LAKES

# Imagine an islnd that is in the shape of a bar graph. When it rains, certain areas of the island fill up with rainwater to form lakes. Any excess rainwater the island cannot hold in lakes will run off the island to the west or east and drain into the ocean.
#
# Given an array of positive integers representing 2-D bar heights, design an algorithm (or write a function) that can compute the total volume (capacity) of water that could be held in all lakes on such an island given an array of the heights of the bars. Assume an elevation map where the width of each bar is 1.
#
# Example: Given [1,3,2,4,1,3,1,4,5,2,2,1,4,2,2], return 15 (3 bodies of water with volumes of 1,7,7 yields total volume of 15)

#      1   3 1 3     2 2 3
# [1,3,2,4,1,3,1,4,5,2,2,1,4,2,2]
# [0,1,3,3,4,4,4,4,4,5,5,5,5,5,5]
# [5,5,5,5,5,5,5,5,4,4,4,4,2,2,0]
def volumeOfLakes(heights):
    leftMaxHeights = [0 for i in range(len(heights))]
    for i in range(1, len(heights)):
        leftMaxHeights[i] = max(leftMaxHeights[i-1], heights[i-1])

    rightMaxHeights = [0 for i in range(len(heights))]
    for j in reversed(range(0, len(heights)-1)):
        rightMaxHeights[j] = max(rightMaxHeights[j+1], heights[j+1])

    totalVolume = 0
    for i in range(len(heights)):
        volume = min(leftMaxHeights[i], rightMaxHeights[i]) - heights[i]
        if volume > 0:
            totalVolume+= volume
    return totalVolume




volumeOfLakes([1,3,2,4,1,3,1,4,5,2,2,1,4,2,2]) == 15
volumeOfLakes([1,3,5,3,1]) == 0
volumeOfLakes([1,1]) == 0
volumeOfLakes([1,2, 1, 2, 1]) == 1


# Given an integer n, return whether or not it is a “magical” number. A magical number is an integer such that when you repeatedly replace the number with the sum of the squares of its digits its eventually becomes one.
#
# Ex: Given the following integer n…
#
# n = 19, return true.
# 1^2 + 9^2 = 82
# 8^2 + 2^2 = 68
# 6^2 + 8^2 = 100
# 1^2 + 0^2 + 0^2 = 1.
# Ex: Given the following integer n…
#
# n = 22, return false



def isMagicNumber(num, seen = set()):
    currentSum = 0
    while num > 0:
        currentSum += (num % 10) * (num % 10)
        num //=10

    if currentSum == 1:
        return True
    if currentSum in seen:
        return False
    else:
        seen.add(currentSum)

    return isMagicNumber(currentSum, seen)

assert isMagicNumber(19)
assert not isMagicNumber(22)

# Find longest word
# Given a string S and a set of words D, find the longest word in D that is a subsequence of S.
#
# Word W is a subsequence of S if some number of characters, possibly zero, can be deleted from S to form W, without reordering the remaining characters.
#
# Note: D can appear in any format (list, hash table, prefix tree, etc.
#
#     For example, given the input of S = "abppplee" and D = {"able", "ale", "apple", "bale", "kangaroo"} the correct output would be "apple"
#
# The words "able" and "ale" are both subsequences of S, but they are shorter than "apple".
#     The word "bale" is not a subsequence of S because even though S has all the right letters, they are not in the right order.
#     The word "kangaroo" is the longest word in D, but it isn't a subsequence of S.
def longestWord(S, D):
    longestWord = ""
    for word in D:
        if isSubstring(S, word) and len(word) > len(longestWord):
            longestWord = word

    return longestWord

def isSubstring(word, candidate):
    wordIdx = 0
    candidateIdx = 0

    while wordIdx < len(word) and candidateIdx < len(candidate):
        if word[wordIdx] == candidate[candidateIdx]:
            wordIdx+=1
            candidateIdx+=1
        else:
            wordIdx+=1
    return candidateIdx == len(candidate)



assert longestWord("abppplee", ["able", "ale", "apple", "bale", "kangaroo"]) == "apple"
assert longestWord("abppplee", ["able", "ale", "apple", "bale", "kangaroo", "applee"]) == "applee"
assert longestWord("abppplee", ["able"]) == "applee"

# WORD SQUARE
# A “word square” is an ordered sequence of K different words of length K that, when written one word per line, reads the same horizontally and vertically. For example:
#
# Copy
# BALL
# AREA
# LEAD
# LADY
#
# In this exercise you're going to create a way to find word squares.
#
# First, design a way to return true if a given sequence of words is a word square.
#
# Second, given an arbitrary list of words, return all the possible word squares it contains. Reordering is allowed.
#
# For example, the input list
#
# [AREA, BALL, DEAR, LADY, LEAD, YARD]
#
# should output
#
# [(BALL, AREA, LEAD, LADY), (LADY, AREA, DEAR, YARD)]
#
# Finishing the first task should help you accomplish the second task.

def containsWordSquare(words):
    k = len(words[0])

    subsets = set()

    generateSubsets(0, words, subsets, k)

    print(subsets)

    result = []
    for subset in subsets:
        if isWordSquare(subset, k):
            result.append(subset)
    return result

def isWordSquare(subset, k):
    rowIdx = 0
    columnIdx = 0

    while rowIdx < k -1 and columnIdx < k -1:
        for i in range(rowIdx, k):
            if subset[rowIdx][i] != subset[i][columnIdx]:
                return False
        rowIdx+=1
        columnIdx+=1
    return True

def generateSubsets(startIdx, words, subsets, k):
    if startIdx == len(words):
        subsets.add(tuple(words[:k]))

    for i in range(startIdx, len(words)):
        wordsCopy = words[:]
        wordsCopy[startIdx], wordsCopy[i] = wordsCopy[i], wordsCopy[startIdx]
        print(wordsCopy)
        generateSubsets(i+1, wordsCopy, subsets, k)
        wordsCopy[startIdx], wordsCopy[i] = wordsCopy[i], wordsCopy[startIdx]



assert containsWordSquare(["AREA", "BALL", "DEAR", "LADY", "LEAD", "YARD"]) == [(BALL, AREA, LEAD, LADY), (LADY, AREA, DEAR, YARD)]

# GROUP ANAGRAMS
# Given a list of strings, return a list of strings containing all anagrams grouped together.
# 
# Ex: Given the following list of strings strs…
# 
# strs = ["car", "arc", "bee", "eeb", "tea"], return
# [
#     ["car","arc"],
#     ["tea"],
#     ["bee","eeb"]
# ]


# Time complexity O(N * M log M )
# Space complexity O(N)
def groupAnagrams(strs):
    anagrams = {}

    for str in strs:
        sortedStr = "".join(sorted(str))
        if sortedStr not in anagrams:
            anagrams[sortedStr] = [str]
        else:
            anagrams[sortedStr].append(str)

    return list(anagrams.values())

assert groupAnagrams(["car", "arc", "bee", "eeb", "tea"]) == [["car","arc"],["bee","eeb"], ["tea"]]
assert groupAnagrams(["car", "ac", "ca", "carc"]) == [["car"],["ac","ca"], ["carc"]]
assert groupAnagrams([]) == []
assert groupAnagrams(["abc", "cba", "cab", "cac"]) == [["abc", "cba", "cab"], ["cac"]]

# Implement grep

# {
#0# accc ,
#1# aa  ,
#2# abc,
#3# cba,
#4# aa,
#5# vca,
#6# vca
#]



# 0 - 10000
def grep(lines, pattern, a=0, b=0, c=0):
    occurences = search(pattern, lines)
    if a > 0 or b > 0 or c > 0:
        ranges= []
        for idx in occurences:
            if a >0:
                ranges.append((idx, min(idx + a, len(lines))))
            elif b > 0:
                ranges.append((max(idx - b, 0), idx+1))
            elif c > 0:
                ranges.append((max(idx - c, 0), min(idx + c, len(lines))))

        sortedRanges = sorted(ranges, key=lambda x: x[0])
        print(sortedRanges)
        result = [sortedRanges[0]]
        for i in range(0, len(sortedRanges)):
            if sortedRanges[i][0] < result[-1][1]:
                previous = result.pop()
                result.append((previous[0], sortedRanges[i][1]))
            else:
                result.append(sortedRanges[i])
        return result
    else:
        return occurences


def search(pattern, lines):
    indices = []
    for idx, line in enumerate(lines):
        if pattern in line:
            indices.append(idx)
    return indices

lines = ["accc", "aa", "abc", "cba", "aa", "vca", "vca"]
#2# abc,
#3# cba,
#4# aa,
#5# vca,
#6# vca]
assert grep(lines, "aa") == [1, 4]
assert grep(lines, "aa", a=1) == [1,2,4, 5]
assert grep(lines, "aa", b=1) == [0, 1,3,4]
assert grep(lines, "aa", c=2) == [0, 1, 2, 3, 4 , 5]

# COMBINED TIMES
# Given a list of interval object, merge all overlapping intervals and return the result.
# Note: an interval object is a simple object that contains a start time and end time and can be constructed by passing a starting and ending time to the constructor.
#
# Ex: Given the following intervals...
#
# intervals = [[1, 3], [1, 8]], return a list of interval objects containing [[1, 8]].
# Ex: Given the following intervals...
#
# intervals = [[1, 2], [2, 6], [7 ,10]], return a list of interval objects containing [[1, 6], [7, 10]].

# Time complexity O(NlogN)
# Space complexity O(N)
def combineTimes(intervals):
    intervals.sort(key=lambda x: x[0])
    combinedIntervals = [intervals[0]]

    for i in range(1, len(intervals)):
        if combinedIntervals[-1][1] >= intervals[i][0]:
            lastInterval = combinedIntervals.pop()
            combinedIntervals.append([lastInterval[0], max(intervals[i][1], lastInterval[1])])
        else:
            combinedIntervals.append(intervals[i])
    return combinedIntervals

assert combineTimes([[1,3], [1,8]]) == [[1,8]]
assert combineTimes([[1,8], [2,3]]) == [[1,8]]
assert combineTimes([[1,8], [1,3]]) == [[1,8]]
assert combineTimes([[1, 2], [2, 6], [7 ,10]]) == [[1,6], [7, 10]]


# LARGEST ISLAND
# Given a 2D array grid, where zeroes represent water and ones represent land, return the size of the largest island.
# Note: An island is one or more cells in grid containing a value of one that may be connected vertically or horizontally. Each cell in an island contributes a value of one to the current island’s size.
#
# Ex: Given the following grid...
#
# grid = [
#            [1, 1, 0],
#            [1, 1, 0],
#            [0, 0, 1],
#        ], return 4.
def largestIsland(grid):
    largestIslandSize = 0

    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] == 1:
                currentIslandSize = sink(grid, i, j)
                largestIslandSize = max(currentIslandSize, largestIslandSize)

    return largestIslandSize

def sink(grid, i, j):
    if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[i]) or grid[i][j] == 0:
        return 0

    grid[i][j] = 0
    size = 1
    size+=sink(grid, i+1, j)
    size+=sink(grid, i, j+1)
    size+=sink(grid, i-1, j)
    size+=sink(grid, i, j-1)

    return size


assert largestIsland([[1, 1, 0], [1, 1, 0],[0, 0, 1] ]) == 4
assert largestIsland([[0, 0, 0], [0, 0, 0],[0, 0, 0] ]) == 0
assert largestIsland([[0, 0, 0], [0, 0, 0],[0, 0, 1] ]) == 1
assert largestIsland([[1, 1, 1], [1, 1, 1],[1, 1, 1] ]) == 9
assert largestIsland([[1, 1, 1], [0, 0, 0],[1, 0, 1] ]) == 3

# COMPRESS STRING
# This question is asked by Facebook. Given a character array, compress it in place and return the new length of the array.
# Note: You should only compress the array if its compressed form will be at least as short as the length of its original form.
#
# Ex: Given the following character array chars…
#
# chars = ['a', 'a', 'a', 'a', 'a', 'a'], return 2.
# chars should be compressed to look like the following:
# chars = ['a', '6']
# Ex: Given the following character array chars…
#
# chars = ['a', 'a', 'b', 'b', 'c', 'c'], return 6.
# chars should be compressed to look like the following:
# chars = ['a', '2', 'b', '2', 'c', '2']
# Ex: Given the following character array chars…
#
# chars = ['a', 'b', 'c'], return 3.
# In this case we chose not to compress chars.
#
#   i                        j
# ['a', 'a', 'a', 'a', 'a', 'a']

def compress(chars):
    i = 0
    j = 1
    index = 0
    while i < len(chars):
        while j < len(chars) and chars[i] == chars[j]:
            j+=1
        index+=1
        count = str(j - i)
        for c in range(len(count)):
            index+=1

        i = j
    return index if index <= len(chars) else len(chars)





assert compress(['a', 'a', 'a', 'a', 'a', 'a']) == 2
assert compress(['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a']) == 3
assert compress(['a', 'a', 'b', 'b', 'c', 'c']) == 6
assert compress(['a', 'b', 'c']) == 3

# Power of three
# Given an integer n, return whether or not it is a power of three.
#
# Ex: Given the following value for n...
#
# n = 9, return true
# Ex: Given the following value for n...
#
# n = 50, return false

# Runtime: O(logn) (using base 3).
# Space complexity: O(1) or constant.
def powerOfThree(n):
    while n != 1:
        if n % 3 == 0:
            n = n // 3
        else:
            return False

    return True


assert powerOfThree(9)
assert not powerOfThree(50)


# Given a string s, remove the minimum number of parentheses to make s valid. Return all possible results.
#
# Ex: Given the following string s...
#
# s = "(()()()", return ["()()()","(())()","(()())"].
# Ex: Given the following string s...
#
# s = "()()()", return ["()()()"].

# Time complexity N * 2 ^ N
# Space complexity
def removeBrackets(s):
    openingCounter = 0
    closingCounter = 0
    for c in s:
        if c == "(":
            openingCounter+=1
        else:
            closingCounter+=1

    diff = openingCounter - closingCounter
    if diff == 0:
        return [s]

    toRemove = ""
    if diff > 0:
        toRemove = "("
    else:
        toRemove = ")"

    validCombinations = set()

    for i in range(len(s)):
        explore(i, s, validCombinations, abs(diff), toRemove)


    return list(validCombinations)

def explore(startIdx, s, result, k, toRemove):
    if startIdx >=  len(s) or k < 0:
        return

    if k == 0:
        if isValid(s):
            result.add(s)

    if s[startIdx] == toRemove:
        explore(0, s[:startIdx] + s[startIdx+1:], result,  k -1, toRemove)
    else:
        explore(startIdx+1, s, result,  k, toRemove)

def isValid(s):
    stack = []

    for c in s:
        if c == "(":
            stack.append(c)
        elif c == ")" and stack[-1] != "(":
            return False
        else:
            stack.pop()
    return len(stack) == 0


assert removeBrackets("(()()()") == ['(())()', '(()())', '()()()']
assert removeBrackets("()()()") == ["()()()"]


# LINKED LIST LENGTH - EVEN or ODD
# Given a linked list of size N, your task is to complete the function isLengthEvenOrOdd() which contains head of the linked list and check whether the length of linked list is even or odd.
#
# Input:
# The input line contains T, denoting the number of testcases. Each testcase contains two lines. the first line contains N(size of the linked list). the second line contains N elements of the linked list separated by space.
#
# Output:
# For each testcase in new line, print "even"(without quotes) if the length is even else "odd"(without quotes) if the length is odd.
#
# User Task:
# Since this is a functional problem you don't have to worry about input, you just have to  complete the function isLengthEvenOrOdd() which takes head of the linked list as input parameter and returns 0 if the length of the linked list is even otherwise returns 1.
#
# Constraints:
# 1 <= T <= 100
# 1 <= N <= 103
# 1 <= A[i] <= 103
#
# Example:
# Input:
# 2
# 3
# 9 4 3
# 6
# 12 52 10 47 95 0
#
# Output:
# Odd
# Even
#
# Explanation:
# Testcase 1: The length of linked list is 3 which is odd.
# Testcase 2: The length of linked list is 6 which is even.
#

class Node:
    def __init__(self, value, next = None):
        self.value = value
        self.next = next

def isEven(root):
    length = 0

    while root:
        length+=1
        root = root.next

    return length% 2 ==0

assert not isEven(Node(2, Node(3, Node(4))))
assert isEven(Node(2, Node(3, Node(4, Node(5)))))

# BEST TIME TO BUY AND SELL STOCK |
# You are given an array prices where prices[i] is the price of a given stock on the ith day.
# You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.
# Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.
#
# Input: prices = [7,1,5,3,6,4]
# Output: 5
# Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
# Note that buying on day 2 and selling on day 1 is not allowed because you must buy before you sell.
#    m
# [7,1,5,3,6,4]

def maxProfit(prices):
    min = float("inf")
    maxProfit = 0ver

    for p in prices:
        if p < min:
            min = p
        else:
            maxProfit = max(maxProfit, p - min)
    return maxProfit

assert maxProfit([7,1,5,3,6,4]) == 5
assert maxProfit([7,6,5,5,4,3]) == 0

# BEST TIME TO BUY AND SELL STOCK II
# You are given an integer array prices where prices[i] is the price of a given stock on the ith day.
#
# On each day, you may decide to buy and/or sell the stock. You can only hold at most one share of the stock at any time. However, you can buy it then immediately sell it on the same day.
#
# Find and return the maximum profit you can achieve.

# Input: prices = [7,1,5,3,6,4]
# Output: 7
# Explanation: Buy on day 2 (price = 1) and sell on day 3 (price = 5), profit = 5-1 = 4.
# Then buy on day 4 (price = 3) and sell on day 5 (price = 6), profit = 6-3 = 3.
# Total profit is 4 + 3 = 7.
# Example 2:
#
# Input: prices = [1,2,3,4,5]
# Output: 4
# Explanation: Buy on day 1 (price = 1) and sell on day 5 (price = 5), profit = 5-1 = 4.
# Total profit is 4.

def maxProfit(prices):
    maxProfit = 0

    for i in range(len(prices)-1):
        if prices[i] < prices[i+1]:
            maxProfit+= prices[i+1] - prices[i]

    return maxProfit

assert maxProfit([7,1,5,3,6,4]) == 7
assert maxProfit([1,2,3,4,5]) == 4


#Best Time to Buy and Sell Stock III
# You are given an array prices where prices[i] is the price of a given stock on the ith day.
#
# Find the maximum profit you can achieve. You may complete at most two transactions.
#
# Note: You may not engage in multiple transactions simultaneously (i.e., you must sell the stock before you buy again).
#
#
#
# Example 1:
#
# Input: prices = [3,3,5,0,0,3,1,4]
# Output: 6
# Explanation: Buy on day 4 (price = 0) and sell on day 6 (price = 3), profit = 3-0 = 3.
# Then buy on day 7 (price = 1) and sell on day 8 (price = 4), profit = 4-1 = 3.
# one transaction consist of buying and selling
#       j
#         i
#   1,7,5,3,6,4
# 0 0 0 0 0 0 0
# 1 0 6 6 6 6 6
# 2 0 6 6
def maxProfit(prices):
    dp = [[0 for i in range(len(prices))] for _ in range(3)] # max profits till k transaction and p price

    for k in range(1, 3):
        for i in range(1, len(prices)):
            minBuyPrice = prices[0]
            for j in range(1, i+1):
                '''
                For k transactions, on i-th day,
                if we don't trade then the profit is same as previous day dp[k, i-1];
                and if we bought the share on j-th day where j=[0..i-1], then sell the share on i-th day then the profit is prices[i] - prices[j] + dp[k-1, j-1] .
                '''
                minBuyPrice = min(minBuyPrice, prices[j] - dp[k-1][j-1]) # keep track of max profit while buying the stock on j-th day. If profit with one less transaction and one less day is greater the buy price it will be added in the next line.

            dp[k][i] = max(dp[k][i-1], prices[i] - minBuyPrice)

    return dp[-1][-1]
# https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/discuss/135704/Detail-explanation-of-DP-solution



assert maxProfit([3,3,5,0,0,3,1,4]) == 6


from queue import PriorityQueue
def topKFrequent(words, k):
    freqs = {}

    for word in words:
        if word in freqs:
            freqs[word] +=1
        else:
            freqs[word] = 1

    maxHeap = PriorityQueue()

    for key,v in freqs.items():
        maxHeap.put(((-1) * v, key))

    mostFrequent = maxHeap.queue[0]

    result = []
    while maxHeap.qsize() > 0 and k > 0:
        topFrequent = []
        while maxHeap.queue[0] == mostFrequent:
            topFrequent.append(maxHeap.get()[1])
        topFrequent.sort()
        result.extend(topFrequent)
        if maxHeap.qsize() > 0:
            mostFrequent = maxHeap.queue[0]
        k-=1
    return result

assert topKFrequent(["i","love","leetcode","i","love","coding"], 2) == ["i","love"]



# Given an integer array nums and a value, val, remove all instances of val in-place and return the length of the new array.
# Note: It does not matter what values you leave in the array beyond the array’s new length.
# Ex: Given the following nums and val...
#
# nums = [1, 2, 3], val = 2, return 2 (after your modifications your array could look like [1, 3, 3]).


# REMOVE VALUE
def removeValue(nums, v):
    currentLength = len(nums)
    idx = 0
    for num in nums:
        if num != v:
            nums[idx] = num
            idx+=1

    return idx


assert removeValue([1, 2, 3], 2) == 2


# You are building a pool in your backyard and want to create the largest pool possible. The largest pool is defined as the pool that holds the most water. The workers you hired to dig the hole for your pool didn’t do a great job and because of this the depths of the pool at different areas are not equal. Given an integer array of non-negative integers that represents a histogram of the different heights at each position of the hole for your pool, return the largest pool you can create.
#
# Ex: Given the following heights...
#
# heights = [1, 4, 4, 8, 2], return 8.
# You can build your largest pool (most water) between indices 1 and 3 (inclusive) for a water volume of 4 * 2 = 8.

# Time complexity O(N)
# Space complexity O(1)
def largestPool(heights):
    startIdx = 0
    endIdx = len(heights) -1
    maxPool = 0
    while startIdx < endIdx:
        minValue = min(heights[startIdx], heights[endIdx])
        maxPool = max(maxPool, minValue * (endIdx - startIdx))
        if heights[startIdx] < heights[endIdx]:
            startIdx+=1
        else:
            endIdx-=1
    return maxPool

assert largestPool([1, 4, 4, 8, 2]) == 8

#
# Students in a class are lining up in ascending height order, but are having some trouble doing so. Because of this, it’s possible that some students might be out of order. In particular, a student that is taller than both their neighboring students (i.e. the person to both their left and right) sticks out like a sore thumb. Given an integer array that represents each students height, return the index of a “sore thumb”.
# Note: If there are multiple sore thumbs you may return the index of any of them. All numbers in the array will be unique. You may assume that to the left and right bounds of the array negative infinity values exist.
#
# Ex: Given the following students...
#
# students = [1, 2, 3, 7, 5], return 3.


# Time complexity O(N)
def outOfOrder(students):
    left = 0
    right = len(students) -1

    while left < right:
        mid = (right + left) // 2
        if students[mid] > students[mid+1]:
            right = mid
        else:
            left = mid + 1

    return left





assert outOfOrder([1, 2, 3, 7, 5]) == 3



# You are given the reference to the root of a binary tree and are asked to trim the tree of “dead” nodes. A dead node is a node whose value is listed in the provided dead array. Once the tree has been trimmed of all dead nodes, return a list containing references to the roots of all the remaining segments of the tree.
#
# Ex: Given the following binary tree and array dead…
#
#         3
#        / \
#      1     7
#     /  \  / \
#    2   8  4  6, dead = [7, 8], return a list containing a reference to the following nodes [3, 4, 6].


# DFS
# Time complexity O(N)
# Space complexity O(N) - recursion
def trimTree(root, dead):
    roots = []
    trimTreeHelper(root, None,  dead, roots)
    return roots

def trimTreeHelper(root, parent, dead, roots):
    if not root:
        return

    if root.value not in dead and parent == None:
        roots.append(root.value)

    if root.value in dead:
        trimTreeHelper(root.left, None, dead, roots)
        trimTreeHelper(root.right, None, dead, roots)
    else:
        trimTreeHelper(root.left, root, dead, roots)
        trimTreeHelper(root.right, root, dead, roots)


class Node:
    def __init__(self, value, left= None, right= None):
        self.value = value
        self.left = left
        self.right = right

root = Node(3, Node(1, Node(2), Node(8)), Node(7, Node(4), Node(6)))

assert trimTree(root, [7,8]) == [3,4,6]
assert trimTree(root, [3,8]) == [1,7]
assert trimTree(root, []) == [3]
assert trimTree(Node(3), [2]) == [3]
assert trimTree(Node(3), [3]) == []
assert trimTree(None, [3]) == []
assert trimTree(None, None) == []


# COIN CHANGE
# You are given an integer array coins representing coins of different denominations and an integer amount representing a total amount of money.
#
# Return the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1.
#
# You may assume that you have an infinite number of each kind of coin.
#
#
#
# Example 1:
#
# Input: coins = [1,2,5], amount = 11
# Output: 3
# Explanation: 11 = 5 + 5 + 1
# Example 2:
#
# Input: coins = [2], amount = 3
# Output: -1
# Example 3:
#
# Input: coins = [1], amount = 0
# Output: 0


def minCoinChange(coins, amount):
    coinCounter = 0

    coins.sort(reverse = True)
    for coin in coins:
        if coin == amount:
            coinCounter+=1
            return coinCounter
        elif coin < amount:
            currentCoins =  amount // coin
            coinCounter += currentCoins
            amount = amount % coin

    return -1 if amount != 0 else 0

assert minCoinChange([1,2,5], 11) == 3
assert minCoinChange([2], 3) == -1
assert minCoinChange([1], 0) == 0

# Given a list of integers, nums, return a list containing all triplets that sum to zero.
#
# Ex: Given the following nums...
#
# nums = [0], return [].
# Ex: Given the following nums...
#
# nums = [0, -1, 1, 1, 2, -2], return [[-2,0,2],[-2,1,1],[-1,0,1]].


# Time complexity O(N^2)
# Space complexity O(N)
def triplets(nums):
    results = []
    results = []
    nums.sort()

    for i in range(len(nums)-2):
        left = i +1
        right = len(nums)-1

        while left < right:
            sum = nums[i] + nums[left] + nums[right]
            if sum == 0:
                results.append([nums[i], nums[left], nums[right]])
                while left < right and nums[left] == nums[left + 1]:
                    left+=1
                while left < right and nums[right] == nums[right-1]:
                    right-=1
                left+=1
                right-=1
            elif sum > 0:
                right-=1
            else:
                left+=1

    return results






assert triplets([0]) == []
assert triplets([0, -1, 1, 1, 2, -2]) == [[-2,0,2],[-2,1,1],[-1,0,1]]


# # Binary Tree rubber
# You’re a thief trying to rob a binary tree. As a thief, you are trying to steal as much money as possible. The amount of money you steal is equivalent to the sum of all the node’s values that you decide to rob. If two adjacent nodes are robbed, the authorities are automatically alerted. Return the maximum loot that you can steal without alerting the authorities.
#
# Ex: Given the following binary tree...
#
#        2
#       / \
#      5   7, return 12 (5 + 7 = 12).
# Ex: Given the following binary tree...
#
#        4
#       / \
#     3     2
#     \     \
#     9     7, return 20 (4 + 9 + 7 = 20).
#

class Node:
    def __init__(self, value, left= None, right=None):
        self.value = value
        self.left = left
        self.right = right

#Runtime: O(N) where N is the total number of elements in our tree.
#Space complexity: O(N) where N is the total number of elements in our tree.
def maxLoot(root):
    if not root:
        return 0

    loot = 0
    if root.left:
        loot += maxLoot(root.left.left) + maxLoot(root.left.right)
    if root.right:
        loot += maxLoot(root.right.right) + maxLoot(root.right.left)

    return max(loot + root.value, maxLoot(root.right) + maxLoot(root.left))


root = Node(2, Node(5), Node(7))
assert maxLoot(root) == 12

root = Node(4, Node(3, None, Node(9)), Node(2, None, Node(7)))
assert maxLoot(root) == 20



# 31. Next Permutation
# Time complexity N * N!
# Space complexity N * N!
def nextPermutation(nums):
    """
    Do not return anything, modify nums in-place instead.
    """
    permutations = []

    findPermutations(0, nums, permutations)

    permutations.sort()
    print(permutations)
    for idx, permutation in enumerate(permutations):
        if permutation == nums and idx == len(permutations) - 1:
            return permutations[0]
        elif permutation == nums:
            return permutations[idx+1]


def findPermutations(startIdx, nums, permutations):
    if startIdx == len(nums) -1:
        permutations.append(nums[:])
        return

    for i in range(startIdx, len(nums)):
        nums[startIdx], nums[i] = nums[i], nums[startIdx]
        findPermutations(startIdx+1, nums, permutations)
        nums[startIdx], nums[i] = nums[i], nums[startIdx]

assert nextPermutation([1,2,3]) == [1,3,2]
assert nextPermutation([2,3,1]) == [3,1,2]
assert nextPermutation([3,2,1]) == [1,2,3]


# LEVEL UP THE TREE
# Given the reference to a binary search tree, “level-up” the tree. Leveling-up the tree consists of modifying every node in the tree such that every node’s value increases by the sum of all the node’s values that are larger than it.
# Note: The tree will only contain unique values and you may assume that it is a valid binary search tree.
#
# Ex: Given a reference to the following binary search tree...
#
#   0
#    \
#     3, modify the tree such that it becomes...
#   3
#    \
#     3
# Ex: Given a reference to the following binary search tree...
#
#   2
#  / \
# 1   3, modify the tree such that it becomes...
#   5
#  / \
# 6   3


#   2
#  /  \
# 1    4, modify the tree such that it becomes...
# \   / \
#  0 3   5

# Time complexity O(N)
# Space complexity O(N)
class Node:
    def __init__(self, value, left= None, right=None):
        self.value = value
        self.left = left
        self.right = right

def levelUp(bst):
    currentSum = [0]
    traverse(bst, currentSum)

def traverse(root, currentSum):
    if not root:
        return
    traverse(root.right, currentSum)
    root.value += currentSum[0]
    currentSum[0] = root.value
    traverse(root.left, currentSum)

bst = Node(0, None, Node(3))
levelUp(bst)
assert bst.value == 3
assert bst.right.value == 3

bst = Node(2, Node(1), Node(3))
levelUp(bst)
assert bst.value == 5
assert bst.left.value == 6
assert bst.right.value == 3


#   2
#  /  \
# 1    4
# \   / \
#  0 3   5

#   14
#  /  \
# 15    9
# \   / \
#  14 12   5

bst = Node(2, Node(1, None, Node(0)), Node(4, Node(3), Node(5)))
levelUp(bst)

assert bst.value == 14
assert bst.left.value == 15
assert bst.right.value == 9
assert bst.right.right.value == 5
assert bst.left.right.value == 14


# ADD VALUE
# Given an array digits that represents a non-negative integer, add one to the number and return the result as an array.
#
# Ex: Given the following digits...
#
# digits = [1, 2], return [1, 3].
# Ex: Given the following digits...
#
# digits = [9, 9], return [1, 0, 0].

# Time complexity O(N)
# Space complexity O(1)
def addOne(digits):
    currentIdx = len(digits) - 1
    while currentIdx >=0 and digits[currentIdx] == 9:
        digits[currentIdx] = 0
        currentIdx-=1

    if currentIdx < 0:
        digits = [1] + digits
    else:
        digits[currentIdx] +=1
    return digits


#addOne(None) == [1]
assert addOne([0]) == [1]
assert addOne([1,2]) == [1,3]
assert addOne([9,9]) == [1,0,0]
assert addOne([9,9,9,9]) == [1,0,0,0,0]


# EXPENSIVE INVENTORY
# You are given a list of values and a list of labels. The ith element in labels represents the label of the ith element.
# Similarly, the ith element in values represents the value associated with the ith element (i.e. together, an “item” could be thought of as a label and a price).
# Given a list of values, a list of labels, a limit, and wanted,
# return the sum of the most expensive items you can group such that the total number of items used is less than wanted and the number of any given label that is used is less than limit.
#
# Ex: Given the following variables...
#
# values = [5,4,3,2,1]
# label = [1,1,2,2,3]
# wanted = 3
# limit = 1
# return 9 (the sum of the values associated with the first, third, and fifth items).


from queue import PriorityQueue


# Time complexity O(NlogN) + N ~ NLogN
# Space complexity O(N)
def expensiveInventory(values, labels, wanted, limit):
    maxHeap = PriorityQueue()
    for idx, v in enumerate(values):
        label = labels[idx]
        maxHeap.put((-1 * v, label))

    sum = 0
    currentUsed = 0
    visited = {}
    while maxHeap.qsize() > 0 and currentUsed <= wanted:
        node = maxHeap.get()
        currentValue = node[0] * -1
        currentLabel = node[1]

        usedCounter = visited[currentLabel] if currentLabel in visited else 0
        if usedCounter == 0:
            visited[currentLabel]  = 0
        visited[currentLabel] +=1

        if usedCounter < limit:
            sum+=currentValue
            currentUsed+=1

    return sum

assert expensiveInventory([5,4,3,2,1], [1,1,2,2,3], 3, 1) == 9
assert expensiveInventory([5,4,3,2,1], [1,1,1,1,1], 3, 1) == 5
assert expensiveInventory([5,4,3,2,1], [1,2,1,1,1], 2, 1) == 9
assert expensiveInventory([5,5,4], [1,1,1], 2, 2) == 10

'''
A: A1->A2
          \
           C1->C2->C3
          /
B: B1->B2
return a reference to node C1.
'''
class ListNode:
    def __init__(self, next=None):
        self.value = value
        self.next = next

def listIntersection(a, b):
    lastA = a
    while lastA.next:
        lastA = lastA.next

    lastA.next = b

    slow = a
    fast = a

    while fast and fast.next:
        slow, fast = slow.next, fast.next.next

        if slow == fast:
            slow = a
            while slow != fast:
                slow = slow.next
                fast = fast.next
            lastA.next = None
            return slow
    lastA.next = None
    return None

'''
SUBARRAY SUM EQUALS K
Given an array of integers nums and an integer k, return the total number of subarrays whose sum equals to k.

[1,2,1,2,3] k =3 expected = 4
prefix sum
[1,3,4,6,9] 1
{
1:1 result 0
3 result 1
4 result 2 ((4-k) = 1 is already visited)
}
'''

def subarraySum(self, nums: List[int], k: int) -> int:
    pass


assert subarraySum([1,1,1], 2) == 2
assert subarraySum([1,2,3], 3) == 2


'''
STRING VALIDITY
Given a string s that contains only the following characters: (, ), and *, where asterisks can represent either an opening or closing parenthesis, return whether or not the string can form a valid set of parentheses.

Ex: Given the following string s…

s = "*)", return true.
Ex: Given the following string s…

s = "(**)", return true.
Ex: Given the following string s…

s = "((*", return false.


'''


# Greedy algorithm
# number of possible values of extra left open brackets
# https://leetcode.com/problems/valid-parenthesis-string/solution/
def isValid(s):
    pass

assert isValid("(**)")
assert isValid("*)")
assert not isValid("((*)")


'''
IS SUBTREE

This question is asked by Amazon. Given two trees s and t return whether or not t is a subtree of s.
Note: For t to be a subtree of s not only must each node’s value in t match its corresponding node’s value in s, but t must also exhibit the exact same structure as s. You may assume both trees s and t exist.

'''

class Node:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

def isSubTree(s, t):
    if not s:
        return False

    if isSameTree(s, t):
        return True
    else:
        return isSubTree(s.left, t) or isSubTree(s.right, t)

def isSameTree(s, t):
    if not s or not t:
        return not s and not t
    elif s.value == t.value:
        return isSameTree(s.left, t.left) and isSameTree(s.right, t.right)
    else:
        return False




s1 = Node(1, Node(3), Node(8))
t1 = Node(1, None, Node(8))

assert not isSubTree(s1, t1)

s2 = Node(7, Node(8), Node(3))
t2 = Node(7, Node(8), Node(3))
assert isSubTree(s2, t2)

s3 = Node(7, Node(8), Node(3))
t3 = Node(7, Node(8), Node(3, Node(1)))
assert not isSubTree(s3, t3)

s4 = Node(1, Node(8), Node(7, Node(8), Node(3)))
t4 = Node(7, Node(8), Node(3))
isSubTree(s4, t4) == True


'''
VALID PARENTHESIS - STRING VALIDITY
Given a string s that contains only the following characters: (, ), and *, where asterisks can represent either an opening or closing parenthesis, return whether or not the string can form a valid set of parentheses.

keep track if opening and closing brackets are balanced


   h
 l
 (**)
    h
'''
def checkValidity(s):
    low = 0
    high = 0

    for c in s:
        low += 1 if c == "(" else -1
        high += 1 if c != ")" else -1

        if high < 0:
            break
        low = max(low, 0)
    return low == 0

assert checkValidity("*)")
assert checkValidity("(**)")
assert not checkValidity("((*")


'''
Given an n-ary tree, return a list containing the post order traversal of the tree. Write your solution iteratively.

        1
      / | \
     2  3  4, return [2, 3, 4, 1]

Putting values on stack: 1,2 (nothing to traverse - just append to result), 3 (nothing to traverse - just append to result), 4 (nothing to traverse - just append to result), 1 (already explored the childs - just append to result)

2 -> 
'''
