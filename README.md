#items = "FDFFDFDD", return 2
#"FD" creates the first balanced meal.
#"FFDFDD" creates the second balanced meal.


def balancedMeals(items):
foodCounter = 0
drinkCounter = 0
mealsCounter = 0

    for item in items:
        if item == "F":
            foodCounter+=1
        else:
            drinkCounter+=1

        if foodCounter == drinkCounter:
            mealsCounter +=1

    return mealsCounter


balancedMeals("FDFFDFDD")
balancedMeals("FDFDFD")

# prices = [[40,30],[300,200],[50,50],[30,60]], return 310
# prices = [[30,60], [40,30], [50,50], [300,200],,], return 310
# Fly the first personn to office B.
# Fly the second person to office B.
# Fly the third person to office A.
# Fly the fourth person to office A.

def minimumCost(prices):
minCost = 0
officeThreshold = len(prices) / 2
prices.sort(key=lambda x: x[0]-x[1], reverse=True)

    for i in range(len(prices)):
        if i < officeThreshold:
            minCost += prices[i][0]
        else:
            minCost += prices[i][1]

    return minCost


minimumCost([[40,30],[300,200],[50,50],[30,60]])

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
bricksCounter +=1
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
prefixes = [False for i in range(len(s)+1)]
prefixes[0] = True

    for i in range(len(s)+1):
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
#
# [1, 2]
# [R, R]
# [1,3,5] [2,4,6]
for i in range(1, len(costs)):
costs[i][0] += min(costs[i-1][1], costs[i-1][2])
costs[i][1] += min(costs[i-1][0], costs[i-1][2])
costs[i][2] += min(costs[i-1][0], costs[i-1][1])

    return min(costs[-1][0], costs[-1][1], costs[-1][2])


minCost([[1, 3, 5],[2, 4, 6],[5, 4, 3]])

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
#[[0,3], [0, 3, 5, 8], [0, 3, 5, 6, 9, 11]]
currentValues = []
currentWeights = []

    if len(weights) > 0:
        currentValues.extend([0,values[0]])
        currentWeights.extend([0,weights[0]])

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
dp = [[0 for _ in range(W + 1)] for _ in range(len(values) +1)]

    for i in range(0, len(dp)):
        for j in range(0, len(dp[i])):
            if i == 0 or j ==0:
                dp[i][j] = 0
            elif j < weights[i-1]:
                dp[i][j] = dp[i-1][j]
            else:
                dp[i][j] = max(values[i-1] + dp[i - 1][j - weights[i-1]], dp[i - 1][j])

    return dp


maxStolenValue2(W = 10, weights = [4, 1, 3], values = [4, 2, 7])
maxStolenValue2(W = 5, weights = [2, 4, 3], values = [3, 7, 2])
maxStolenValue(W = 7, weights = [1, 3, 4], values = [3, 5, 6])

#        i
# [2, 4, 3]
# [3, 7, 2]
#       j
#   0 1 2 3 4 5    W
# 0 0 0 0 0 0 0
# 1 0 0 3 3 3 3
# 2 0 0 3 3 7 7
# 3 0 0 3 3
#V



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
mid = (left + right ) // 2

        if mid == 0:
            return mid
        if isBadRelease(mid) and not isBadRelease(mid -1):
            return mid
        elif isBadRelease(mid):
            right = mid -1
        else:
            left = mid + 1

    return -1

def findBadRelease(N):
return findBadReleaseHelper(0, N)

def findBadReleaseHelper(start, end):
middle = (start + end) // 2

    if middle == 0:
        return 0

    if isBadRelease(middle) and not isBadRelease(middle-1):
        return middle
    elif isBadRelease(middle) and isBadRelease(middle-1):
        return findBadReleaseHelper(start, middle)
    else:
        return findBadReleaseHelper(middle+1, end)

def isBadRelease(i):
if i >= 0:
return True
else:
return False


import math

def isPrime(n):
for i in range(2, int(math.sqrt(n) +1)):
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
n = n/i
return factors

numberOfPrimeFactors(3) #  int(math.sqrt(3)) - number of distinct factors(not only prime numbers) including 1 and n
numberOfPrimeFactors(24) # int(math.sqrt(24)) - number of factors (not only prime numbers) including 1 and n
numberOfPrimeFactors(25) # int(math.sqrt(25)) - number of factors (not only prime numbers) including 1 and n

#This question is asked by Facebook.
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
return int(math.sqrt(n)) # equals to number of distinct factors

# N =1 # O O O
# N =2 # O C O
# N =3 #O C O
gymLockers(3) # equals 1

# N =1 #O O O O
# N =2 #O C O C
# N =3 #O C O C
# N =4 #O C O C
gymLockers(4) # equals 2

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
sum+=num
n = len(nums)
return int((n * (n+1)/2) - sum)


missingValue([1, 4, 2, 0])
guassian([1, 4, 2, 0])
missingValue([6, 3, 1, 2, 0, 5] )
guassian([6, 3, 1, 2, 0, 5] )

## complementary number
# number = 27, return 4.
# 27 in binary (not zero extended) is 11011.
# Therefore, the complementary binary is 00100 which is 4.

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





