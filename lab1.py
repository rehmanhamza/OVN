# Exercise SET 1

# 1
a = input("Enter num1: ")
b = input("Enter num2: ")

a = int(a)
b = int(b)

if a * b > 1000:
    print("Sum =", a+b)
else:
    print("Product =", a*b)

# 2
lst = [1,2,3,4,5]

for i in range(len(lst)):
    lst[i] = int(lst[i])
    if i-1 < 0:
        print(lst[i])
    else:
        print(lst[i] + lst[i-1])


# 3
lst = [1,2,3,4,5,1]
if lst[0] == lst[-1]:
    print("True")
else:
    print("False")

# 4
a = [25, 12, 30, 35, 90, 8, 10]
for i in range(len(a)):
    if a[i] % 5 == 0:
        print(a[i])

# 5
str = "Emma is a good developer. Emma is also a writer"
lst = list(str.split(" "))
print(lst)
count = 0

for i in range(len(lst)):
    if lst[i] == "Emma":
        count = count + 1

print(count)

# 6
l1 = [1,2,3,4,5,6,7,8,9]
l2 = [10,11,12,13,14,15,16,17,18]
l3 = []

i = len(l1)
j = len(l2)
k = 0
while k < i or k < j:
    if l1[k] % 2 != 0:
        l3.append(l1[k])
    if l2[k] % 2 == 0:
        l3.append(l2[k])
    k = k + 1

print(l3)

# 7
s1 = "hamzarehman"
s2 = "lol"

a = int (len(s1) / 2)

s3 = s1[:a] + s2 + s1[a:]

print(s3)

# 8
s1 = "italy"
s2 = "pakistan"

s3 = s1[0] + s2[0] + s1[(int(len(s1) / 2))] + s2[(int(len(s2) / 2))] + s1[-1] + s2[-1]
print(s3)

# 9
a = "haMzA$Re&Hma(N"
lower, upper, special = 0, 0, 0

for i in range(len(a)):
    if a[i] >= 'a' and a[i] <= 'z':
        lower += 1
    elif a[i] >= 'A' and a[i] <= 'Z':
        upper += 1
    else:
        special += 1

print("Lowercase =", lower, "\nUppercase =", upper, "\nSpecial chars =", special)

# 10
str = "USA is comprised of three alphabets U, S and A collectively usa or USA"
str = str.upper()

lst = list(str.split(" "))
jackpot = "USA"
count = 0

for i in range(len(lst)):
    if lst[i] == jackpot:
        count += 1

print("USA occured", count, "times.")

# 11
str = "ham23za78reh891m2an"
lst = []
for i in range(len(str)):
    if str[i] >= '0' and str[i] <= '9':
        lst.append(str[i])

sum = 0
for i in range(len(lst)):
    lst[i] = int(lst[i])
    sum = sum + lst[i]

print("Sum =", sum, "\nAverage =", (sum / len(lst)))

# 12
from collections import Counter

a = "mississippi"
dict = {}

for i in a:
    if i in dict:
        dict[i] += 1
    else:
        dict[i] = 1

print(dict)

res = Counter(a)

print(str(res))
'''

# Exercise SET 2

'''
# 1
listOne = [3, 6, 9, 12, 15, 18, 21]
listTwo = [4, 8, 12, 16, 20, 24, 28]

lst = listOne[::2] + listTwo[1::2]

print(lst)

# 2
sampleList = [34, 54, 67, 89, 11, 43, 94]
sampleList[2] = sampleList[4]
sampleList[-1] = sampleList[4]

print(sampleList)


# 3
sampleList = [11, 45, 8, 23, 14, 12, 78, 45, 89]
l = []
for i in range(len(sampleList)):
    if i % 3 == 0:
        l.append(i)

l1 = sampleList[(int(l[0])):(int(l[1]))]
l2 = sampleList[(int(l[1])):(int(l[2]))]
l3 = sampleList[(int(l[2])):]

print("Three equal chunks of sampleList:\n", l1, l2, l3)


# 4
from collections import Counter

sampleList = [11, 45, 8, 11, 23, 45, 23, 45, 89]
dict = {}

for i in sampleList:
    if i in dict:
        dict[i] += 1
    else:
        dict[i] = 1

print(dict)

res = Counter(sampleList)

print(res)



# 5
firstList = [2, 3, 4, 5, 6, 7, 8]
secondList = [4, 9, 16, 25, 36, 49, 64]

res = zip(firstList, secondList)
res = set(res)

print(res)


# 6
firstSet = {23, 42, 65, 57, 78, 83, 29}
secondSet = {57, 83, 29, 67, 73, 43, 48}

s = firstSet.intersection(secondSet)

for i in s:
    firstSet.remove(i)

print("Intersection:", s)
print(firstSet)


# 7
firstSet = {57, 83, 29}
secondSet = {57, 83, 29, 67, 73, 43, 48}

a = firstSet.issubset(secondSet)
b = firstSet.issuperset(secondSet)

print("firstSet subset of secondSet:", a, "\nfirstSet superset of secondSet:", b)

for i in firstSet:
    secondSet.remove(i)

print("After removing:\n", secondSet)


# 8
rollNumber = [47, 64, 69, 37, 76, 83, 95, 97]
sampleDict ={"John":47, "Emma":69, "Kelly":76, "Jason":97}
l1 = []

for i in rollNumber:
    if i in sampleDict.values():
        l1.append(i)

rollNumber = list(l1)

print(rollNumber)


# 9
speed = {"jan":47, "feb":52, "march":47, "April":44, "May":52, "June":53, "july":54, "Aug":44, "Sept":54}
lst = speed.values()
lst = list(set(lst))

print(lst)


# 10
sampleList = [87, 52, 44, 53, 54, 87, 52, 53]
new_lst = list(set(sampleList))
new_lst = tuple(new_lst)

print(new_lst)
print("Minimum in new_lst =", min(new_lst), "\nMaximum in tuple =", max(new_lst))

string = "Python is the best"
print(string)