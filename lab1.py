# Exercise SET 1

'''
# 1
a = input("Enter num1: ")
b = input("Enter num2: ")

a = int(a)
b = int(b)

if a * b > 1000:
    print("Sum =", a+b)
else:
    print("Product =", a*b)

lst = [1,2,3,4,5,1]
if lst[0] == lst[-1]:
    print("True")
else:
    print("False")

# 3
a = [25, 12, 30, 35, 90, 8, 10]
for i in range(len(a)):
    if a[i] % 5 == 0:
        print(a[i])

# 4
str = "Emma is a good developer. Emma is also a writer"
lst = list(str.split(" "))
print(lst)
count = 0

for i in range(len(lst)):
    if lst[i] == "Emma":
        count = count + 1

print(count)

# 5
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

# 6
s1 = "hamzarehman"
s2 = "lol"

a = int (len(s1) / 2)

s3 = s1[:a] + s2 + s1[a:]

print(s3)

# 7
s1 = "italy"
s2 = "pakistan"

s3 = s1[0] + s2[0] + s1[(int(len(s1) / 2))] + s2[(int(len(s2) / 2))] + s1[-1] + s2[-1]
print(s3)

# 8
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

# 9
str = "USA is comprised of three alphabets U, S and A collectively usa or USA"
str = str.upper()

lst = list(str.split(" "))
jackpot = "USA"
count = 0

for i in range(len(lst)):
    if lst[i] == jackpot:
        count += 1

print("USA occured", count, "times.")

# 10
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

# 11
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
'''

# 3
