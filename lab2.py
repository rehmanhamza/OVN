import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

# Exercise SET 3

'''
#1
#a = np.array([[1,2], [3,4], [5,6], [7,8]])
a = np.empty([4,2], dtype = np.uint16)
print(a)
print(a.shape)
print(a.ndim)

# 2
a = np.arange(100, 200, 10).reshape(5,2)
print(a)


# 3
a = np.array([[11 ,22, 33], [44, 55, 66], [77, 88, 99]])
b = a[..., 1]
print(b)


# 4
a = np.array([[3 ,6, 9, 12], [15 ,18, 21, 24], [27 ,30, 33, 36], [39 ,42, 45, 48], [51 ,54, 57, 60]])
a = a[::2, 1::2]
print(a)


# 5
a = np.array([[5, 6, 9], [21 ,18, 27]])
b = np.array([[15 ,33, 24], [4 ,7, 1]])
c = np.add(a,b)
c = np.square(c)

print(c)


# 6
a = np.array([[34,43,73],[82,22,12],[53,94,66]])
a = np.sort(a)

print(a)


# 7
a = np.array([[34,43,73],[82,22,12],[53,94,66]])
a1 = np.amin(a, 1)
a2 = np.amax(a, 0)
print(a1, "\n", a2)


# 8
a = np.array([[34,43,73],[82,22,12],[53,94,66]])
new_column = np.array([10,10,10])
a = np.delete(a, 1, axis=1)
a = np.insert(a, 1, new_column, axis=1)

print(a)


# Exercise SET 4

# 1
df = pd.read_csv("sales_data.csv")
profit_list = df['total_profit'].tolist()
months = df['month_number'].tolist()

plt.plot(months, profit_list)
plt.xticks(months)
plt.yticks(np.arange(100000, 600000, 100000))
plt.xlabel("Months")
plt.ylabel("Profits")
plt.title("Company profit per month")
plt.show()


# 2
df = pd.read_csv("sales_data.csv")
profit_list = df['total_profit'].tolist()
months = df['month_number'].tolist()

plt.plot(months, profit_list, label='Profit data of last year', color='r', marker='o', markerfacecolor='k', linestyle='dashed', linewidth=3)
plt.title('Company profit per month')
plt.xticks(months)
plt.yticks(np.arange(100000, 600000, 100000))
plt.legend()
plt.show()


# 3
df = pd.read_csv("sales_data.csv")
months = df['month_number'].tolist()
faceCream = df['facecream'].tolist()
faceWash = df['facewash'].tolist()
toothPaste = df['toothpaste'].tolist()
bathingSoap = df['bathingsoap'].tolist()
shampoo = df['shampoo'].tolist()
moisturizer = df['moisturizer'].tolist()

plt.plot(months, faceCream, label='facecream sales Data', color='b', marker='o', linewidth=3)
plt.plot(months, faceWash, label='facewash sales Data', color='g', marker='o', linewidth=3)
plt.plot(months, toothPaste, label='toothpaste sales Data', color='r', marker='o', linewidth=3)
plt.plot(months, bathingSoap, label='bathingsoap sales Data', color='c', marker='o', linewidth=3)
plt.plot(months, shampoo, label='shampoo sales Data', color='m', marker='o', linewidth=3)
plt.plot(months, moisturizer, label='moisturizer sales Data', color='k', marker='o', linewidth=3)

plt.xlabel('Month Number')
plt.ylabel('Number of units Sold')
plt.xticks(months)
plt.yticks(np.arange(1000, 20000, 2000))
plt.legend()
plt.show()


# 4
df = pd.read_csv("sales_data.csv")
months = df['month_number'].tolist()
toothPaste = df['toothpaste'].tolist()

plt.scatter(months, toothPaste, label='toothpaste Sales Data', alpha=0.5)
plt.xlabel('Month Number')
plt.ylabel('Number of units Sold')
plt.xticks(months)
plt.yticks(np.arange(4500, 9000, 500))
plt.legend(loc='upper left')
plt.grid(True, linewidth=1, linestyle="--")
plt.show()


# 5
df = pd.read_csv("sales_data.csv")
months = df['month_number'].tolist()
bathingSoap = df['bathingsoap'].tolist()

plt.bar(months, bathingSoap, label='bathingsoap Sales Data')
plt.xlabel('Month Number')
plt.ylabel('Number of units Sold')
plt.xticks(months)
plt.yticks(np.arange((min(bathingSoap) - 100), (max(bathingSoap) + 500), 1000))
plt.legend(loc = 'upper left')
plt.show()


# 6
df = pd.read_csv("sales_data.csv")
profit_list = df['total_profit'].tolist()
profit_range = list(np.arange((min(profit_list) - 5000), (max(profit_list) + 5000), 50000))

plt.hist(profit_list, profit_range, label='Profit Data')
plt.xticks(profit_range)
plt.legend(loc='upper right')
plt.show()


# 7
df = pd.read_csv("sales_data.csv")
months = df['month_number'].tolist()
bathingSoap = df['bathingsoap'].tolist()
faceWash = df['facewash'].tolist()

plt.subplot(2,1,1)
plt.plot(months, bathingSoap, label='bathingsoap Sales Data', color='g', marker='o', linewidth=3)
plt.xticks(months)
plt.ylabel('Number of units Sold')
plt.legend(loc='upper left')
plt.subplot(2,1,2)
plt.plot(months, faceWash, label='facewash Sales Data', color='r', marker='o', linewidth=3)
plt.xticks(months)
#plt.title("facewash Sales Data")
plt.xlabel("Month Number")
plt.ylabel('Number of units Sold')
plt.legend(loc='upper left')

plt.show()


# Exercise SET 5
'''
# 1