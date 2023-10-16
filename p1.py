#Date:11-10-2023
print("Hello World")
from platform import python_version
print(python_version())

a="Welcome to code unnati 2.0"
def hello():
    a=20
    print(a)
    return
print(a)

a=10
print(a)
print(type(a))

b,c,d = 11,12,13
print(a,b,c,d)

x=5
x+=3 
#x=x+3
print(x)

x=5
y=3
print(x==y)
print(x!=y)


name="Romil"
course = 'code unnati'
print(f"Hello,my name is {name} and i registered for {course}")


str1 = "FACE"
print(str1[0:2])
print(str1[3:0:-1])

cars ={
    "brand":"Mahendra",
    "model":"Thar"
}
print(cars.keys())

cars["color"]="black"

subjects1={'Physics','Chemistry','Maths','Hindi'}
print(subjects1)
print(type(subjects1))

subjects2={'History','ss','biology','Hindi'}
print(subjects2)

print(subjects1.union(subjects2))
print(subjects1.intersection(subjects2))

fruits=['apples','banans','cherry']
print(len(fruits))

name='Shivaji Maharaj'
print(len(name))

theater={
    "Movie":"Pushpa",
    "Actor":"Allu Arjun",
    "year":2021
}
print(sorted(theater))

numbers=[1,3,8,7,5,4,6,8,5]
print(max(numbers))

numbers=[1,3,8,7,5,4,6,8,5]
print(sum(numbers))

print(dir(list))

'''print(str(list))
print(tuple(list))'''

n=10
print(type(n))

n=float(n)
print(type(n))

n=10
n=str(n)
print(type(n))

fruits=['apples','banans','cherry']
print(type(fruits))

fruits=tuple(fruits)
print(type(fruits))

fruits_list=['apple','banans','orange','grape','mango']
print("First:",fruits_list[0])
print("Last:",fruits_list[4])
print("sec:",fruits_list[1])
print("third:",fruits_list[2])
'''
num1=int(input("Enter a first number:"))
num2=int(input("Enter a second number:"))
'''
def add_numbers(num1,num2):
    result = num1+num2
    print(result)
    return

'''add_numbers(num1,num2)'''

number=10
if number>0:
    print("Number is positive")
elif number == 0:
    print("Zero")
else:
    print("Negative number")

print("This statement is always executed")

counter=0

while counter<3:
    print("Hello world")
    counter=counter+1

for i in range(10):
    print(i)

for letter in "Python":
    print(letter)

fruits=['apples','banans','cherry']

for fruit in fruits:
    print(fruit)

x=["Physics","chemistry","maths"]
del x[0]
print(x)

i=1
while i < 9:
    print(i)
    if i==3:
        break
    i+=1

i=1
while i < 10:
    i+=1
    
    if i==3:
        continue
    print(i)

def calculate_total_bill(items,prices):
    if len(items) != len(prices):
        return "Error:The number of items and prices should be same"
    
x=lambda a:a+10
print(x(5))

x=lambda a,b:a*b
print(x(5,6))
'''
try:
    numerator = int(input("Enter a numerator: "))
    denomminator = int(input("Enter a denomminator: "))
    result=numerator/denomminator
    print("Result:",result)
except ZeroDivisionError:
    print("Error : Cannot divide by zero")
except ValueError:
    print("Error Please enter valid integers for numerator and denominator")
'''

'''
try:
    number = int(input("Enter a number: "))
except ValueError:
    print("Error : Please enter a valid integer")
else:
    square = number ** 2
    print("Square of the number:",square)
'''

try:
    file = open("data.txt","r")
    content = file.read()
    print(content)
except FileNotFoundError:
    print("Error: File not found")
finally:
    if 'file' in locals():
        file.close()

import numpy as np
print(np.__version__)

#Date:12-10-2023
'''
import numpy as np
a=np.array([1,2,3,4,5])
np.save('output.py',a)

import numpy as np
b=np.load('output.py')
print(b)
'''

'''
import numpy as np
a=np.array([1,2,3,4,5])
np.savetxt('out.txt',a)

import numpy as np
arr=np.arange(1,10,2)
print("Element of array: ",arr)
arr1=arr[np.array([4,0,2,-1,-2])]
print("Indexed element of array arr: ",arr1)
'''

import numpy as np
arr=np.arange(12)
arr1=arr.reshape(3,4)
print("Array arr1:\n",arr1)
print("Element at rth row eth column of arr1 is",arr1[0,0])
print("aba",arr1[1,2])


import numpy as np
arr = np.arange(6)
print("array arr:", arr)
print("sliced element of array: ", arr[1:5])

import numpy as np
arr=np.arange(12)
arrl=arr.reshape(3,4)
print("Array arri: \n", arr1)
print("\n")
print("elements of 1st row and 1st column upto last column \n", arr1[1:,1:4])





import numpy as np
a = np.array([[0.0,0.0,0.0],[10.0,10.0,10.0], [20.0,20.0,20.0], [30.0,30.0,30.0]])
b= np.array([1.0,2.0,3.0])
print("First array:")
print(a)
print("\n")
print("Second array:") 
print(b)
print("\n")
print("First Array Second Array")
print(a + b)



#Python program to demonstrate
#Structured array
import numpy as np
a = np.array([('Sana', 2, 21.0), ('Mansi', 7, 29.0)], dtype=[('name', (np. str_, 10)), ('age', np.int32), ('weight', np.float64)])
print(a)

# Minimum Value
import numpy as np
b = np.array([[20,21,22], [25, 26, 27], [30,31,32]])
print(b, "\n")
print(b.dtype, "\n")
print(b.itemsize)

print(b, "\n")
print(np.min(b))
print (np.min(b, axis=0))
print (np.min(b, axis=1))
print(type(b))

import numpy as np
arr = np.arange(0,10)
print(arr)

#-------------------------------------------------------------------pandas-------------------------------
import pandas as pd
print(pd.__version__)

import pandas as pd
import numpy as np
ser=pd.Series()
print(ser)
#simple array
data = np.array(['g','e','e','k','s'])
ser = pd.Series(data)
print(ser)

import pandas as pd
#a simple list
list=['g','e','e','k','s']
#creare series from a list
ser=pd.Series(list)
print(ser)

#importing pandas package
import pandas as pd
data = pd.read_csv("airlines.csv")
print(data)

import pandas as pd
#List of strings
lst = ['Geeks', 'For', 'Geeks', 'is', 'portal', 'for', 'Geeks']

#Calling DataFrame constructor on List

df = pd.DataFrame(lst)

print(df)



#Python code demonstrate creating

# DataFrame from dict narray / Lists #By default addresses.

import pandas as pd

# intialise data of lists.
data = { 'Name': ['Tom', 'nick', 'krish', 'jack'], 'Age': [20, 21, 19, 18]}
# Create DataFrame
df = pd.DataFrame(data)
#Print the output.
print(df)




import pandas as pd
#Create dataframe
info= pd.DataFrame({"P":[4, 7, 1, 8, 9],
                    "Q":[6, 8, 10, 15, 11],
                    "R":[17, 13, 12, 16, 14],
                    "S":[15, 19, 7, 21, 9]},
                    index =["Parker", "William", "Smith", "Terry", "Phill"])
#Print dataframe
print(info)

info.reindex (["A","B","C","D","E"], fill_value=100)
print(info)

import pandas as pd
import numpy as np
unsorted_df = pd.DataFrame(np.random.randn (10,2),index=[1,4,6,2,3,5,9,8,8,7], columns=['col1','col2'])
sorted_df=unsorted_df.sort_index()
print(sorted_df)

import pandas as pd
import numpy as np
unsorted_df = pd.DataFrame(np.random.randn (10,2),index=[1,4,6,2,3,5,9,8,8,7], columns=['col1','col2'])
sorted_df=unsorted_df.sort_index(ascending=False)
print(sorted_df)

import pandas as pd
import numpy as np
unsorted_df = pd.DataFrame(np.random.randn (10,2),index=[1,4,6,2,3,5,9,8,8,7], columns=['col1','col2'])
sorted_df=unsorted_df.sort_index(axis=1)
print(sorted_df)

import pandas as pd
import numpy as np
unsorted_df = pd.DataFrame(np.random.randn (10,2),index=[1,4,6,2,3,5,9,8,8,7], columns=['col1','col2'])
sorted_df=unsorted_df.sort_values(by='col1')
print(sorted_df)

import pandas as pd
import numpy as np
s = pd.Series(['Tom','William Rick','John','Alber@t',np.nan,'1234','SteveSmith'])
print(s.str.lower())
print(s.str.upper())

#Functions
import pandas as pd
import numpy as np
data=pd.read_csv("airlines.csv")
print(data.describe())

#Date:13-10-2023
'''matplotlib'''
'''
import matplotlib.pyplot as plt
# x axis values
X = [1,2,3]
#corresponding y axis values
y = [2,4,1]
# plotting the points
plt.plot(x,y)
#naming the x axis
plt.xlabel('x - axis')
#naming the y axis
plt.ylabel('y-axis') # giving a title to my graph
plt.title('My first graph!')
#function to show the plot
plt.show()


import matplotlib.pyplot as plt
import numpy as np
x = np.arange(0, 10, 0.1)
y = np.sin(x)
# figures in matplotlib using figure()
plt.figure(figsize=(10, 8))
plt.plot(x, y)
plt.show()



import matplotlib.pyplot as plt
import numpy as np
x = np.arange(0, 10, 0.1)
y = np.cos(x)
# figures in matplotlib using figure()
plt.figure(figsize=(10, 8))
plt.plot(x, y)
plt.show()


import matplotlib.pyplot as plt
import numpy as np
x = np.arange(0, 10, 0.1)
y = np.tan(x)
# figures in matplotlib using figure()
plt.figure(figsize=(10, 8))
plt.plot(x, y)
plt.show()
'''

'''
import matplotlib.pyplot as plt
import numpy as np
ypoints= np.array([3, 8, 1, 18])
plt.plot(ypoints, marker = 'o')
plt.show()

import matplotlib.pyplot as plt
import numpy as np
ypoints= np.array([3, 8, 1, 18])
plt.plot(ypoints, color = 'r')
plt.show()

import matplotlib.pyplot as plt
import numpy as np
ypoints= np.array([3, 8, 1, 18])
plt.plot(ypoints, marker = 'o', ms= 20, mec='r')
plt.show()
'''
'''
import matplotlib.pyplot as plt
x=[5,2,9,4,7]
y=[10,5,8,4,2]

plt.bar(x,y)
plt.show()
'''

'''
import matplotlib.pyplot as plt
#values of x and y ases
x=[5, 10, 15, 20, 25, 38, 35, 40, 45, 50]
y= [1, 4, 3, 2, 7, 6, 9, 8, 18, 5]
plt.plot(x, y, 'g')
plt.xlabel("x")
plt.ylabel('y')
#here we set the size for ticks, rutation and color value
plt.tick_params(axis="x", labelsize=18, labelrotation=68, labelcolor="blue")
plt.tick_params(axis="y", labelsize=12, labelrotation=20, labelcolor="black")
plt.show()
'''



'''
#importing modules
import numpy as np
import matplotlib.pyplot as plt
#Y-axis velues
y1 = [2, 3, 4.5]
#Y-axis values
y2= [1, 1.5, 5]
#Function to plot
plt.plot(y1)
plt.plot(y2)
#Function add a Legend
plt.legend(["blue", "grean"], loc="lower right")
#function to show the plat
plt.show()
'''

'''
import matplotlib.pyplot as plt
import numpy as np
data=np.random.normal(170,10,250)
plt.hist(data);
plt.show()
plt.hist(dat,bins=20)
pl.show()

'''

'''
import matplotlib.pyplot as plt
import numpy as np
#draw plot as fig
fig, ax = plt.subplots()
x=np.linspace(0, 20, 1000)
ax.plot(x, np.cos(x))
ax.axis('equal')
#text annotation to the plot where it indicate maximum value of the curv
ax.annotate('local maximum', xy=(6.28, 1), xytext=(10, 4), arrowprops=dict(facecolor='black', shrink=0.05))
#text annotation to the plot where it indicate minimum value of the curv
ax.annotate('local minimum', xy=(5* np.pi, -1), xytext=(2, -6), arrowprops=dict(arrowstyle="->", connectionstyle="angle3, angleA=0, angleB=-90"));
plt.show()
'''

#import libraries
import matplotlib.pyplot as plt
import numpy as np
#create dataset
cars = ['Audi','BMW','Ford','Tesla','Jaguar','Mercedes']
data=[23,17,35,29,12,41]
flg=plt.figure(figsize=(10,7))
plt.pie(data,labels=cars)
plt.show()

import matplotlib.pyplot as plt
#exponential function y = 10^x
data= [10**i for i in range(5)]
#convert y-axis to Logarithmic scale
plt.yscale("log")
plt.plot(data)
plt.show()

