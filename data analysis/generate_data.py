import random

def generate(filename, x, y, low, high):

    file = open(filename, "w")

    for i in range (0, y, 1):

        temp = ""
        
        for j in range (0, x - 1, 1):

            temp = str(random.uniform(low, high)) +  ","  + temp

        temp = temp + str(random.uniform(low, high))
        file.write(temp + "\n")
