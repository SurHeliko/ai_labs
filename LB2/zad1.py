import numpy as np

def main():
    size = 10
    list = np.random.randint(0,10,size)
    print(list)

    counter = 0
    for i in list:
        if i % 2 == 0:
            counter += i

    print(counter)

if __name__ == "__main__":
     main()