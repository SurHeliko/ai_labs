import random
import torch

n = 2

def main():
    X = torch.randint(1, 3, (3, 6), dtype=torch.int32, device=torch.device('cpu'))
    print(X)

    X = X.to(dtype=torch.float32)

    X.requires_grad = True
    print(X)

    Y = X**n
    print(Y)
    Y = Y * random.randint(1,10)
    print(X)
    Y = Y.exp()
    print(Y)
    Y.backward(torch.ones(3,6))
    print(X.grad)

        
if __name__ == '__main__':
    main()