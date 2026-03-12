import torch

def main():
    inputs = torch.tensor(
        [
            [0.43, 0.15, 0.89],
            [0.55, 0.87, 0.66],
        ]
    )
    query = inputs[1]
    for i, x_i in enumerate(inputs):
        print(torch.dot(x_i, query))

if __name__ == "__main__":
    main()