import time

def cazzo_peppe(x: int) -> int:
    if x == 0:
        return 0

    if x == 10:
        print("ciao")

    return cazzo_peppe(x - 1)



if __name__ == "__main__":
    cazzo_peppe(100)