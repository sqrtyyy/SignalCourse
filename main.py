import time

import intelligent_placer
import os


def main():
    total_start = time.time()
    for i, file in enumerate(os.listdir("Data/InArea")):
        start = time.time()
        print(file)
        print(intelligent_placer.check_image(f"Data/InArea/{file}"))
        print(f"Elapsed: {time.time() - start}")
    print("________________________________________")
    print(f"Total: {time.time() - total_start}")
if __name__ == "__main__":
    main()
