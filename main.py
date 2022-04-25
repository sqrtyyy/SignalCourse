import img_processing
import cv2
import os


def main():
    for i, file in enumerate(os.listdir("Data/InArea")):
        print(file)
        img = cv2.imread(f"Data/InArea/{file}")
        finder = img_processing.Finder(img)
        finder.save_result(f"out/InArea/result_{i}")

    for i, file in enumerate(os.listdir("Data/OutOfArea")):
        print(file)
        img = cv2.imread(f"Data/OutOfArea/{file}")
        finder = img_processing.Finder(img)
        finder.save_result(f"out/OutOfArea/result_{i}")

if __name__ == "__main__":
    main()
