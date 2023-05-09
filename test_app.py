import os
import requests
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor


def preprocess_image(image_path):
    transform = Compose([
        Resize((512, 512)),
        ToTensor()
    ])
    img = Image.open(image_path)
    return transform(img)


def save_image_as_temp(img, image_path):
    temp_path = f"temp_{os.path.basename(image_path)}"
    img = img.mul(255).byte().permute(1, 2, 0).numpy()
    img = Image.fromarray(img)
    img.save(temp_path)
    return temp_path


def get_prediction(image_path):
    img = preprocess_image(image_path)
    temp_path = save_image_as_temp(img, image_path)

    with open(temp_path, "rb") as f:
        response = requests.post(
            "http://localhost:8000/predict",
            files={"file": (os.path.basename(image_path), f, "image/jpeg")}
        )

    os.remove(temp_path)
    return response.json()


if __name__ == "__main__":
    image_path = input("Enter the image path: ")
    if not os.path.exists(image_path):
        print("File does not exist. Exiting...\n")
        exit()
    prediction = get_prediction(image_path)
    print(
        f"Class ID: {prediction['class_id']}, Class Name: {prediction['class_name']}\n")
