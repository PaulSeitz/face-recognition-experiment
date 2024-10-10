import cv2 as cv
import torchvision.transforms as transforms

def resize_image(img, size=(128, 128)):

    # convert to black and white (i.e. delete the 3rd dimension)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Define the transformation to apply to the face images
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Apply the transformation
    img_tensor = transform(img)

    return img_tensor
