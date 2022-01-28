import os
import onnxruntime
from PIL import Image
import torchvision.transforms as transforms
import torch


def predict_image(image_path):
    this_dir, _ = os.path.split(__file__)
    model_path = os.path.join(this_dir, "anime_or_not_model.onnx")
    ort_session = onnxruntime.InferenceSession(model_path)
    img = Image.open(image_path)
    width, height = img.size
    assert width > 224 and height > 224, "Image should be at least 224x224"

    test_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                          transforms.ToTensor(),
                                          # Imagenet mean and std
                                          transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                          ])
    transformed_img = test_transforms(img)
    # reshape to create a batch dimension ie. (3,224, 224) -> (1,3,224, 224)
    transformed_img = torch.unsqueeze(transformed_img, dim=0)
    # Inputs to onnx type
    ort_inputs = {ort_session.get_inputs()[0].name: transformed_img.detach().cpu().numpy() if transformed_img.requires_grad else transformed_img.cpu().numpy()}
    # "Forward pass"
    ort_outs = ort_session.run(None, ort_inputs)
    # "Index with highest confidence"
    t = torch.tensor(ort_outs[0][0])
    return round(torch.softmax(t, dim=0)[1].item() * 100,2)


if __name__ == "__main__":
    image_path = 'weebs/img.png'
    prediction = anime_or_not(image_path)
    print(prediction)
