from io import BytesIO

import max.driver
import numpy as np
import requests
import torch
from max.driver import Accelerator
from max.graph import DeviceRef
from PIL import Image

# torchvision is a demo-only extra, not a project dependency (pip install it
# yourself to run this script).
from torchvision import models, transforms  # ty: ignore[unresolved-import]

from torch_mojo_backend.torch_compile_backend.exporter import export_to_max_graph

model = models.vgg11(pretrained=True)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)
max_model = export_to_max_graph(model, (dummy_input,), force_device=DeviceRef.GPU(0))

dummy_input_max_gpu = max.driver.Buffer.from_numpy(
    np.random.randn(1, 3, 224, 224).astype(np.float32)
).to(Accelerator(0))
print(max_model(dummy_input_max_gpu))


# usage:
# max_model(np.random.randn(1,3,224,224))


preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def load_image(image_path_or_url: str) -> Image.Image:
    if image_path_or_url.startswith("http"):
        response = requests.get(image_path_or_url)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_path_or_url)

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def load_imagenet_labels() -> list[str]:
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    response = requests.get(url)
    labels = response.text.strip().split("\n")
    return labels


def predict_image(image_path_or_url: str, top_k: int = 5):
    image = load_image(image_path_or_url)

    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension
    input_batch = max.driver.Buffer.from_dlpack(input_batch).to(Accelerator(0))
    output = max_model(input_batch)

    output = torch.tensor(output[0].to_numpy())

    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    top_prob, top_class = torch.topk(probabilities, top_k)

    labels = load_imagenet_labels()

    print("Top Predictions: (boxer should come first)")
    for i in range(top_k):
        class_idx = int(top_class[i].item())
        prob = top_prob[i].item()
        label = labels[class_idx]
        print(f"{i + 1:2d}. {label:30s} ({prob:.3f})")


if __name__ == "__main__":
    image_url = "https://raw.githubusercontent.com/jigsawpieces/dog-api-images/refs/heads/main/boxer/n02108089_10229.jpg"
    predict_image(image_url, top_k=5)
