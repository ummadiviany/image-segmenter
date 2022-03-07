import torch
from torchvision.models.segmentation import fcn_resnet50
from torchvision.utils import draw_segmentation_masks
from PIL import Image
import io
from torchvision.transforms import transforms
from torchvision.utils import save_image
import torchvision.transforms.functional as F

def get_model():
    model = fcn_resnet50(pretrained=True, progress=True)
    model = model.eval()
    return model

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    image.convert("RGB")
    image.save("static/org_pil.jpg")
    print("Saved Original image successfully")
    return my_transforms(image).unsqueeze(0)



sem_classes = [
    '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}


def get_prediction(image_bytes):
    try:
        print('Entered here')
        tensor = transform_image(image_bytes=image_bytes)
        model = get_model()
        output = model(tensor)['out']
        print(output.shape)
    except Exception:
        return 0, 'Failed'

    print(output.shape, output.min().item(), output.max().item())

    normalized_masks = torch.nn.functional.softmax(output, dim=1)
    num_classes = normalized_masks.shape[1]
    class_dim = 1
    all_classes_masks = normalized_masks.argmax(class_dim) == torch.arange(num_classes)[:, None, None, None]
    all_classes_masks = all_classes_masks.swapaxes(0, 1)

    dogs_with_masks = [
        draw_segmentation_masks(img, masks=mask, alpha=.6)
        for img, mask in zip(tensor.to(dtype=torch.uint8), all_classes_masks)
    ]
    img = dogs_with_masks[0]
    img = img.detach()
    img = F.to_pil_image(img)
    img.save("static/masked.jpg")
    print("Saved masked image successfully")
    return None
