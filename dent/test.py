import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import cv2
import numpy as np
import torch
import torchvision
from pytorch_grad_cam import AblationCAM, EigenCAM
from pytorch_grad_cam.ablation_layer import AblationLayerFasterRCNN
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_accross_batch_and_channels, scale_cam_image
import matplotlib.pyplot as plt 
import requests
from model.transformer import Encoder, Dent_Pt
from PIL import Image


def predict(input_tensor, model, device, detection_threshold, coco_names):
    outputs = model(input_tensor)
    logits, boxes = outputs #torch.softmax(outputs['pred_logits'], dim=-1)[0], outputs['pred_boxes'][0]
    score, labels = torch.max(logits, dim=-1)
    l = labels < len(coco_names)
    labels = labels[l]
    score = score[l]
    boxes = boxes[l]
    pred_classes = [coco_names[i] for i in labels.detach().cpu().numpy()]
    pred_labels = labels.detach().cpu().numpy()
    pred_scores = score.detach().cpu().numpy()
    pred_bboxes = boxes.detach().cpu().numpy()
    
    boxes, classes, labels, indices = [], [], [], []
    for index in range(len(pred_scores)):
        if pred_scores[index] >= detection_threshold:
            boxes.append(pred_bboxes[index])
            classes.append(pred_classes[index])
            labels.append(pred_labels[index])
            indices.append(index)
    
    # boxes = box_cxcywh_to_xyxy(torch.tensor(boxes))
    _, _, _,h, w = input_tensor.shape
    # boxes[...,::2] *= w
    # boxes[...,1::2] *= h
    print(boxes, classes, labels, indices)
    return boxes, classes, labels, indices


class FasterRCNNBoxScoreTarget:

  def __init__(self, labels, bounding_boxes, iou_threshold=0.5):
      self.labels = labels
      self.bounding_boxes = bounding_boxes
      self.iou_threshold = iou_threshold

  def __call__(self, model_outputs):
      output = torch.Tensor([0])
      if torch.cuda.is_available():
          output = output.cuda()

      if len(model_outputs["boxes"]) == 0:
          return output

      for box, label in zip(self.bounding_boxes, self.labels):
          box = torch.Tensor(box[None, :])
          if torch.cuda.is_available():
              box = box.cuda()

          ious = torchvision.ops.box_iou(box, model_outputs["boxes"])
          index = ious.argmax()
          if ious[0, index] > self.iou_threshold and model_outputs["labels"][index] == label:
              score = ious[0, index] + model_outputs["scores"][index]
              output = output + score
      return output

def ttty(x):
    print(x.size())
    return x

def major_stuff(m, tgets, colors, input_tensor, image_float_np):
  print(input_tensor.shape)
  grayscale_cam = m(input_tensor, targets=tgets)
  # Take the first image in the batch:
  grayscale_cam = grayscale_cam[0, :]
  cam_image = show_cam_on_image(image_float_np, grayscale_cam, use_rgb=True)
  # And lets draw the boxes again:
  return cam_image
  # image_with_bounding_boxes = draw_boxes(boxes, labels, classes, cam_image, colors)
  # return image_with_bounding_boxes
  


def stuff(colors, model, labels, boxes, input_tensor, image_float_np, classes):
#   for i, n in model.named_parameters():
#       if n.requires_grad:
#         print(i)
  target_layers = [model.encoder.backbone.layer4[-1]]
  # target_layers = [model.transformer.encoder.layers[-1].norm2]
  targets = [FasterRCNNBoxScoreTarget(labels=labels, bounding_boxes=None)]
  cam = EigenCAM(model,
                target_layers, 
                use_cuda=torch.cuda.is_available(),
                reshape_transform=ttty)
  cam_image=major_stuff(cam, targets, colors, input_tensor, image_float_np)
#   image_with_bounding_boxes = draw_boxes(boxes, labels, classes, cam_image, colors)
  return cam_image #image_with_bounding_boxes


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1).numpy()

def draw_boxes(boxes, labels, classes, image, COLORS):
    
    for i, box in enumerate(boxes):
        color = COLORS[labels[i]+1]
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2
        )
        cv2.putText(image, classes[i], (int(box[0]), int(box[1] - 5)),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.8, color, 1,
                    lineType=cv2.LINE_AA)
    return image

def get_model():
    encoder = Encoder(hidden_dim=256,num_encoder_layers=6, nheads=8)
    model = Dent_Pt(encoder, hidden_dim=256, num_class=2)
    return model

def pre():
  coco_names = ['__background__', 'Fovea', 'SCR']
  # This will help us create a different color for each class
  COLORS = np.random.uniform(0, 255, size=(len(coco_names)+5, 3))


#   image_url = "https://raw.githubusercontent.com/jacobgil/pytorch-grad-cam/master/examples/both.png"
#   image = np.array(Image.open(requests.get(image_url, stream=True).raw).resize((496,220)))
  image = np.array(Image.open('imgc.jpg').convert(mode='RGB'))
  print(image.shape, type(image), type(image[0][0][0]))
  image_float_np = np.float32(image) / 255
  transform = torchvision.transforms.Compose([
      torchvision.transforms.ToTensor(),
  ])

  input_tensor = transform(image)
  print(input_tensor.shape, image.shape, type(image), type(image[0][0][0]))
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  input_tensor = input_tensor.to(device)
  input_tensor = torch.stack([input_tensor, input_tensor, input_tensor]).unsqueeze(0)
  input_tensor = input_tensor.permute(1,0,2,3,4)
  print(input_tensor.shape)
  # Add a batch dimension:
#   input_tensor = (input_tensor.unsqueeze(0)).unsqueeze(0)

  # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
  #   model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
  #   model.eval().to(device)

  model = get_model()
  model.eval().to(device)

  # Run the model and display the detections
  boxes, classes, labels, indices = predict(input_tensor=input_tensor, model=model, device=device, detection_threshold=0.5, coco_names=coco_names)
#   print(boxes[:3], classes, labels, type(boxes), type(classes), type(labels))
#   boxes[:3] = np.array([[5,30,72,195], [94,28,353,191], [393,29,632,194]])
#   classes[:3] = ['SCR', 'SCR', 'SCR']
#   labels[:3] = [2,2,2]
#   image = draw_boxes(boxes, labels, classes, image, COLORS)

  atn = stuff(COLORS, model, None, None, input_tensor, image_float_np, classes)
  return image, atn


image, atn = pre()
# Show the image:
plt.imsave('imgc_filelike.png', image)
plt.imsave('imgc_atnlike.png', atn)


