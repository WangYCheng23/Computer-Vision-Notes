import cv2
from PIL import Image
from model.model import parsingNet
from utils.common import merge_config
import torch
import scipy.special
import numpy as np
import torchvision.transforms as transforms
import json

row_anchor = [64,  68,  72,  76,  80,  84,  88,  92,  96, 100, 104, 108, 112,
            116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
            168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
            220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
            272, 276, 280, 284]
args, cfg = merge_config()
assert cfg.backbone in ['18','34','50','101','152','50next','101next','50wide','101wide']
cls_num_per_lane = 56

img_transforms = transforms.Compose([
    transforms.Resize((288, 800)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

def init():
    model_path = "/project/train/models/best.pth"
    net = parsingNet(pretrained=False, backbone=cfg.backbone, cls_dim=(cfg.griding_num + 1, cls_num_per_lane, 4),
                     use_aux=False).cuda()  # we dont need auxiliary segmentation in testing

    state_dict = torch.load(model_path, map_location='cpu')
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()
    return net

def process_image(net, input_image, args=None):
    img_h, img_w = input_image.shape[:2]
    y_samples = list(range(0, img_h, 10))
    if y_samples[-1]!= img_h-1:
        y_samples.append(img_h-1)
    
    img = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    x = img_transforms(img)
    with torch.no_grad():
        x = x.unsqueeze(0).cuda()
        out = net(x)
    
    col_sample = np.linspace(0, 800 - 1, cfg.griding_num)
    col_sample_w = col_sample[1] - col_sample[0]
    
    out_j = out[0].data.cpu().numpy()
    out_j = out_j[:, ::-1, :]
    prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
    idx = np.arange(cfg.griding_num) + 1
    idx = idx.reshape(-1, 1, 1)
    loc = np.sum(prob * idx, axis=0)
    out_j = np.argmax(out_j, axis=0)
    loc[out_j == cfg.griding_num] = 0
    out_j = loc
    
    lanes = []
    for i in range(out_j.shape[1]):
        if np.sum(out_j[:, i] != 0) > 2:
            x_list, y_list = [], []
            for k in range(out_j.shape[0]):
                if out_j[k, i] > 0:
                    x = out_j[k, i] * col_sample_w * img_w / 800 - 1
                    y = img_h * (row_anchor[cls_num_per_lane - 1 - k] / 288) - 1
                    x_list.append(x)
                    y_list.append(y)
            if len(x_list)>0:
                coeffs = np.polyfit(x_list, y_list, 1)  ###一阶多项式 y=a1x+a0 拟合，返回两个系数 [a1,a0]。
                fit_x = list(map(lambda x:(x-coeffs[1])/coeffs[0], y_samples))
                fit_x = list(map(int, fit_x))
                lanes.append(fit_x)
            
    return json.dumps({'model_data': {"lanes": lanes, "h_samples":y_samples}})


if __name__ == "__main__":
    
    imgpath = '/home/data/123/factory_in_15_2.jpg'
    input_image = cv2.imread(imgpath)
    vis = input_image.copy()
    net = init()
    result = process_image(net, input_image)
    result = json.loads(result)['model_data']
    
    for lane in result['lanes']:
        for i,x in enumerate(lane):
            cv2.circle(vis, (x, result['h_samples'][i]), 5, (0, 0, 255), -1)
    
    cv2.imwrite('result.jpg', vis)