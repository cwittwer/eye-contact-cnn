import dlib
import cv2
import argparse, os, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import pandas as pd
import numpy as np
from model import model_static
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from colour import Color

import cv2
import imagezmq as iz
import socket
import threading
from queue import Queue

parser = argparse.ArgumentParser()

parser.add_argument('--video', type=str, help='input video path. live cam is used when not specified')
parser.add_argument('--face', type=str, help='face detection file path. dlib face detector is used when not specified')
parser.add_argument('--model_weight', type=str, help='path to model weights file', default='data/model_weights.pkl')
parser.add_argument('--jitter', type=int, help='jitter bbox n times, and average results', default=0)
parser.add_argument('-save_vis', help='saves output as video', action='store_true')
parser.add_argument('-display_off', help='do not display frames', action='store_true')

args = parser.parse_args()

CNN_FACE_MODEL = 'data/mmod_human_face_detector.dat' # from http://dlib.net/files/mmod_human_face_detector.dat.bz2

image_hub = iz.ImageHub()
sender = iz.ImageSender(connect_to='tcp://192.168.133.64:5556')

name_host=socket.gethostname()

def bbox_jitter(bbox_left, bbox_top, bbox_right, bbox_bottom):
    cx = (bbox_right+bbox_left)/2.0
    cy = (bbox_bottom+bbox_top)/2.0
    scale = random.uniform(0.8, 1.2)
    bbox_right = (bbox_right-cx)*scale + cx
    bbox_left = (bbox_left-cx)*scale + cx
    bbox_top = (bbox_top-cy)*scale + cy
    bbox_bottom = (bbox_bottom-cy)*scale + cy
    return bbox_left, bbox_top, bbox_right, bbox_bottom


def drawrect(drawcontext, xy, outline=None, width=0):
    (x1, y1), (x2, y2) = xy
    points = (x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)
    drawcontext.line(points, fill=outline, width=width)

def infer(args):

    video_path = args.video
    face_path = args.face
    model_weight = args.model_weight
    jitter = args.jitter
    vis = args.save_vis
    display_off = args.display_off

    global q
    q = Queue()
    # set up vis settings
    red = Color("red")
    colors = list(red.range_to(Color("green"),10))
    font = ImageFont.truetype("data/arial.ttf", 40)

    # set up output file
    if vis:
        outvis_name = os.path.basename(video_path).replace('.avi','_output.avi')
        outvid = cv2.VideoWriter(outvis_name,cv2.VideoWriter_fourcc('M','J','P','G'), 15, (1280,720))

    # set up face detection mode
    if face_path is None:
        facemode = 'DLIB'
    else:
        facemode = 'GIVEN'
        column_names = ['frame', 'left', 'top', 'right', 'bottom']
        df = pd.read_csv(face_path, names=column_names, index_col=0)
        df['left'] -= (df['right']-df['left'])*0.2
        df['right'] += (df['right']-df['left'])*0.2
        df['top'] -= (df['bottom']-df['top'])*0.1
        df['bottom'] += (df['bottom']-df['top'])*0.1
        df['left'] = df['left'].astype('int')
        df['top'] = df['top'].astype('int')
        df['right'] = df['right'].astype('int')
        df['bottom'] = df['bottom'].astype('int')

    if facemode == 'DLIB':
        cnn_face_detector = dlib.cnn_face_detection_model_v1(CNN_FACE_MODEL)
    frame_cnt = 0

    # set up data transformation
    test_transforms = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # load model weights
    model = model_static(model_weight)
    model_dict = model.state_dict()
    snapshot = torch.load(model_weight, map_location=torch.device('gpu'))
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)

    model.gpu()
    model.train(False)

    try:
        # video reading loop
        while(True):

            while q.qsize() > 1:
                q.get()
            
            if not q.empty():
                frame = q.get()
                height, width, channels = frame.shape
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                frame_cnt += 1
                bbox = []
                if facemode == 'DLIB':
                    dets = cnn_face_detector(frame, 1)
                    for d in dets:
                        l = d.rect.left()
                        r = d.rect.right()
                        t = d.rect.top()
                        b = d.rect.bottom()
                        # expand a bit
                        l -= (r-l)*0.2
                        r += (r-l)*0.2
                        t -= (b-t)*0.2
                        b += (b-t)*0.2
                        bbox.append([l,t,r,b])
                elif facemode == 'GIVEN':
                    if frame_cnt in df.index:
                        bbox.append([df.loc[frame_cnt,'left'],df.loc[frame_cnt,'top'],df.loc[frame_cnt,'right'],df.loc[frame_cnt,'bottom']])

                frame = Image.fromarray(frame)
                for b in bbox:
                    face = frame.crop((b))
                    img = test_transforms(face)
                    img.unsqueeze_(0)
                    if jitter > 0:
                        for i in range(jitter):
                            bj_left, bj_top, bj_right, bj_bottom = bbox_jitter(b[0], b[1], b[2], b[3])
                            bj = [bj_left, bj_top, bj_right, bj_bottom]
                            facej = frame.crop((bj))
                            img_jittered = test_transforms(facej)
                            img_jittered.unsqueeze_(0)
                            img = torch.cat([img, img_jittered])

                    # forward pass
                    output = model(img.gpu())
                    if jitter > 0:
                        output = torch.mean(output, 0)
                    score = torch.sigmoid(output).item()

                    coloridx = min(int(round(score*10)),9)
                    draw = ImageDraw.Draw(frame)
                    drawrect(draw, [(b[0], b[1]), (b[2], b[3])], outline=colors[coloridx].hex, width=5)
                    draw.text((b[0],b[3]), str(round(score,2)), fill=(255,255,255,128), font=font)

                frame = np.asarray(frame) # convert PIL image back to opencv format for faster display
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                sender.send_image(name_host, frame)
    except:
        if vis:
            outvid.release()
        print('DONE!')


if __name__ == "__main__":
    
    x = threading.Thread(target=infer, args=(args,))
    x.start()
    #infer(args)

    try:
        while True:
            name, image = image_hub.recv_image()
            q.put(image)
            
            image_hub.send_reply(b'ok')

            c = cv2.waitKey(1)
    except KeyboardInterrupt:
        x.join()
        exit()
    except:
        x.join()
        exit()
