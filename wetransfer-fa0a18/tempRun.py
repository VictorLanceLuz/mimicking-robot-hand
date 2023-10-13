import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import cv2
from torch.autograd import Variable
#from serial import Serial
import time
from datetime import datetime
startTime = datetime.now();
data_dir = 'hands'
test_transforms = transforms.Compose([transforms.Resize((100,100)),
                                      transforms.ToTensor(),
                                     ])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
to_pil = transforms.ToPILImage()

model=torch.load('trained.pth')
model.eval()

data = datasets.ImageFolder(data_dir, transform=test_transforms)
classes = data.classes

stream = cv2.VideoCapture(0)

#ser = Serial('COM6', 9600)
#time.sleep(2)

def predict_image(image):
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    output = model(input)
    prob = F.softmax(output, dim=1)
    print('\nclosed: '+str(prob[0][0]*100))
    print('open: '+str(prob[0][1]*100))
    print('spiderman: '+str(prob[0][2]*100))
    index = output.data.cpu().numpy().argmax()
    return index

while(True):
    ret, frame = stream.read()
    image = to_pil(frame)
    cv2.imshow('stream', frame)

    index = predict_image(image)
    #print(index)
    if index == 0:
        command = 'c'
#        ser.write(command.encode())
        print('closed')
    elif index == 1:
        command = 'o'
#        ser.write(command.encode())
        print('open')
    elif index == 2:
        command = 's'
#        ser.write(command.encode())
        print('spiderman')
    #print(ser.readline())
    #time.sleep(0.25)
    if cv2.waitKey(1) & 0xFF == ord('q'):   # Press 'q' to leave
        break
printf(datetime.now() - startTime)

stream.release()        #Release capture object
cv2.destroyAllWindows() #Destroy all the windows

