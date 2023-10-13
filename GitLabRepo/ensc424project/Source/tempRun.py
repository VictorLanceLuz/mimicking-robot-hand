import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import cv2
from torch.autograd import Variable
#from serial import Serial
import time

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
time.sleep(2)

def predict_image(image):
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    output = model(input)
    index = output.data.cpu().numpy().argmax()
    return index
#def get_random_images(num):
#    indices = list(range(len(data)))
#    np.random.shuffle(indices)
#    idx = indices[:num]
    
#    sampler = SubsetRandomSampler(idx)
#    loader = torch.utils.data.DataLoader(data, sampler=sampler, batch_size=num)
#    dataiter = iter(loader)
#    images, labels = dataiter.next()
#    return images, labels

while(True):
    ret, frame = stream.read()
    image = to_pil(frame)
    cv2.imshow('stream', frame)

    index = predict_image(image)
    #print(index)
    if index == 0:
        command = 'c'
        #ser.write(command.encode())
        print('closed')
    elif index == 1:
        command = 'o'
        #ser.write(command.encode())
        print('open')
    #print(ser.readline())
    time.sleep(0.5)
    if cv2.waitKey(1) & 0xFF == ord('q'):   # Press 'q' to leave
        break

stream.release()        #Release capture object
cv2.destroyAllWindows() #Destroy all the windows



#images, labels = get_random_images(4)
#fig=plt.figure(figsize=(10,10))
#for ii in range(len(images)):
#    image = to_pil(images[ii])
#    index = predict_image(image)
#    sub = fig.add_subplot(1, len(images), ii+1)
#    res = int(labels[ii]) == index
#    sub.set_title(str(classes[index]) + ":" + str(res))
#    plt.axis('off')
#    plt.imshow(image)
#plt.show()