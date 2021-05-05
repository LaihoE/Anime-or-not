import time
import cv2
import mss
import numpy
import torch
import torchvision
import torchvision.transforms as transforms

# title of our window
title = "FPS benchmark"
# set start time to current time
start_time = time.time()
# displays the frame rate every 2 second
display_time = 2
# Set primarry FPS to 0
fps = 0
# Load mss library as sct
sct = mss.mss()
# Set monitor size to capture to MSS
monitor = {"top": 40, "left": 0, "width": 800, "height": 640}
# Set monitor size to capture
mon = (0, 40, 800, 640)


model_path="weebmodel3.pt"
path=r'C:\Users\emill\PycharmProjects/CLIP/livepics'
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dataset_path=path
train_transforms = transforms.Compose([transforms.Resize((184,184)),
                                      transforms.ToTensor()])
train_dataset=torchvision.datasets.ImageFolder(root=train_dataset_path,transform=train_transforms)
model=torch.load(model_path)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
model.eval()




def screen_recordMSS():
    global fps, start_time
    while True:
        # Get raw pixels from the screen, save it to a Numpy array
        img = numpy.array(sct.grab(monitor))
        # to get real color we do this:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(r'C:\Users\emill\PycharmProjects/CLIP/livepics/xd/this.jpg',img=img)
        with torch.no_grad():
            for x, y in train_loader:
                x = x.to(device=device)
                scores = model(x)
                _, predictions = scores.max(1)
                print(scores)
                if predictions.item() == 1:
                    can = "Anime pic"
                else:
                    can = "Normal pic"
                if can == "Anime pic":
                    print("WARNING")
                    jesus = cv2.imread("bigjesus.jpg")
                    cv2.imshow("jesus", jesus)


        cv2.imshow(title, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        fps+=1
        TIME = time.time() - start_time
        if (TIME) >= display_time :
            print("FPS: ", fps / (TIME))
            fps = 0
            start_time = time.time()
        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break

screen_recordMSS()

