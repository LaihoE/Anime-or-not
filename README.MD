# Neural Network for classifying if images include anime
## Installing
```
pip install anime-or-not
```
##Useage
```python
from anime_or_not import predict_image

probability = predict_image(r'C:\Users\emill\Desktop/dog.jpg')
print(probability)
```


Model has been trained on 250k anime images and 250k of other images.  
Architecture: Effnetv2-s

![alt text](https://github.com/LaihoE/Anime-or-not/blob/main/images/exampleanime.png?raw=true)