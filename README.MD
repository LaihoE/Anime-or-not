# Neural Network for classifying if images include anime
## Installing
```
pip install anime-or-not
```
## Useage
```python
from anime_or_not import predict_image

probability = predict_image('path_to_image')
print(probability)
```


Model has been trained on 250k anime images and 250k of other images.  
Architecture: Effnetv2-s  
Testing accuracy was around 98-99% (The line between anime and something else is not that clear)

![alt text](https://github.com/LaihoE/Anime-or-not/blob/main/images/exampleanime.png?raw=true)