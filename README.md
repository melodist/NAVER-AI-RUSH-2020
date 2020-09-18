# NAVER AI RUSH 2020
![title](./header_img/header.png)

- **ROUND1: Spam Image Classificiation** 
- 29 th / 50
- July 13 - July 31

## Solution
Average Model with 2 ResNet50_v2 using class weights

### Metrics
- Geometric Mean of F1 score for each class

### Tried
- Data Augmentation, Normalization
- EfficientNet B0-B3
- Using Multi GPUs: Mistake of hyperparameters 

### Cannot do because of timeout
- EfficientNet + ResNet50_v2
- Using TF 2.x: Need new environments including CUDA libraries