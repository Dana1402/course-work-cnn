## Course_work_CNN

The course work contains several variants of convolutional Neural Neural network to distinguish emotions on an image is got from the camera on PC.
link to the competition on Kaggle in here https://www.kaggle.com/competitions/skillbox-computer-vision-project/leaderboard

List of files:
BaseLineModels.ipynb - Contains two simple convolutional networks. Accuracy on the best of these networks  ~ 0.44 

Augmentation.ipynb - Contains convolutional NNetworks with augmentation. Accuracy on the best of these networks ~ 0.47

FineTuning.ipynb - It contains several fine-tuned models, basing on VGGFace and Imagenet. The best score among them is ~ 0.43

launch_camera.py - It is the imitation of app for catching screen from the camera of PC, detection face of person and emotion on it.
As an inference model was used the imagenet model, finetuned on our dataset.

![Снимок1](https://user-images.githubusercontent.com/65036612/228519545-62d14f19-fadd-4ebb-98ac-e8a50c948bfd.JPG)
