## Course_work_NN

The course work contains several variants of convolutional Neural Neural network to distinguish emotions on an image is got from the camera on PC.
link to the competition on Kaggle in here https://www.kaggle.com/competitions/skillbox-computer-vision-project/leaderboard

List of files:
SimpleCNN.ipynb - Contains two simple convolutional networks with no adds and relularization. Accuracy on the best of these networks  ~ 0.41 
Saved trained models for SimpleCNN.ipynb:

* model_main11.h5
* model_main12.h5
* model_main01_reg2.h5
* model_main02_reg.h5

CNNaugmentation.ipynb - Contains convolutional NNetworks with augmentation. Accuracy on the best of these networks ~ 0.43
Saved trained models for CNNaugmentation.ipynb:

* model_main20.h5
* model_main21.h5
* model_main23.h5
* model_main25.h5

FineTuning.ipynb - It contains two fine-tuned models, basing on VGGFace and Imagenet. The best score among them is ~ 0.42
Saved trained models for SimpleCNN.ipynb:

* model_main3_vggface.h5
* model_main3_imagenet_augm.h5

app.ipynb - It is the imitation of app for catching screen from the camera of PC, detection face of person and emotion on it.
As an inference model was used the imagenet model, finetuned on our dataset.



