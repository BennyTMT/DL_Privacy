# erGAN, privacy leakage from Face Embedding 
This document supports the work ***The Many-faced God: Attacking Face Verification System with Embedding and Image Recovery*** (Accepted by **[ACSAC 2021](https://www.acsac.org/)** Annual Computer Security Applications). [PAPER LINK](https://cpb-us-e2.wpmucdn.com/faculty.sites.uci.edu/dist/5/764/files/2021/10/acsac21a.pdf)

**Author Invloved: Mingtian Tan,Zhe Zhou,Zhou Li**   


## Tutorials
```
pyhton3 main.py -b 32 -l 0.0015
```
This file will conduct the whole Attack directly. "-b" means batch size and "-l" means learning rate. There are several variables setting you should pay attention, such as "datset path" or "model save path", which are related to your own project file structure. Also, you can change the Hyper Parameters in the file independently or adjust the architecture of the model, which may result in better performance in face recovery task. 

```
erGan.py
```
This file is about the whole architecture of our erGAN model. Specifically, this file is about recover [Embedding-1024 Face Model](https://www.clarifai.com/developers/pre-trained-models) in real world, also you can change the model interface to fit your own recover task. ***"_generator()"*** is about how we extract information from embedding and recover face from it, pipline showing below:

![](/src/designPipline.png  "pipline")	

This is the performance ranmdomly choosed from **Testing Data**. The first line is the oringial face images from public dataset "LFW". Second line is the face images recovered from embedding 1024 from an online face classification application **Clarifai-1024**. 
![](/src/faceRecovery.png  "face")	



One can conduct the whole project by 


We will put our experiment files in this documents gradually. 

If there are something you are interested in, **Plz Contact Us**
