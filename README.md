# Image_caption
Final project for DL Summer22
## Description
Image caption referes to the process of describing an image using a sentence. By asking computers to generate captions based on input image, we may find ways to improve life quality for those with visual impairment. In addition, with image caption we could also build and optimize the search engine for image content. <br><br>
In this project, we build and use a CNN-RNN model to automatically generate descriptions.<br>

> CNN-RNN model: The image is encoded into a context vector by a CNN which can then be passed to a RNN decoder [^1]
![This is an image](https://miro.medium.com/max/1400/0*Z0KrVxXpDqTacrsF.)
Here, we use the resnet18 model for our CNN encoder part. <br>
[^1]: [Image reference](https://blog.mlreview.com/multi-modal-methods-image-captioning-from-translation-to-attention-895b6444256e)
## Data
We use the [Flickr Image dataset](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset) from kaggle, with a total of 31.8k images and corresbonding human annotated captions.
## Model
## Summary
## Discussions
Some other implemations using captions could be:<br>
- Given a sentence, find out the matching images.

