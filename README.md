# Audio Classification using Keras and Jupyter Notebooks

In this developer code pattern, we first create a Deep Learning model to classify audio embeddings. We train the model on IBM Deep Learning as a Service (DLaaS) platform and then perform inference/evaluation on IBM Watson Studio. 

The model will use audio embeddings generated by the VGG-ish model as an input and generate output probabilities/scores for 527 classes. For the purposes of illustrating the concept and exposing a developer to the features on IBM Cloud platforms, we use Google's Audioset data where the embeddings have been pre-processed and available readily.  However, a developer can leverage this model to create their own custom audio classifier trained on their own audio embeddings/data.

When the reader has completed this Code Pattern, they will understand how to:

* Setup an IBM Cloud object storage bucket and upload the training data to the cloud.
* Upload a Deep Learning model to IBM DLaaS for training.
* Integrate the object storage buckets into IBM Watson Studio.
* Perform inference on an evaluation dataset using Jupyter Notebooks over IBM Watson Studio.

## Flow

## Included Components
* [IBM Cloud Object Storage](https://www.ibm.com/cloud/): insert description here
* [IBM Deep Learning as a Service](https://www.ibm.com/cloud/): insert description here
* [Watson Studio](https://www.ibm.com/cloud/watson-studio): insert description here 
* [Jupyter Notebooks](http://jupyter.org/): An open source web application that allows you to create and share documents that contain live code, equations, visualizations, and explanatory text.
* [Keras](https://keras.io/): The Python Deep Learning library.
* [Tensorflow](https://www.tensorflow.org/): An open-source software library for Machine Intelligence.

## Featured Technologies

* [Cloud](https://www.ibm.com/developerworks/learn/cloud/): Accessing computer and information technology resources through the Internet.
* [Data Science](https://medium.com/ibm-data-science-experience/): Systems and scientific methods to analyze structured and unstructured data in order to extract knowledge and insights.
* [Artificial Intelligence](https://www.ibm.com/developerworks/learn/cognitive/index.html):Artificial intelligence can be applied to disparate solution spaces to deliver disruptive technologies.
* [Python](https://www.python.org/): Python is a programming language that lets you work more quickly and integrate your systems more effectively.

# Prerequisites

1. Setup cloud object storage and aws command line tools.
2. Create accounts on DLaaS and WML.

# Steps

The steps can be broadly classified into the following topics:

1. Clone the repository.
2. Upload training data to cloud object storage.
3. Setup and upload model on DLaaS to train.
4. Upload evaluation notebook to Watson Studio.
5. Run evaluation on Watson Studio.

## Clone the repository

* Clone this repository and change into the new directory:

```
git clone https://github.com/IBM/audioset-classification
cd audioset-classification
```




