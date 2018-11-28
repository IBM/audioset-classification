# Train and evaluate an audio embedding classifier

This developer code pattern will guide you through training a Deep Learning model to classify audio embeddings on IBM's Deep Learning as a Service (DLaaS) platform  - Watson Machine Learning - and performing inference/evaluation on IBM Watson Studio. 

The model will use audio [_embeddings_](https://www.tensorflow.org/programmers_guide/embedding) as an input and generate output probabilities/scores for 527 classes. The classes cover a broad range of sounds like speech, genres of music, natural sounds like rain/lightning, automobiles etc. The full list of sound classes can be found at [Audioset Ontology](https://research.google.com/audioset/ontology/index.html). 
The model is based on the paper ["Multi-level Attention Model for Weakly Supervised Audio Classification"](https://arxiv.org/abs/1803.02353). As outlined in the paper, the model accepts [_embeddings_](https://www.tensorflow.org/programmers_guide/embedding) of 10-second audio clips as opposed to the raw audio itself. The embedding vectors for raw audio can be generated using  VGG-ish [model](https://github.com/tensorflow/models/tree/master/research/audioset). The VGG-ish model converts each second of raw audio into an embedding(vector) of length 128 thus resulting in a tensor of shape 10x128 as the input for the classifier. For the purposes of illustrating the concept and exposing a developer to the features of the IBM Cloud platform, Google's Audioset data is used, where the embeddings have been pre-processed and available readily. Though Audioset data is used here, a developer can leverage this model to create their own custom audio classifier trained on their own audio data. They would however have to first generate the audio embeddings as mentioned above. 

When the reader has completed this Code Pattern, they will understand how to:

* Setup an IBM Cloud Object Storage bucket and upload the training data to the cloud.
* Upload a Deep Learning model to Watson ML for training.
* Integrate the object storage buckets into IBM Watson Studio.
* Perform inference on an evaluation dataset using Jupyter Notebooks over IBM Watson Studio.

![](doc/source/images/flow.png)

## Flow

1. Upload training files to Cloud Object Storage.
2. Train on Watson Machine Learning.
3. Transfer trained model weights to new bucket on IBM Cloud Object Storage and link it to IBM Watson Studio.
4. Upload and run the attached Jupyter notebook on Watson Studio to perform inference. 

## Included Components

* [IBM Cloud Object Storage](https://www.ibm.com/cloud/):  A highly scalable cloud storage service, designed for high durability, resiliency and security.
* [IBM Cloud Watson Machine Learning](https://www.ibm.com/cloud/machine-learning/pricing): Create, train, and deploy self-learning models. 
* [Watson Studio](https://www.ibm.com/cloud/watson-studio): Build, train, deploy and manage AI models, and prepare and analyze data, in a single, integrated environment.
* [Jupyter Notebooks](http://jupyter.org/): An open source web application that allows you to create and share documents that contain live code, equations, visualizations, and explanatory text.
* [Keras](https://keras.io/): The Python Deep Learning library.
* [Tensorflow](https://www.tensorflow.org/): An open-source software library for Machine Intelligence.

## Featured Technologies

* [Cloud](https://www.ibm.com/developerworks/learn/cloud/): Accessing computer and information technology resources through the Internet.
* [Data Science](https://medium.com/ibm-data-science-experience/): Systems and scientific methods to analyze structured and unstructured data in order to extract knowledge and insights.
* [Artificial Intelligence](https://www.ibm.com/developerworks/learn/cognitive/index.html): Artificial intelligence can be applied to disparate solution spaces to deliver disruptive technologies.
* [Python](https://www.python.org/): Python is a programming language that lets you work more quickly and integrate your systems more effectively.

# Prerequisites

1. Provision a Cloud Object Storage service instance on IBM Cloud.
2. Provision a Watson Machine Learning service instance on IBM Cloud.
3. Set up IBM Cloud and AWS command line tools.

### Provision a Cloud Object Storage (COS) service instance

* Log in to the [IBM Cloud](https://console.bluemix.net/). Sign up for a free account if you don't have one yet.
* [Provision a Cloud Object Storage service instance](https://console.bluemix.net/catalog/services/cloud-object-storage) on IBM Cloud if you don't have one 
* Create credentials for either reading and writing or just reading
	* From the IBM Cloud console page (https://console.bluemix.net/dashboard/apps/), choose `Cloud Object Storage`
	* On the left side, click the `service credentials`
	* Click on the `new credentials` button to create new credentials
	* In the `Add New Credentials` popup, use this parameter `{"HMAC":true}` in the `Add Inline Configuration...`
	* When you create the credentials, copy the `access_key_id` and `secret_access_key` values.
	* Make a note of the endpoint url
		* On the left side of the window, click on `Endpoint`
		* Copy the relevant public or private endpoint. [I choose the us-geo private endpoint].
* In addition setup your [AWS S3 command line](https://aws.amazon.com/cli/) which can be used to create buckets and/or add files to COS.
   * Export `AWS_ACCESS_KEY_ID` with your COS `access_key_id` and `AWS_SECRET_ACCESS_KEY` with your COS `secret_access_key`

### Provision a Watson Machine Learning service instance

* Log in to the [IBM Cloud](https://console.bluemix.net/). Sign up for a free account if you don't have one yet.
* [Provision a Watson Machine Learning service instance](https://console.bluemix.net/catalog/services/machine-learning) on IBM Cloud if you don't have one 
* Create new credentials 
	* On the left side, click the `service credentials`
	* Click on the `new credentials` button to create new credentials
	* In the `Add New Credentials` popup accept the defaults and create the credentials.
	* View the generated credentials and take note of the values for `instance`, `username`, `password`, and `url`. You will need this information when you configure your IBM Cloud CLI.	


### Set up IBM CLI & ML CLI

* Install [IBM Cloud CLI](https://console.bluemix.net/docs/cli/reference/bluemix_cli/get_started.html#getting-started)
  * Log in using `ibmcloud login` or `ibmcloud login --sso` if within IBM
* Install [ML CLI Plugin](https://dataplatform.ibm.com/docs/content/analyze-data/ml_dlaas_environment.html)
  * Verify that the latest plug-in version is installed
    * `ibmcloud plugin update machine-learning`
  * Define the following environment variables, using the values you've collected earlier: `ML_INSTANCE`,`ML_USERNAME`, `ML_PASSWORD`, `ML_ENV`

# Steps

The steps can be broadly classified into the following topics:

1. Clone the repository.
2. Upload training data to Cloud Object Storage.
3. Setup and upload model on Watson ML (DLaaS) to train.
4. Upload evaluation notebook to Watson Studio.
5. Run evaluation on Watson Studio.

> Note: If you want to perform inference with pre-trained weights, skip to step 4 directly. 

## 1. Clone the repository

Clone this repository and change into the new directory:

```
$ git clone https://github.com/IBM/audioset-classification
$ cd audioset-classification
```

A few things to mention about the contents of the repository:

* [audio_classify.zip](audioset_classify.zip): This is the core training code we will be running on IBM Cloud. 
* [audioset_classify](audioset_classify/): The training code for reference. We will use the first .zip file to upload this code to the cloud.
* [training-runs.yml](training-runs.yml) This file is required to perform the training on IBM Cloud. It is used to setup training metadata and connection information.
* [audioclassify_inference.ipynb](audioclassify_inference.ipynb) This is the notebook we will be using to perform inference after training. 

## 2. Upload training data to cloud storage

* Download the data from [here](https://github.com/qiuqiangkong/audioset_classification) under the section `download dataset`.

* Create buckets on your Cloud Object Storage instance. This can be done either through the UI or through the command line as shown below. 

We will create one bucket to put the training data and one bucket where the code will save the results/models at the end of training. 

```
$ aws s3 mb s3://training-audioset-classify
$ aws s3 mb s3://results-audioset-classify
```

Developers within IBM will need to add an endpoint URL to all `aws s3` commands. The above commands will thus look like this: 

```
$ aws --endpoint-url=http://s3-api.us-geo.objectstorage.softlayer.net s3 mb s3://training-audioset-classify
$ aws --endpoint-url=http://s3-api.us-geo.objectstorage.softlayer.net s3mb s3://results-audioset-classify
```

Now we can move the files to the cloud storage:

```
$ aws s3 cp bal_train.h5 s3://training-audioset-classify/
$ aws s3 cp unbal_train.h5 s3://training-audioset-classify/
$ aws s3 cp eval.h5 s3://training-audioset-classify/
```

## 3. Setup and upload model to Watson ML 

Now that we have our training data setup, we upload our model and submit a training job. 

* Replace the `api_key_id` and `secret_api_key_id` fields in the [training-runs.yml](training-runs.yml) file with your credentials as mentioned in the pre-requisites step.

* Run the below code on the terminal to start training:

```
$ ibmcloud ml train audioset_classify.zip training-runs.yml
```

After the train is started, it should print the training-id that is going to be necessary for steps below

```
Starting to train ...
OK
Model-ID is 'training-GCtN_YRig'
```

### Monitor the  training run

* To list the training runs - `ibmcloud ml list training-runs`
* To monitor a specific training run - `ibmcloud ml show training-runs <training-id>`
* To monitor the output (stdout) from the training run - `ibmcloud ml monitor training-runs <training-id>`
	* This will print the first couple of lines, and may time out.

Once the training is complete, you can access the model and weights from the cloud object storage. The weights can be downloaded from the UI. 

The file we will be using for the next steps will be called `final_weights.h5`. It can be found on the object storage bucket `results-audioset-classify` under `<your_training_id>/models/main/balance_type=balance_in_batch/model_type=decision_level_multi_attention/final_weights.h5`. 

![](doc/source/images/1.png)

> The above screenshot shows the final weights/model checkpoints saved during training. 

The file can be downloaded via UI or via command line using the below command:

```
$ aws s3 cp s3://results-audioset-classify/<your_training_id>/models/main/balance_type=balance_in_batch/model_type=decision_level_multi_attention/final_weights.h5 final_weights.h5
```

## 4. Upload evaluation notebook on Watson Studio

1. [Log in to Watson Studio](https://www.ibm.com/cloud/watson-studio). Create a free account if you don't have one yet.
2. Create a new project `Audioset Classification` in Watson Studio.
3. Navigate to `Assets -> Notebooks` and click on `New notebook`.
4. On the next screen click on `From file` and upload the [audioclassify_inference.ipynb](audioclassify_inference.ipynb) file. 
5. Upload `final_weights.h5` (file which we downloaded in the previous step) and `eval.h5` to the object storage linked to Watson Studio. This can be done by navigating to to `assets->New data asset` or clicking on the icon on the right to popup the data upload GUI as shown in the screenshot below. 

![](doc/source/images/2.png)

5. Similarly, upload [audioset_classify/metadata/eval_segments.csv](audioset_classify/metadata/eval_segments.csv) and [audioset_classify/metadata/class_label_indices.csv](audioset_classify/metadata/class_label_indices.csv) files. These files contain metadata such as YouTube URLs, class labels and start/end times. 

## Run inference

Now that all the data has been setup, you can open the uploaded Jupyter notebook and follow inline comments / directions. 
The first section loads the data into memory where applicable. Each cell mentions any action to be performed prior to executing the cell if applicable. 
Example: to load the credentials for the data, navigate to the `Files` section on the right and click on the required file and click on `Insert to code->Insert credentials`. This will insert a code snippet with required credentials/API keys.  

![](doc/source/images/3.png)

> Screenshot showing how to insert file credentials into the notebook.

* Run all cells as-is unless stated otherwise on cell comments. 

* We now demonstrate two cool applications at the end of this tutorial. 

### Real Time Demo

The Real Time Demo section takes in a random audio (given as a number which is referenced from the eval.h5 list) and performs real time inference on that embedding and outputs class labels and a YouTube embedding plays the corresponding video/audio snippet. You can try out different videos and see that the performance matches human level annotation. 
For example setting `video_number = 350` the top 5 class predictions are as shown in the screenshot below and it matches the audio perfectly. 

![](doc/source/images/4.png)

### Reverse search Audio using keywords

Now we perform inference on the entire eval set and generate top 5 class predictions for each evaluation example. We then use these to retrieve suggestions when queried for a particular keyword. A example of this is shown below where the keyword is 'Car' and we see that results are pretty accurate. 
Feel free to replace the `search_query = 'Car'` with your own keyword. For a list of all support keywords, refer to [class_labels_indices.csv](audioset_classify/metadata/class_labels_indices.csv).

![](doc/source/images/5.png)

## References

* [1] Gemmeke, Jort F., et al. "Audio set: An ontology and human-labeled dataset for audio events." Acoustics, Speech and Signal Processing (ICASSP), 2017 IEEE International Conference on. IEEE, 2017.
* [2] Kong, Qiuqiang, et al. "Audio Set classification with attention model: A probabilistic perspective." arXiv preprint arXiv:1711.00927 (2017).
* [3] Yu, Changsong, et al. "Multi-level Attention Model for Weakly Supervised Audio Classification." arXiv preprint arXiv:1803.02353 (2018).

## External links
The core model for audio classification is based on the [implementation](https://github.com/qiuqiangkong/audioset_classification) and paper by  Qiuqiang Kong. 

The original [implementation](https://github.com/ChangsongYu/Eusipco2018_Google_AudioSet) of [3] is created by Changsong Yu. 

# Learn more

* **Artificial Intelligence Code Patterns**: Enjoyed this Code Pattern? Check out our other [AI Code Patterns](https://developer.ibm.com/code/technologies/artificial-intelligence/).
* **AI and Data Code Pattern Playlist**: Bookmark our [playlist](https://www.youtube.com/playlist?list=PLzUbsvIyrNfknNewObx5N7uGZ5FKH0Fde) with all of our Code Pattern videos
* **With Watson**: Want to take your Watson app to the next level? Looking to utilize Watson Brand assets? [Join the With Watson program](https://www.ibm.com/watson/with-watson/) to leverage exclusive brand, marketing, and tech resources to amplify and accelerate your Watson embedded commercial solution.

# License

This code pattern is licensed under the Apache Software License, Version 2.  Separate third party code objects invoked within this code pattern are licensed by their respective providers pursuant to their own separate licenses. Contributions are subject to the Developer [Certificate of Origin, Version 1.1 (DCO)] (https://developercertificate.org/) and the [Apache Software License, Version 2] (http://www.apache.org/licenses/LICENSE-2.0.txt).

ASL FAQ link: http://www.apache.org/foundation/license-faq.html#WhatDoesItMEAN