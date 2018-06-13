# Blog Post

# What?

We approach the problem of audio event classification in this code pattern. Given a short clip (~10 seconds) of raw audio, can we build a classifier that predicts what the sound was? If so, how? What all classes should be included in such a classifier? 

# Why?

Is it a bird? Or is it a plane? 
Okay, that sounds really easy from an audio classification perspective..right? Wrong. Computers do not percieve audio the same way that we humans do. Just like they `see` one pixel at a time, they `hear` one bit at a time. So what seems trivial to us is actually not so straightforward after all. 
In this code pattern, we convert raw audio into a series of numbers which effectively captures the essence of the signal - called an embedding - and how we can use such embeddings to effectively classify different kinds of audio. 
Additionally, while there are a lot of open source resources addressing problems from Computer Vision, audio understanding while not a new topic, is yet to reach mainstream developers. We hope this code pattern helps you get started in this area.

# How?

As hinted in the above section, we use embeddings for our core classification task. Think of embeddings as a form of dimensionality reduction, taking in 1 second of audio (which may contain 1000s of bits) and converting it into a vector of length 128. The embeddings by themselves have no meaning, but prove very useful when used for downstream tasks like classification or clustering. We use the [VGG-ish embedding model](https://github.com/tensorflow/models/tree/master/research/audioset) and for the purposes of this demo, we use the Audioset data released by Google which already has these embeddings pre-procssed. 
We then feed these embeddings into a deep neural network (Deep Learning is ALWAYS the answer!) as outlined in this [research paper](https://www.researchgate.net/publication/323627323_Multi-level_Attention_Model_for_Weakly_Supervised_Audio_Classification) and related [code](https://github.com/qiuqiangkong/audioset_classification). After training for 50000 epochs on the full `bal+unbal` [dataset](https://research.google.com/audioset/download.html), we perform inference on the eval segments of the data and demonstrate human level accuracy. 
Go on, try for yourself! 
