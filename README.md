# Musical key detection with Deep Learning

## Papers
https://paperswithcode.com/paper/deeper-convolutional-neural-networks-and  
https://paperswithcode.com/paper/musical-tempo-and-key-estimation-using  
https://arxiv.org/pdf/1706.02921.pdf  

## Topic
Musical feature extraction

## Project type
Bring your own model

## Summary
My main focus for this project will be the topic of key detection in musical pieces. I hope to reuse existing approaches and further refine them to improve on their performance. Steps are planned in the following order:
1. Rebuild an existing solution as quoted above
2. Experiment with network architectures, types of feature extraction, and applications to the waveform itself.
3. Expand the currently existing collection of key detection data sources with simple self-made compositions. Using modern DAWs it should be somewhat trivial to construct a series of short audio samples in various keys using different instruments setups.

I plan to use the following datasets:
* [GiantSteps](https://github.com/GiantSteps/giantsteps-key-dataset) & [GiantSteps MTG](https://github.com/GiantSteps/giantsteps-mtg-key-dataset)  
These seems to be common datasets to use for key extraction and also provide us with some comparable approaches from other models
* [Children's Songs](https://dagshub.com/kinkusuma/children-song-dataset)  
This is a set of vocal recordings only
* Optionally, my own dataset

My main focus will be on model generation. However, if I can reach satisfying results before my estimated time is used up, I will invest the remaining time into dataset creation. Hence the following breakdown is somewhat flexible:
* Dataset collection: 2-8 hrs
* Design/build network: 9-15 hrs
* Train/tune network: 18 hrs
* Build application: 8 hrs
* Write report: 6 hrs
* Presentation: 6 hrs
Total: 55 hrs


# Phase 2 - Hacking report

## Plan
My references paper used accuracy ratings (micro-averaged from my understanding) as well as the Mirex score. I am using both scores to be comparable.

The state of the art in Mirex score is around 75. My aim was to reach at least 70.

For the implementation I went for stripped-down version of InceptionKeyNet. I implemented some of the blocks, but stopped noticing performance increases after a while, and in fact it seemed that the network decreased in accuracy. Personally I think the full network is overkill for key detection only, which likely depends on a few base frequencies for the most part. My next steps will include experimentation with more simple models.

## Installation
All code and notes can be found in the Jupyter notebook. Please install the dependencies outlined in requirements.txt. The audio files must be downloaded using the repository links above. The project is configured for Giantsteps and Giantsteps MTG and allows setup of data locations within the notebook. It also includes a conversion script from mp3 to wav. I neeeded to load the files in wav format on my windows machine, or else a significant chunk would not load

Some of the files could not be opened by librosa. The notebook includes a workaround, but it may be hard to follow. The affected Giantsteps files are

1149778.LOFI.mp3  
1164898.LOFI.mp3  
1193612.LOFI.mp3  
1198688.LOFI.mp3  
1206025.LOFI.mp3  
1234668.LOFI.mp3  
1234670.LOFI.mp3  
1234750.LOFI.mp3  
1234751.LOFI.mp3  
1234752.LOFI.mp3  
1257593.LOFI.mp3  
1486770.LOFI.mp3  

## Results
The final Mirex best score for my network is currently around 60 across various recomputed train-test splits using the optimal configuration (should be set up at time of submissions), meaning I am sadly behind my established target at the moment.

Rough time investment:
* Dataset collection: 3 hrs
* Design/build network: 30 hrs
* Train/tune network: 20 hrs
