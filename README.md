# Musical key detection with Deep Learning

## Papers
https://paperswithcode.com/paper/deeper-convolutional-neural-networks-and
https://paperswithcode.com/paper/musical-tempo-and-key-estimation-using
https://arxiv.org/pdf/1706.02921.pdf

## Topic
Musical feature extraction

## Project type
Bring your own model / data

## Summary
My main focus for this project will be the topic of key detection in musical pieces. I hope to reuse existing approaches and further refine them to improve on their performance. Steps are planned in the following order:
1. Rebuild an existing solution as quoted above
2. Experiment with network architectures, types of feature extraction, and applications to the waveform itself.
3. Expand the currently existing collection of key detection data sources with simple self-made compositions. Using modern DAWs it should be somewhat trivial to construct a series of short audio samples in various keys using different instruments setups.

I plan to use the following datasets:
* [GiantSteps](https://github.com/GiantSteps/giantsteps-key-dataset) & [GiantSteps MTG](https://github.com/GiantSteps/giantsteps-mtg-key-dataset)
These seems to be common datasets to use for key extraction and also provides us with some comparable approaches from other models
* [Children's Songs](https://dagshub.com/kinkusuma/children-song-dataset)
This is a set of vocal recordings only
* Optionally, my own dataset

My main focus will be on model generation. However, if I can reach satisfying results before my estimated time is used up, I will invest the remaining time into dataset creation. Hence the following breakdown is somewhat flexible:
* Dataset collection: 3-8 hrs
* Design/build network: 9-14 hrs
* Train/tune network: 18 hrs
* Build application: 8 hrs
* Write report: 6 hrs
* Presentation: 6 hrs
Total: 55 hrs
