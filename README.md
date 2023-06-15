# Coding3-Final-Project

## Introduction
### "90 Seconds After Falling into a Pile of Hats” 🎩
This work utilizes the pix2pix algorithm to predict the 90 seconds following the end of the animated short film “Ice Merchants". Every 100 frames of the video were extracted as a frame for the dataset to train the model. The final frame of the image is used as the input image, with the output image serving as the input for the next frame. After iterations, I obtain the 90 seconds after the story ends.

🎬 Video URL: 

![image](https://github.com/YutianWeii/jpg/blob/main/%E6%88%AA%E5%B1%8F2023-06-15%2016.28.50.png)


## Process
### 1. Iteration of ideas
In Week8, I chose the Pix2Pix model of the GAN algorithm, intending to import satellite image datasets (or other datasets), and then implement a real-time transformation of gestures into satellite map images (or other images).
The final effect can be referred to the following two works:
https://twitter.com/quasimondo/status/982711735001010176
https://www.memo.tv/works/gloomy-sunday/

Code references:
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
https://www.tensorflow.org/tutorials/generative/pix2pix?hl=zh-cn

Dataset references:
Satellite Image Classification https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification
or
Micro-Organism Image Classification https://www.kaggle.com/datasets/mdwaquarazam/microorganism-image-classification


### 2. Technical reference




### 3. Production process

![image]()
![image]()
![image]()


## Presentation


![image]()
![image]()
![image]()


## Conclusion
The final result is not very satisfying. Although the colors and vague shapes align with the ambiance of the original video, there are no clear lines or objects generated. I think this might be due to the complexity of the original video; the model could not learn and predict an unseen story very well. The two works I referenced had static camera positions and simple scenes (a train and a roller coaster), which made it easier for the machine to learn and generate more realistic images.

I started to think about the relationship between concept and final effect. I could have chosen a video with a simpler scene, but that would run counter to my concept. I wanted to input this short film and only this short film. Once the training was completed, the AI's feedback seemed real and somewhat objective to me, free from bias. Given the current state of pix2pix technology, this is what the 90 seconds after the end of this short film looks like, which makes me think that perhaps the final effect isn't so important.

When we create with AI, how should we balance art and computational rationality? In the generated video, how much participation does the original director and production team have? It can be considered as 100% or 0%, but in reality, it is neither 100% nor 0%.

In conclusion, this was an interesting AI experiment. With technology, I can realize some bold ideas and contemplate their intrinsic relationships. I enjoyed the process, and I hope you did too.

## Reference
1.
2.
3.
4.
5.


