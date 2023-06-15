# Coding3-Final-Project

## Introduction
### "90 Seconds After Falling into a Pile of Hats” 🎩
This work utilizes the pix2pix algorithm to predict the 90 seconds following the end of the animated short film “Ice Merchants". Every 100 frames of the video were extracted as a frame for the dataset to train the model. The final frame of the image is used as the input image, with the output image serving as the input for the next frame. After iterations, I obtain the 90 seconds after the story ends.

### 🎬 Video URL: 

![image](https://github.com/YutianWeii/jpg/blob/main/%E6%88%AA%E5%B1%8F2023-06-15%2016.28.50.png)


## Process
### 1. Iteration of ideas
### Week 8
In Week 8, I chose the Pix2Pix model of the GAN algorithm, intending to import satellite image datasets (or other datasets), and then implement a real-time transformation of gestures into satellite map images (or other images).
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

### Week 9
In week 9, I changed part of the project idea. I still want to use the pix2pix algorithm, but I plan to change the generated content. Since incomplete datasets will bias AI algorithms, I plan to use a dataset that only contains male faces, and the output part will all be male faces, reflecting situations where women are overlooked. The input images will be images similar to the shape of a human face, generating corresponding male face images.

Code references:

Pix2Pix tool: https://www.tensorflow.org/tutorials/generative/pix2pix?hl=zh-cn

StyleGAN tool recommended by Jasper (for converting target images to the style of pix2pix model input images): https://github.com/jasper-zheng/StyleGAN-Canvas

At that time, I didn't find a suitable male face dataset. I plan to look for it again. If not, I plan to use gender detection code to select the male part as my dataset (https://www.kaggle.com/code/bmarcos/image-recognition-gender-detection-inceptionv3). Also, the correspondence relationship of the pix2pix algorithm dataset needs to be tested. If I can't achieve real-time interactive effects, I consider making the input and output images into GIF images.


### Week 10
In the tenth week, I discovered these two works that used the pix2pix model to predict video, which was very interesting.

Jonathan Fly: Start with a video clip of a roller coaster and then predict the next frame with a pix2pix model, feeding the previous frame
https://twitter.com/jonathanfly/status/1185843103271444480

![image](https://github.com/YutianWeii/jpg/blob/main/%E6%88%AA%E5%B1%8F2023-06-15%2017.55.24.png)

Damien Henry: Pix2pix repeatedly takes the generated output as the new input to create a one-hour long video of train windows
https://www.aiartonline.com/community/damien-henry/

![image](https://github.com/YutianWeii/jpg/blob/main/%E6%88%AA%E5%B1%8F2023-06-15%2017.55.12.png)

So, I planned to use AI to predict one minute after a movie ends. I plan to take each frame of a movie as a dataset and use machine learning to understand the patterns of change between frames. Finally, the last frame is used as the input image to generate the next frame after the movie ends. Then the next frame is used as the input image to generate the next frame, and after repeating this many times, all the generated images are combined into a video.
At this point, I had determined the concept for my final project.



### 2. Technical reference
I looked up information about the pix2pix model and tried running the sample code in TensorFlow. The original dataset of the pix2pix sample code is images of size 512 * 256, containing one 256 * 256 input image and one 256 * 256 target image. When preparing my dataset, I also need to adjust the pictures to this format.

Pix2pix model:

https://phillipi.github.io/pix2pix/

https://www.tensorflow.org/tutorials/generative/pix2pix?hl=zh-cn

I also discovered the vid2vid model, which I can try in the future.

Vid2vid model:
https://github.com/NVIDIA/vid2vid



### 3. Production process
Firstly, I used a short jellyfish video that I shot before to test this idea. It is necessary to split the video into each frame image and then export it to a 'frames' folder. As the size of the input images does not match the model settings, I combined two adjacent pictures into one 512 * 256 size picture and imported it into the dataset folder. I planned to change the picture to a 1920 * 1080 size after the final training and the generation of new pictures, and then combine them into a video. I generated this step with the help of ChatGPT (the code can be viewed in the "video—frames—dataset.ipynb" file).

Then I opened the pix2pix sample code in Colab and changed the code to import the dataset. Since Colab requires uploading a local dataset to import, I uploaded the dataset folder to Google Drive and then imported the files from Google Drive. (Code was generated with the help of ChatGPT and showed in the following image. The code can be viewed in the "pix2pix-Yutian.jpynb" file, and it is recommended to run it in Colab).

![image](https://github.com/YutianWeii/jpg/blob/main/%E6%88%AA%E5%B1%8F2023-06-15%2018.19.50.png)

While running the code, I realized that I had forgotten to import the dataset into different training, validation, and test sets. So I returned to the "video—frames—dataset.ipynb" file, and added code to import the images from the dataset into three new folders, with 70% of the images used as the training set, 15% as the validation set, and 15% as the test set. (The code was generated with the help of ChatGPT and can be viewed in the "video—frames—dataset.ipynb" file)

I ran the pix2pix code again, and the model training was successful, and I was able to generate images.

![image](https://github.com/YutianWeii/jpg/blob/main/%E6%88%AA%E5%B1%8F2023-06-15%2020.08.04.png)

Then I started to import the last frame of the image using the code and generate new images based on the model. (The code was generated with the help of ChatGPT) But the images generated in this step were all black. I used plt.imshow(input_image) plt.show() and found out that the problem was not with the input image. I guessed it might be a problem with the model. But even after retraining the model, I could still only generate completely black pictures. 

![image](https://github.com/YutianWeii/jpg/blob/main/%E6%88%AA%E5%B1%8F2023-06-15%2020.08.18.png)

I consulted with Jasper, and after he made several guesses, he found out that there are a lot of Batchormalization layers in my model. I needed to add "training = True" in the code (the correct code should be output_image = generator(input_image, training = True)) to successfully generate pictures.

About the Batchormalization: https://stackoverflow.com/questions/50047653/set-training-false-of-tf-layers-batch-normalization-when-training-will-get-a/50071072#50071072

![image](https://github.com/YutianWeii/jpg/blob/main/%E6%88%AA%E5%B1%8F2023-06-15%2020.17.49.png)

Then, with the help of ChatGPT, I completed the code for generating images in a loop and composing a video. (The code can be viewed in the last part of the "pix2pix-Yutian.jpynb" file, and it is recommended to run it in Colab)

Once it was confirmed that the video could be generated, I began to formally produce using "Ice Merchant".

First, the video needed to be split into every frame, resulting in 20,960 images.

![image](https://github.com/YutianWeii/jpg/blob/main/%E6%88%AA%E5%B1%8F2023-06-15%2021.24.59.png)

Then, the previous frame and the next frame were combined into one image as a dataset, resulting in 20,959 images, and at this time the size of the images was changed to 512*256.

![image](https://github.com/YutianWeii/jpg/blob/main/combined_18560%E7%9A%84%E5%89%AF%E6%9C%AC.jpg)

I imported the images from the dataset into three new folders, with 70% of the images used as the training set, 15% as the validation set, and 15% as the test set. I then uploaded the three folders to Google Drive.

I started to run the pix2pix code to train the model, and I was able to generate some images.

![image](https://github.com/YutianWeii/jpg/blob/main/%E6%88%AA%E5%B1%8F2023-06-15%2021.26.55.png)

However, because the difference between each frame and the next frame is very small, the final 1800 generated images are almost identical to the input images, and there isn't much change in the video. So I planned to change the dataset, extract one frame every 100 frames from the original video, roughly extracting 210 frames to train the model. But the final generated video still didn't change much.

![image](https://github.com/YutianWeii/jpg/blob/main/%E6%88%AA%E5%B1%8F2023-06-15%2021.27.10.png)

![image](https://github.com/YutianWeii/jpg/blob/main/%E6%88%AA%E5%B1%8F2023-06-15%2021.32.55.png)

I changed the input image to one frame from the beginning of the video, but the result was still that each frame was nearly the same.

I suspected that the training times were not enough, so I changed the training step to 80,000, but the result still did not change.

I tried to use other videos for training, and downloaded a moon video from the NASA official website, but the result was still similar. At this point, I began to wonder if the result generated by AI was too different from my expectation, and whether I should change my idea.

![image](https://github.com/YutianWeii/jpg/blob/main/%E6%88%AA%E5%B1%8F2023-06-15%2021.31.43.png)

I tried to change the code, changing the input from the last frame to all frames. This way, individual frames of the video served as the dataset, and the model generated a new video based on each frame. I named it "Seeing the Small and Knowing the Big".

However, I was not very satisfied with this idea, so I asked Jasper and Rebecca why the images generated by inputting the last frame were all similar. Jasper believed that the train work I referenced had rapid changes in each frame, but the rhythm of my original video was relatively slow and there wasn't much change between two adjacent frames. He suggested that I modify the extraction frequency. However, even after I changed it to extracting one frame every 200 frames for training, there was still no noticeable change.

Finally, I found that the reason why the generated images were all similar was that I did not use the output image from the last round as the input image for the next round in the loop. I made a very silly mistake, which caused me to spend more than a day repeatedly training the model. Fortunately, the final result met my expectations.

![image](https://github.com/YutianWeii/jpg/blob/main/%E6%88%AA%E5%B1%8F2023-06-15%2021.38.25.png)

The final video was actually generated based on the training results of every 100 frames, so I should slow down the video speed. The final video is approximately 600 frames at a speed of 90 seconds.



## Presentation
🎬 Video 1 URL (includes the complete animation short and AI-predicted video, 15 min long):

🎬 Video 2 URL (comparison version, 1:36 long):

Several screenshots from the AI-predicted video:

![image](https://github.com/YutianWeii/jpg/blob/main/%E6%88%AA%E5%B1%8F2023-06-15%2021.41.53.png)

![image](https://github.com/YutianWeii/jpg/blob/main/%E6%88%AA%E5%B1%8F2023-06-15%2021.42.00.png)

![image](https://github.com/YutianWeii/jpg/blob/main/%E6%88%AA%E5%B1%8F2023-06-15%2021.42.13.png)

![image](https://github.com/YutianWeii/jpg/blob/main/%E6%88%AA%E5%B1%8F2023-06-15%2021.42.21.png)


## Conclusion
The final result is not very satisfying. Although the colors and vague shapes align with the ambiance of the original video, there are no clear lines or objects generated. I think this might be due to the complexity of the original video; the model could not learn and predict an unseen story very well. The two works I referenced had static camera positions and simple scenes (a train and a roller coaster), which made it easier for the machine to learn and generate more realistic images.

I started to think about the relationship between concept and final effect. I could have chosen a video with a simpler scene, but that would run counter to my concept. I wanted to input this short film and only this short film. Once the training was completed, the AI's feedback seemed real and somewhat objective to me, free from bias. Given the current state of pix2pix technology, this is what the 90 seconds after the end of this short film looks like, which makes me think that perhaps the final effect isn't so important.

When we create with AI, how should we balance art and computational rationality? In the generated video, how much participation does the original director and production team have? It can be considered as 100% or 0%, but in reality, it is neither 100% nor 0%.

In conclusion, this was an interesting AI experiment. With technology, I can realize some bold ideas and contemplate their intrinsic relationships. I enjoyed the process, and I hope you did too.


## Reference
1. Fingerplay https://twitter.com/quasimondo/status/982711735001010176
  
2. Learning to see: Gloomy Sunday https://www.memo.tv/works/gloomy-sunday/

3. Pix2pix https://www.tensorflow.org/tutorials/generative/pix2pix?hl=zh-cn

4. Jonathan Fly's work https://twitter.com/jonathanfly/status/1185843103271444480

5. Damien Henry's work https://www.aiartonline.com/community/damien-henry/

6. Pix2pix https://phillipi.github.io/pix2pix/

7. Vid2vid https://github.com/NVIDIA/vid2vid

8. About the Batchormalization: https://stackoverflow.com/questions/50047653/set-training-false-of-tf-layers-batch-normalization-when-training-will-get-a/50071072#50071072

9. Ice Merchants https://www.bilibili.com/video/BV1j44y1o7eY/?spm_id_from=333.337.search-card.all.click&vd_source=4e11ecf8ce84e2c57ec01266377e6b4f



