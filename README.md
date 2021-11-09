# deepexplain

Welcome to the GitHub repository of our tutorial on explainability in machine learning!

#####
Write an introduction on why explainability is important for deep learning here
#####

This readme is intended to help navigate the tutorial that is organized in a modular manner.



+ Next, we also provide a (brief) general introduction on deep learning models and related utilities. This is just to make sure to get to the same page with novices. If you already worked with Pytorch then you may want to skip this part. [intro_to_basics.ipynb](./01.1_intro_to_basics.ipynb)


+ In this [notebook](./01.2_model_training.ipynb) and [script](./data_and_models.py) we finally get down to business: the data loading utilities and basic models are implemented here at one place so that they don't have to be repeated over and over before the different explanatory methods.

+ Then, we will look at a simple explainability method : training a simple model where the weights can be interpreted by a human ! We will also look to improve this method by using integrated gradients [here](02_model_weights_and_integrated_gradients.ipynb)

+ The method after that aims at hiding pieces of the input and looking at the changes that it implies in the output. Combined with deconvolutions (the reverse operation of the convolution), it is a cool method for looking at a popular image analysis model : the CNN. The notebook for these methods can be found [here](./03_occlusion_deconv.ipynb)

+ Then, we will dig deeper into more sophisticated methods for explaining deep learning models, [shapley values](./04_shapley_values.ipynb) and [CAM](05_scoreCAM.ipynb).
