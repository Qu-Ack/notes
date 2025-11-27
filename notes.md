# Readings

### game wining probabilitites and predictions.

Ganguly, S., Frank, N.: The problem with win probability. In: Proceedings of MIT
Sloan Sports Analytics Conference (2018) 

Fern´andez, J., Bornn, L.: Soccermap: A deep learning architecture for visually-
interpretable analysis in soccer. arXiv preprint arXiv:2010.10202 (2020)

Sicilia, A., Pelechrinis, K., Goldsberry, K.: Deephoops: Evaluating micro-actions in
basketball using deep feature representations of spatio-temporal data. In: Proceed-
ings of ACM International Conference on Knowledge Discovery & Data Mining,
KDD 2019, Anchorage, AK, USA, 4–8 Aug 2019, pp. 2096–2104 (2019

### 3d game simulation

Rematas, K., Kemelmacher-Shlizerman, I., Curless, B., Seitz, S.: Soccer on your
tabletop. In: Proceedings of the IEEE Conference on Computer Vision and Pattern
Recognition, pp. 4738–4747 (2018)

Shah, R., Romijnders, R.: Applying deep learning to basketball trajectories. arXiv
preprint arXiv:1608.03793 (2016)

 Hsieh, H.Y., Chen, C.Y., Wang, Y.S., Chuang, J.H.: Basketballgan: Generating
basketball play simulation through sketching. arXiv preprint arXiv:1909.07088
(2019)

### game action categorization

Baccouche, M., Mamalet, F., Wolf, C., Garcia, C., Baskurt, A.: Action classifica-
tion in soccer videos with long short-term memory recurrent neural networks. In:
Proceedings of International Conference on Artificial Neural Networks, pp. 154–
159. Springer, Heidelberg (2010)

Agyeman, R., Muhammad, R., Choi, G.S.: Soccer video summarization using deep
learning. In: Proceedings of IEEE Conference on Multimedia Information Process-
ing and Retrieval, pp. 270–273. IEEE (2019)


Ullah, A., Ahmad, J., Muhammad, K., Sajjad, M., Baik, S.W.: Action recogni-
tion in video sequences using deep bi-directional LSTM with CNN features. IEEE
Access 6, 1155–1166 (2017)

# SoccerMap: A deep learning architecture for visually-interpretable analysis in soccer

## the problem formulation and research objec­tive

They provide a neural network architecture that is capable of estimating probability surfaces for potential passes in a football game fig. shows the architecture the research uses 


this allows coaches to develop a fine-grained analysis of players’ positioning and decision-making


## the methodological framework adopted by the authors,

The model uses convolution layers to pick up meaningful patterns from the soccer field map at different zoom levels while keeping the image size unchanged and avoiding border issues. It downsamples the map with pooling to understand broader, more general patterns, then upsamples it smoothly to regain detail without creating visual artifacts. A fully convolutional design allows the model to make a prediction at every location on the field rather than just one final output. Finally, small 1×1 filters are used to turn these learned features into passing-probability predictions at each spot, and information from different zoom levels is combined so the model benefits from both fine and coarse details.

## the datasets utilized, including details on their acquisition and construction

high frequency, spatio temporal data is used. tracking data extracted from videos of soccer match consisting of 2d locations of players and the ball at 10 frames per second, allong with manually tagged passes, they arranged the data in a matrix of c x l x h, where l and h are the high level field coordinates and c represents the low level parameters that are passed to this architecture. low level features are calculated with things like liklehood of nearby teammates

they used tracking data, and event data from 740 English Premier League matches
from the 2013/2014 and 2014/2015 season, provided by STATS LLC.

## the models or algorithms implemented

they used a combination of models combining them in a single architecture, they are mentioned in fig.1

## results and conclusions

They split the data into 60:20:20 split which means 60% training data, 20% validation data, 20% test data.

They benchmarked against two models. Logistic Net and Dense2 Net. Logistic Net consists of a single sigmoid activation unit while Dense2 is Neural network with two dense layers followed by ReLu activation unit and a sigmoid output unit.

The Metrics they used are Negative Log Loss and ECE. for every model 


Below is the table of the results of their research, fig 2 shows the visual representation of the results.

Model         Log-loss ECE     Inferencetime Number of parameters
Naive         0.5451   −        −            0
Logistic Net  0.384    0.0210  0.00199s      11
Dense2 Net    0.349    0.0640  0.00231s      231
SoccerMap     0.217    0.0225  0.00457s      401, 259


# DeepHoops: Evaluating Micro-Actions In Basketball 

## the problem formulation and research objec­tive,
They created a Neural Network Architecture called DeepHoops which analyses spatio temoporal data of NBA games and predicts expected points to be scored as progression progresses. They estimate the probabiliites of terminal actions (e.g take goal, foul, turnover), each of these terminal actions is associated with an expected point value.

## the methodological framework adopted by the authors,
they used a concept of possessions which is a sequence of n moments where each moment is a 24-dimenstional vector. the first 20 capture the location of 10 players (x, y). the next three represent the court location and height of the ball (x, y, z), the last elements represents the current value of shot clock. 

their dataset consisted of more than 134,000 such possessions of interest.

they define a temporal window which acts as input to the DeepHoops architecture. window defined at a moment tao captures T moments up to a time of interest.

each window is labelled with an outcome that represents the type of terminal action that occured at the end of the window.

set of terminal actions included
- field goal attempts
- shooting foul
- Non-shooting foul
- turnover
- null

the architecture is mentioned above in fig 3. they used two modules one was a stacked _LSTM [LINK] [KEYWORD] network which was responsible in learning representation of the data up to time tao, this allowed for important information about on-court actions. the additional module was used to model who is on the courl to assess the impact of different lineups

they use 32 LSTM cells for each 3 layers.

## the datasets utilized, including details on their acquisition and construction

they used optical tracking data of 750 NBA games, which represents NBA games as a three dimensional coordinate system. the data they used was highly annotated and allowed labelling.

## the models or algorithms implemented

They used a stacked LSTM network. mentioned in fig 3

## results and conclusions

The metric they used to calculate the accuracy was _Brier Score_ [KEYWORD] [LINK] over 5 _epochs_ [KEYWORD] with minimum _improvement rate_ [KEYWORD] of 0.01

        BS      BSref   BSS     Epoch Time (s)
K = 1   0.4569  0.6070  0.2472  2180
K = 2   0.3598  0.4920  0.2686  2929
K = 3   0.3094  0.4017  0.2299  3552
K = 4   0.2659  0.3371  0.2114  4200

below are the reliability curves of the DeepHoops Model

# Soccer On Your TableTop  

# Objective

    The objective of the researchers was to transform a monocular video of a soccer game into a 3d reconstruction, in which players would be rendered interactively with an Augmented Reality Device.

# the methodological framework adopted by the authors,

A key component for the researchers was to estimate the _depth map_ [KEYWORD] of a particular player. 

They used a depth estimation neural network. the input was a 256x256 RGB cropped image. The input is processed by a series of 8 _hourglass modules_ [KEYWORD] [LINK] and the output is a 64x64x50 volume. representing 49 quantized depths and 1 background class.

The model was trained with entropy loss with a batch size of 6 for 300 epochs. 

they estimate a virtual vertical plane passing through the
middle of the player and calculate its depth w.r.t. the cam-
era. Then, we find the distance in depth values between a
player’s point and the plane. The distance is quantized into
49 bins (1 bin at the plane, 24 bins in front, 24 bins behind)
at a spacing of 0.02 meters, roughly covering 0.5 meters in
front and in back of the plane (1 meter depth span).


then they follow a pipeline to reconstruct the full 3d game 

which includes camera Pose Estimation then Player Detection and Tracking and then mesh generation this way they are able to get a 3d reconstruction of the orignal 2d game.

# the datasets utilized, including details on their acquisition and construction

researchers aquire the dataset of depthmaps for players by intercepting GPU calls between the game engine and the GPU of the game FIFA using RenderDoc. they aquired the _NDC_ [KEYWORD] (Normalized Device Coordinates). This process gives them a point cloud and after removing everything but the player they collected 12000 image-depth pairs

# the models or algorithms implemented
they used a _CNN_ [KEYWORD] [LINk] with hour glass modules to estimate the depth buffers for 2d images.  

# results and conclusions

They evaluate their performance using a held-out datasets from FIFA video game captures. the data was created using the same process as in training data and contained 32 RGB depth pairs of images. The metric they used to measure their performance was (scale invariant- Root Mean Squared) _st-RMSE_ [KEYWORD].  

They compare with three different approaches 

+ non human-specific depth estimation [LINK]
+ human-specific depth estimation [LINK]
+ fitting a parametric human shape model to 2D pose estimation [LINK]

The table below shows the result obtained

                        st-RMSE         IoU
Non-human training      0.92            -
Non-soccer training     0.16            0.41
Parametric Shape        0.14            0.61
Their Model             0.06            0.86


# Deep Learning in Basketball Trajectories


# the problem formulation and research objec­tive,

This research uses Recurrent Neural Networks _(RNN)_ [KEYWORD] [LINK] to predict whether a 3 point shot would be successfull or not 

# the methodological framework adopted by the authors,

Popular variant of RNN with long-short term memory (LSTM) is used. the network architecture relies on a two layered LSTM using peephole connections. the input to the LSTM is the XYZ data and the game clock. at each time step RNN predicts the probabilitiy of a successfull shot. the probability comes from a softmax layer and is trained based on cross entropy error.

An Adam optimizer is used in the model.

# the datasets utilized, including details on their acquisition and construction

the dataset used in the study stems from the publically availaible SportsVu dataset. SportVu is an optical tracking system
installed by the National Basketball Association (NBA) in all 30 courts to collect real-time data. The tracking system 
records the spatial position of the ball and players on the court 25 times a second during a game.

The dataset consisted of over 20,000 three point shots attempts from 631 games. the data was taken from the NBA site in the beginning of 2015-2016. The percentage of made shots in the data set is 35.7%. Fig 5 shows the example of the dataset used.

The first dataset consists of only the X, Y, Z, and game
clock variables representing the location of the ball in three
dimensions over time. X refers to the length of the court,
Y is the width of the court, and Z is the height of the ball.
A second dataset is created with additional variables based
on the physics of ball trajectories. The belief was that these
variables would add more information over just the location
data for machine learning models. Specifically, the added
variables included the difference in movement over each time
period for each dimension. Three other variables included:
the distance to the center point of the rim, the difference
over time for this distance, and the angle of the ball with
respect to the rim.


# the models or algorithms implemented

The research uses RNN (Recurrent Neural Networks) with the long short-term memory(LSTM) units 

# results and conclusions

The data is split in a 80:20 split for training and test respectively.

The metric they used to measure their accuracy was _AUC_ [KEYWORD] [LINK]. they build classifiers using a Generalized Linear Model (GLM) and gradient boosted machines (GBM) 

The below table showcase the results of the model with the baseline models.

        GLM         GBM     RNN
AUC     0.53        0.80    0.843


# Action Classification using LSTM RNN

# the problem formulation and research objec­tive,
    The main problem the researches are trying to solve is how can a video sequence which is just a series of frames be classified according to the actions that are beign peformed in the video. They picked Soccer to conduct this research on. They relied solely on the visual content analysis to classify different actions which is different from previous approaches who utilized prior knowledge to classify actions.

# the methodological framework adopted by the authors,

For every video, we divide it into frames, and each frame is converted into a descriptor (one descriptor per image). Then we train an LSTM-RNN to predict which action is being performed. These descriptors change over time according to the frames. The final decision is made by combining all frame-level decisions.

fig. 7 show's the approach.


Features are extracted in the following way :- 

+ Visual content representation: A Bag of Words approach
BoW is a method that recognizes objects using a histogram of visual words (meaning a pattern that repeats across many images). One BoW is taken for each frame.

+ A SIFT-based approach for Dominant Motion Estimation
Researchers added an extra feature called dominant motion, which captures movement in the video, especially the camera's movement.
They extract SIFT feature points from two consecutive frames, then match these points (using a KD-tree) to understand the motion. TV logos, which have no motion, are removed. RANSAC is used so that only camera motion remains while ignoring players' random movement.


# the datasets utilized, including details on their acquisition and construction
they used the MICC-Soccer-Actions-4 Dataset. 

# the models or algorithms implemented

Action classification using _LSTM-RNN_[KEYWORD][LINK] is done by feeding each frame’s descriptor to the network timestep-by-timestep, where the LSTM, an improved version of RNN, handles long term information using CEC (Constant Error Carousel) and gates that decide what to store or discard, overcoming the issue of RNNs forgetting old information in long sequences. The network architecture consists of one hidden RNN layer whose size depends on the input features, and a SoftMax layer at the output to make predictions at each timestep. A total of 150 LSTM cells are used more can cause _overfitting_ [KEYWORD], while fewer may prevent proper learning and the model is trained using Online _BPTT_ [KEYWORD] [LINK] with a _learning rate_[KEYWORD] of 10⁻⁴ and momentum 0.9.

# results and conclusions

all the experiments conducted by the researchers were carried out on MICC-
Soccer-Actions-4 dataset [LINK] with a _3-fold cross validation scheme_ [KEYWORD]

below is the table that showcase the results

                                            Classification Rate
BoW + k-NN [LINK]                              52,75 %
BoW + SVM  [LINK]                              73,25 %
BoW + LSTM-RNN                                 76 %
Dominant motion + LSTM-RNN                     77 %
BoW + dominant motion + LSTM-RNN               92 %

Fig 8 showcases the confusion matrices of different approaches

