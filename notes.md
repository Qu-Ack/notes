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

Model
Log-loss
ECE
Inference time Number of parameters
Naive
0.5451
−
−
0
Logistic Net
0.384
0.0210
0.00199s
11
Dense2 Net
0.349
0.0640
0.00231s
231
SoccerMap
0.217 0.0225
0.00457s
401, 259


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

the architecture is mentioned above in fig 3. they used two modules one was a stacked LSTM network which was responsible in learning representation of the data up to time tao, this allowed for important information about on-court actions. the additional module was used to model who is on the courl to assess the impact of different lineups

they use 32 LSTM cells for each 3 layers.

## the datasets utilized, including details on their acquisition and construction

they used optical tracking data of 750 NBA games, which represents NBA games as a three dimensional coordinate system. the data they used was highly annotated and allowed labelling.

## the models or algorithms implemented

They used a stacked LSTM network. mentioned in fig 3

## results and conclusions

The metric they used to calculate the accuracy was Brier Score over 5 epochs with minimum improvement rate of 0.01

BS
BSref
BSS
Epoch Time (s)
K = 1
0.4569
0.6070
0.2472
2180
K = 2
0.3598
0.4920
0.2686
2929
K = 3
0.3094
0.4017
0.2299
3552
K = 4
0.2659
0.3371
0.2114
4200

below are the reliability curves of the DeepHoops Model


