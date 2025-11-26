#set page(
  paper: "us-letter",
  header: align(right)[
    Deep Learning In Soccer And Basketball
  ],

  numbering: "(1/1)",
  columns: 2,
)

#set par(justify: true)


#place(
  top + center,
  float: true,
  scope: "parent",
  clearance: 2em,
)[


#show title: set text(size: 17pt)
#show title: set align(center)
#show title: set block(below: 1.2em)

#title[
  A Survey on Applications  \ 
  of Deep Learning Algorithms in \
  Basketball and Soccer
]


#grid(
  columns: (1fr, 1fr, 1fr),
  align(center)[
    Daksh Sangal \
    IIIT Kota \
    #link("2023kucp1010@iiitkota.ac.in")
  ],
  align(center)[
    Ankit Goyal \
    IIIT Kota \
    #link("2023kucp1003@iiitkota.ac.in")
  ],
  align(center)[
    Mahesh Waghmare \
    IIIT Kota \
    #link("2023kucp1004@iiitkota.ac.in")
  ],
)

#align(center)[
  #set par(justify: true)
  *Abstract* \

  #box(width: 70%)[
  Deep learning and sports analytics represent a rapidly developing field among researchers. However, limited surveys exist in this domain and even fewer focus specifically on sport oriented research applications. To address this gap, we have combined two similar and highly popular sports, basketball and football, as our primary domains of investigation. We are trying to compare different researches in the area of application of deep learning in sports, examining how various neural network architectures and machine learning techniques have been employed to solve complex problems in game analysis and prediction. Our goal would be to compare research on three main topics for now: first, 2D to 3D modeling of games, which involves reconstructing three-dimensional spatial information from two dimensional video footage; second, Game Action Categorization, which focuses on identifying and classifying specific events, movements, and tactical maneuvers during game play; and third, Game Actions predictions or winning predictions, which encompasses forecasting future plays, player decisions, and overall match outcomes. We will compare scores of different models used in different researches across these three application areas, presenting the performance metrics, methodologies, and results in a comprehensive manner, and let the reader decide which is best suited for their specific use case or research direction.
  ]

]

]

#set heading(numbering: "1.")

#show heading: set align(center)
#show heading: set block(below: 1.2em)
#show heading: set text(
  size: 13pt,
  weight: "bold",
)
#show heading: smallcaps

#show heading.where(level: 2): set block(below:1.0em)
#show heading.where(level: 2): set text(
  size: 12pt,
  weight: "bold",
  style: "italic",
)

#show heading.where(level: 3): set text(
  size: 11pt,
  weight: "bold",
)

#show heading.where(level: 4): set text(
  size: 10pt,
  weight: "bold",
  style: "italic",
)

= Introduction
Sports analytics powered by deep learning represents an area where significant research has been conducted to improve the understanding and analysis of athletic performance. However, despite these efforts, this remains a field with limited comprehensive surveys and is still very much a developing domain. There is considerable scope for advancement in this area, and we aim to help with this survey by comparing different research that has happened since the early applications of deep learning in sports and examining what the future prospects might be for researchers looking to contribute to this field.
\

For our analysis, we have chosen soccer and basketball as the primary sports to focus on because they are among the most popular sports worldwide. These sports are kind of similar in nature, both are ball oriented team sports with dynamic gameplay, and importantly, much more research has been conducted in these two sports due to their global popularity and high consumer demand for advanced analytics. Therefore, we hope that conducting a survey that focuses primarily on these sports but is not limited to them alone would be able to generate the greatest impact and provide valuable insight that could potentially be extended to other sports as well.
\

After studying numerous researches in this domain, we have identified some key areas in sports analytics that researchers are actively trying to solve with deep learning techniques. Some of the major areas we will focus on are: first, 2D to 3D conversion, which involves transforming flat video footage into three dimensional spatial representations; second, Game Prediction and Win Prediction, which uses historical and real time data to forecast match outcomes; and third, Game Action Categorization, which involves identifying and classifying specific events and movements during gameplay.
\

What our aim is to systematically summarize the researches that different people and research groups have done on these topics. We will list their goods and bads, highlighting the strengths and limitations of each approach detail the models they used, examine the different approaches they took to solve similar problems, and document the datasets they used for training and validation. We will provide this comparison to the reader in a nice tabular and visual way, making it easy to understand and compare multiple studies at a glance. We hope to help fellow researchers with these comparisons so that if in the future someone extends this research or begins new work in these areas, they can take the best path forward by learning from previous successes and avoiding known pitfalls.
\
Additionally, we will be linking references to all the study material we read and analyzed, providing access to the datasets that were used by the researches that we mention in our survey, and including clear definitions for technical keywords and terminology to ensure our survey is accessible to both newcomers and experienced researchers in the field.


= KeyWords

This section will act as a reference to the reader providing quick definitions and explanations for terms commonly used in deep learning. this section is indented for readers that are not well versed in the field of deep learning and the explanations will reflect our intent. All the keywords whose definitions are mentioned here will be written in italics.


- ReLu 
- Sigmoid
- Negative Log Loss
- ECE
- Spatio Temporal Data
- LSTM 
- tanh

= Researches


This section constitutes the principal body of the survey. The research works examined are organized into three major thematic categories: Game Prediction, Game Action Categorization, and 3D Game Simulation. Each category is subdivided into multiple subsections, with each subsection dedicated to a single study included in the survey.

For every research work, we provide a structured and critical summary that encompasses:

- the problem formulation and research objectives,

- the methodological framework adopted by the authors,

- the datasets utilized, including details on their acquisition and construction,

- the models or algorithms implemented, and

- the empirical results and conclusions reported.

This organizational structure ensures clarity, coherence, and comparability across the diverse set of studies reviewed.

== Game Prediction

In this category researches focused on predicting probabilities of micro actions in game that might occur using different parameters such as player positions, ball positions, linup etc. This section is composed of two researches one for soccer and another for basketball and will summarize their approach the problem they solved, the dataset they used, the models they used and of course the results that were obtained along with the metrics that were used to measure them.

=== SoccerMap [LINK]

#figure(
  image("soccer_map_arch.jpg", width: 100%),
  caption: [Architecture of the SoccerMap model.]
)

==== Objective

SoccerMap aims to estimate continuous probability surfaces representing potential passing locations during a soccer match. The method enables coaches and analysts to visually inspect and understand player positioning, decision-making tendencies, and tactical structures throughout a game.

==== Methodological Framework

The model uses convolution layers to pick up meaningful patterns from the soccer field map at different zoom levels while keeping the image size unchanged and avoiding border issues. It downsamples the map with pooling to understand broader, more general patterns, then upsamples it smoothly to regain detail without creating visual artifacts. A fully convolutional design allows the model to make a prediction at every location on the field rather than just one final output. Finally, small 1×1 filters are used to turn these learned features into passing-probability predictions at each spot, and information from different zoom levels is combined so the model benefits from both fine and coarse details.

==== Dataset

high frequency, spatio temporal data is used. tracking data extracted from videos of soccer match consisting of 2d locations of players and the ball at 10 frames per second, allong with manually tagged passes, they arranged the data in a matrix of c x l x h, where l and h are the high level field coordinates and c represents the low level parameters that are passed to this architecture. low level features are calculated with things like liklehood of nearby teammates

they used tracking data, and event data from 740 English Premier League matches
from the 2013/2014 and 2014/2015 season, provided by STATS LLC [LINK].

==== Models Implemented

they used a combination of models combining them in a single architecture, they are mentioned in fig.1

==== Results and Conclusions

They split the data into 60:20:20 split which means 60% training data, 20% validation data, 20% test data.

They benchmarked against two models. Logistic Net and Dense2 Net. Logistic Net consists of a single _sigmoid_ [KEYWORD] activation unit while Dense2 is _Neural network_ [KEYWORD] with two dense layers followed by _ReLu_ [KEYWORD] [LINK] activation unit and a sigmoid output unit.

The Metrics they used are _Negative Log Loss_ [KEYWORD] and _ECE_ [KEYWORD] for every model 


#figure(
  image("soccer_map_results.jpg", width: 80%),
  caption: [Visual comparison of model outputs on pass-probability surfaces.]
)

#show table: set text(
  size: 8pt,
)

#table(
  columns: (auto, auto, auto, auto, auto),
  inset: 10pt,
  align: horizon,
  table.header(
    [*Model*], [*Log-loss*], [*ECE*], [*Inference time*], [*Number of parameters*],
  ),
  "Naive",
  $0.5451$,
  "-",
  "-",
  $0$,
  "Logistic Net",
  $0.384$,
  "0.0210",
  "0.00199s",
  $11$,
  "Dense2 Net",
  $0.349$,
  "0.0640",
  "0.00231s",
  $231$,
  "Soccer Map",
  $0.217$,
  "0.0225",
  "0.00457s",
  $401, 259$,
)


=== DeepHoops [LINK]

#figure(
  image("deephoops_arch.jpg", width: 100%),
  caption: [Architecture of the DeepHoops model.]
)

==== Objective

They created a Neural Network Architecture called DeepHoops which analyses spatio temoporal data of NBA games and predicts expected points to be scored as progression progresses. They estimate the probabiliites of terminal actions (e.g take goal, foul, turnover), each of these terminal actions is associated with an expected point value.

==== Methodological Framework

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

the architecture is mentioned above in fig 3. they used two modules one was a stacked _LSTM_ [LINK] [KEYWORD] network which was responsible in learning representation of the data up to time tao, this allowed for important information about on-court actions. the additional module was used to model who is on the courl to assess the impact of different lineups

they use 32 LSTM cells for each 3 layers.

==== Dataset

they used optical tracking data of 750 NBA games, which represents NBA games as athree dimensional coordinate system. the data they used was highly annotated and allowed labelling.

==== Models Implemented

They used a stacked LSTM network. mentioned in fig 3

==== Results and Conclusions

The metric they used to calculate the accuracy was _Brier Score_ [KEYWORD] [LINK] over 5 _epochs_ [KEYWORD] with minimum _improvement rate_ [KEYWORD] of 0.01

#show table: set text(
  size: 8pt,
)


#figure(
  caption: [DeepHoops Brier Score (BS ), Climatology Model Brier Score (BSref ), and DeepHoops Brier Skill Score (BSS ).
DeepHoops outperforms the climatology (baseline) model in all cases. Performance is best for K = 2 (among the values
examined). Epoch Time (in seconds) is lowest over all epochs],
  table(
    columns: (auto, auto, auto, auto, auto),
    inset: 10pt,
    align: horizon,
    table.header(
      [], [*BS*], [*BS_ref*], [*BSS*], [*Epoch Time (s)*],
    ),
    "K = 1",
    $0.4569$,
    "0.6070",
    "0.2472",
    $2180$,
    "K = 2",
    $0.3598$,
    "0.4920",
    "0.2686s",
    $2929$,
    "K = 3",
    $0.3094$,
    "0.4017",
    "0.2299s",
    $3552$,
    "K = 4",
    $0.2659$,
    $0.3371$,
    $0.2114$,
    $4200$,
  )
)
#figure(
  image("soccer_map_results.jpg", width: 80%),
  caption: [reliability Curves for DeepHoops’ probability estimates. The dashed line y = x represents perfect calibration]
)

