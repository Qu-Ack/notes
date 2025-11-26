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





