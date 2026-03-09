# Model Classes Implementation

## 1. only model implementation

My goal in this repository is just implementing model classes that I studied in papers. So if I need to experiment, I input the dataset and my code into a Large Language Model to run experiments. and experiment something that I'm curious about.

In this repository, I will post only code that I implemented, and the experiment's result. If I get insight with experimentation, I will write the conclusion, too.


## 2.0. weighted EASE - Motivation

Motivation was concept of confidence proposed in "Collaborative Filtering for Implicit Feedback Datasets" published on 2018.
In that paper, C uses count of interaction between user-item. And C operates how fit by count of interaction.
I used item-popularity rather than count of interaction, because there are many binary datasets that express only interacted/uninteracted.

Before I experimented weighted EASE, I observed popularity-based weight was effective at some models(BERT4Rec, gSASRec). So I thought "What is there changes in EASE with confidence?". So I experimented that, However, I did not observe meaningful improvements.

## 2.1. weighted EASE - Experimentation

I experimented with weighted EASE. It is in file "./Recommendation/EASE.py". Its implementation detail is popularity-based confidence.

Let C is confidence and P is popularity(item frequency),
    P_ij = P_i + P_j
    C_ij = 1 + alpha * log( 1 + P_ij )

I input my weighted EASE model code, and I experimented on movielens 100k dataset. But I can't observed meaningful result.

| k    | EASE hit | EASE ndcg | wEASE hit | wEASE ndcg | wEASE_with_C hit | wEASE_with_C ndcg |
|------|----------|-----------|-----------|------------|------------------|-------------------|
| @10  | 0.1400   | 0.0731    | 0.1474    | 0.0737     | 0.1453           | 0.0733            |
| @50  | 0.3934   | 0.1279    | 0.3913    | 0.1258     | 0.3902           | 0.1258            |
| @100 | 0.5589   | 0.1547    | 0.5514    | 0.1517     | 0.5493           | 0.1516            |

I observed improvements on only recall@10. There's not result following my expectation. So I didn't draw conclusion about this experiment.

