
# Article - Inference and Train

## Introduction

This project is part of an application that analysis the stock market sentiment through NLP.
The goal of this project was to build a back-end that would handle both the inference and training of a Deep Learning model, in this case Bert.

The Bert model is fine tuned on articles from the stock market that were scrapped on different specialized websites and labeled by hand. 

Once launched, the back-end gives access to two main endpoints:

```
/articles-inference 
method: post
requires a json in this format: 
{
    "article": "your article"
}
```
```
/summary
method: post
requires a json in this format: 
{
    "article": "your article"
}
```

The first endpoint returns a list of sentences with the sentiment, all wrapped in a json format: 
```
{
    "sentiment": [ "sentence number 1": "positive" ]
}
```
As for the summary, using Gensim's LSA implementation we provide a json in this shape:
```
{
    "summary": "summarized article"
}
```

## Technologies used
Flask was used for the back-end, as it is light weight, scalable and easy to associate with a Machine Learning project. 

The Deep Learning part of the project relies on HuggingFace library, as it provides flexibility with different ML frameworks. 
