# Script

> Talk outline / script. Beats to hit, transitions, what to show on screen, what to say at each turn.

## Open

(How the lesson opens — hook, story, question, or the demo we walk in on.)

sw 1.0 vs 2.0 vs 3.0
what is progress, who drives it, who does what

first say why we use ML: we tend to use ML when we don't know how to code, or we are just lazy maybe

Show example of feature and of test case. show green/ red dashboards 

Pick now a classifier. be it ML or gen AI.
in AI we need many examples. we think less about corner cases (maybe we should), but in general we want a test dataset that is "representative"



Example: email classifier for high priority vs normal email

pick a metric
pick a dataset
pick an eval workflow


Alternative metrics: recall, coverage (classifier that says "not sure" )




For now we assume we have as test set a random sample of our customer or production data. 

Accuracy

Confidence of prediction


imagine for each new release you pick a new test set, and results improve. what changed? what make the result go up?

Now, imagine you don't pick a new test set and use the old one....

Even if we could tak all data from our customers, we cant predict future data. so we always have uncertainty. 

We list (just to flash them) the source of uncertainty. here we focus on one.


Deep dives:
- list all the wikipedia metrics
- AUC


## Beats

1. 
2. 
3. 

## Demo / playground walkthrough

(What to show, in what order, with what numbers. What the student should feel by the end of this segment.)

- Demo a classification on a dataset, show what a dataset looks like
- Demo a traditional classifier (prebuilt, given to us)
- Demo a Gen AI classifier


Playground 1: sampling noise, shown via beta distribution
Playground 2: entropy of classes, how much better are we than a baseline classifier



## Close

(The single takeaway you want to leave students with.)


## Notes to self

we introduce multi-label classification (it's actually easier than binary)
Example is ticket assignment.


