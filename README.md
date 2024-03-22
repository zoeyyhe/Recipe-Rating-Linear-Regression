# Palate Prophecy: Decoding the Flavor Formula, Unraveling the Intricacies of Review-Rating Dynamics

## Introduction
Our project will be centered around the factors that contribute to the rating of a recipe. Specifically, the relationship between the average rating a recipe has, accumulated from all of it's individual user ratings, and the preparation time of the recipe. Are recipes rated independently from it's preparation time? 

Our base dataset, before any data cleaning, is the combination of two smaller datasets, "Recipes" and "Ratings". Both datasets come from Food.com. The Recipes dataset has information regarding recipes that users submit, such as nutrition, number of steps, and preparation time. The Ratings dataset has ratings and reviews that users submit to specific recipes. Our final dataset is one that combines the two through recipe ID, such that there will be multiple rows a recipe for each user rating it has. The merged dataset has 234,429 Rows and 16 Columns. The Columns are id, minutes, contributor_id, submitted, tags, nutrition, n_steps, stpes, description, ingredients, n_ingredients, user_id, data, rating, review

| Column          | Description                                                                                  |
|-----------------|----------------------------------------------------------------------------------------------|
| id              | Recipe ID                                                                                    |
| minutes         | Preparation time for recipe                                                                  |
| contributor_id  | User ID who submitted the recipe                                                             |
| submitted       | Date of submission for recipe                                                                |
| tags            | Tags given to recipe by Food.com                                                             |
| nutrition       | Nutrition information in the form [calories (#), total fat (PDV), sugar (PDV), sodium (PDV), protein (PDV), saturated fat (PDV), carbohydrates (PDV)]; PDV stands for “percentage of daily value” |
| n_steps         | Number of steps in recipe                                                                    |
| steps           | Description of each step in recipe                                                           |
| description     | User-given description for recipe                                                            |
| ingredients     | Ingredients needed for recipe                                                                |
| n_ingredients   | Number of ingredients in recipe                                                              |
| user_id         | User ID                                                                                      |
| date            | Date of review submission                                                                    |
| rating          | Rating given to recipe                                                                       |
| review          | Review (text based) given to recipe                                                          |




## Data Cleaning and Exploratory Data Analysis

**1. Convert Data Types**

name               object
id                  int64
minutes             int64
contributor_id      int64
submitted          object
tags               object
nutrition          object
n_steps             int64
steps              object
description        object
ingredients        object
n_ingredients       int64
user_id           float64
date               object
rating            float64
review             object
dtype: object

We noticed that the tags, steps, ingredient, and nutrition columns are lists as type strings instead of list as type lists. For the data inside each value of the nutrition column, it was a list of strings instead of floats. To fix these problems, we converted the columns and their values to their appropriate types. 

**2. Identify Missing Values**

We noticed that the column rating has missing values stored as 0, which we filled with np.NaN.
Below is information about our cleaned dataset in which only columns relevant to our project are included. Details about our data cleaning processes can be found in the next section. 

**3. Extract New Feature Required for Analysis**

These are the new columns that we created:

| Column              | Description                                           |
|---------------------|-------------------------------------------------------|
| avg_rating          | Average rating of recipe from users                   |
| avg_review_length   | Average length of all reviews on recipe               |


We created these new features to better the distribution of ratings and reviews for each recipe. 

Cleaned Dataset Specifications
Number Rows: 234,429
Number Columns: 11

Columns
id, 
name, 
minutes, 
nutrition, 
n_steps, 
description, 
n_ingredients, 
rating, 
review, 
avg_rating, 
avg_review_length

TODO: show head of cleaned dataframe

## EDA
### Univariate Analysis

<iframe
  src="assets/uni_graph1.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

#### Ratings Histogram
<iframe
  src="assets/uni_graph1.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

This histogram shows the distribution of ratings given by users. We can see a left-skewed distribution in which most ratings are 5s. 

#### Average Length Histogram

<iframe
  src="assets/uni_graph2.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

We wanted to analyze the distribution of average review length for each recipe, so this histogram shows the distribution of the count of each average review length. We can see a right skewed distribution which could be the result of outliers.  

### Bivariate Analysis
#### Box plot of Ratings and Preparation Time

We wanted to analyze the relationship between ratings and preparation time. At first we created a scatter plot of ratings and minutes, but we couldn't draw any formal conclusion from it because the graph was impacted by an extreme outlier. Thus, we created a new column called time_level that encodes minutes into 5 quantiles named Very Short, Short, Moderate, Long, and Very Long. Using time_level, we created a box plot of each time level compared to it's respective average rating. From these group of box plots, we can see that across most time levels, the interquartile range falls within 4.5-5 average rating, except for recipes encoded as "Very Short" which has a higher mean and median of average rating compared to other time levels. 

<iframe
  src="assets/bi_graph1.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

<iframe
  src="assets/bi_graph2.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

#### Scatter plot of review length and average rating

<iframe
  src="assets/bi_graph3.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

This is a scatter plot of review length and average rating. We can see a slight positive correlation between average review length and average rating especially for ratings that are integer. 

### Interesting Aggregates

To explore the relationship between review length and preparation time and average rating of the recipes, we created a pivot table. To get a better visualization of the distribution, we encoded a column called "review_level" which assigns labels of Very Short, Short, Moderate, Long, and Very Long to their respective quantiles. We will also use the similarly encoded column called "time_level" that we created in previous analyses. 

From this pivot table, we noticed a tendency of higher average ratings for shorter review lengths and shorter time levels. This shows an interesting relationship between the three variables that resonate with our project goals. It would be natural to think that higher average ratings would have longer reviews, but this pivot shows the contrary. One explanation is that people who don't like the recipe spend more time writing about what they don't like!

## Assessment of Missingness

We have identified the following columns that have missing values:
rating
review
description

### NMAR Analysis
In the dataset, there's a column containing missing values, particularly in the "review" section, which we believe is Not Missing at Random (NMAR). The absence of data in this column is likely tied to reviewers having no strong feelings about the recipe which leads to no review, or they did not feel a need to leave a review. Therefore the missingness of the review would depend on the value itself which defines NMAR. 

Another column with a substantial amount of missing values is the description column. We believe that the missingness mechanism of description is also NMAR because recipe posters could make the decision to not have a description for their recipe. This means that the missingness of the description is not dependent on other columns but the value itself.

**Missingness Dependency**

We would like to analyze the missingness of the ratings column.

#### Minutes and Rating
Firstly, we would like to analyze the distribution of minutes when rating is and is not missing.

**Our hypothesis for our permutation test is:**
- Null Hypothesis: The missingness of rating does not depend on minutes.
- Alternative Hypothesis: The missingness of rating does depend on minutes.

**Our test statistic is:**
- Test Statistic: Absolute Mean Difference of minutes when column rating is missing and is not missing

<iframe
  src="assets/missing_graph1.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

For this test we choose a significance level of 0.05. 

**Our decision:**
Our p_val = 0.1, thus p > 0.05, so our decision is to fail to reject our null hypothesis. 

This means that based on our permutation test, the probability that there are values equal to or at least as extreme as our observed statistic under our null distribution is not significant enough to show that the missingness of data in the ratings doesn't not depend on minutes. As a result, with a significance level of 0.05, we conclude that it is likely that the missingness of ratings is unrelated to minutes.


#### Review Lengths and Rating
Now, we will analyze the distribution of average review lengths when rating is and is not missing to access it's missingness mechanism.

**Our hypothesis for our permutation test is:**
Null Hypothesis: The missingness of rating does not depend on review lengths.
Alternative Hypothesis: Average review lengths is higher when rating is not missing

**Our test statistic is:**
Test Statistic: Mean Difference of review lengths when column rating is missing and is not missing: (average review length when rating is missing - average review length when rating is not missing)

<iframe
  src="assets/missing_graph2.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

For this test, we continue to use a significance level of 0.05. 

**Our decision:**
Our p_val = 0.0 , thus p < 0.05, so our decision is to reject our null hypothesis(that the missingness of rating does not depend on review lengths). 

This statistical test shows that there was sufficient evidence that under the null distribution, our observed statistic is not due to chance. Therefore, we conclude that the missingness of rating is MAR because it is likely to be dependent on review lengths with a significance threshold of 0.05. 

A reason behind average review lengths being higher when the rating is not missing is that a user is more likely to write a review after already taking the effort to rate the recipe.

## Hypothesis Testing

What we are interested in researching is the analysis of the relationship between the average rating of a recipe and the preparation time of a recipe. Are recipes rated independently from preparation time?

**Our hypothesis are as follows:**
Null Hypothesis: Recipes are rated independent of preparation time. 

Alternative Hypothesis: Recipes are rated higher when preparation time is shorter. (There is an inverse relationship between recipe rating and preparation time)

**Our Test Statistic:**
Test Statistic: Difference between average rating of recipes that are of a long preparation time and a short preparation time

<iframe
  src="assets/hypothesis_graph.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

### Hypothesis Testing Conclusion

Our decision:
For this hypothesis test we will use a significance level of 0.05. After running the permutation test, we obtain a p = 0.0, thus p < 0.05, so our decision is to reject the null hypothesis.

This result shows that there is sufficient evidence that under the null distribution, our observed statistic is not due to chance. Therefore, it is likely that recipes are not rated independently. 

This result is plausible because recipes with shorter preparation times may differ from longer preparation in that shorter prep recipes are more convenient, lower chance of failure, and easier to process. Because of these differences shorter prep times would have higher ratings.

## Framing a Prediction Problem

Based on our EDA and hypothesis testing, we believe that factors such as average review length and a recipe's preparation time could have a connection to its rating. So, we are interested in predicting average rating because it would be useful to know what indicators lead to the average rating of a recipe. We would like to use average review length in our future regression model, but we aren't able to becuase at the "time of prediction" we would not have any information about the review as the ratings and review length is submitted together by the user.

**Our prediction problem:**
    is to predict average rating of a recipe, so our response variable is avg_rating. 

**Prediction Type:**
    Our model will be using regression, specifically the LinearRegression model from sklearn. 

**Response variable:**
    We will be predicting the average rating, so that will be our response variable.

**Evaluation metric:**
   will be R^2. 
 
 We chose this over RMSE because we are concerned with the proportion of variance in our response variable that our linear regression model predicts. The R^2 metric will give us more information about the relationship between our features and response variable rather than RMSE which is just the deviation between our actual and predicted ratings.

## Baseline Model
### Model Details
The model we are using is a linear regression model. For the variable we are predicting, we dropped the missing rating rows because it would not be possible to train a model to predict ratings that are missing. 

### Current Features
The features we will use are the minutes and description length.

We have two quantitative features. We did not encode anything because the features came as quantitative.

### Performance
To test our performance, we used a train-test-split from sklearn with a default split proportion of 0.25. We took the R^2 metric using .score on the model. 

The score we got on the training data set is 1.508524236615294e-05.
The score we got on the test data set is -0.00016719642015305958.

We believe that our current model performs poorly. This is because our R^2 is very close to zero for training score and negative for testing score. The coefficient of determination ranges from [0,1], so regression model explains none or close to none of the variability of our avg_ratings variable. R^2 is negative because our model is worse than predicting with a horizontal line. 

## Final Model

**New or Transformed Features:**
*Minutes*: 
We square root scaled and encoded the minutes using quantiles.

<iframe
  src="assets/final_model_graph.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

Firstly, we square root scaled it because the scatter plot between the minutes and avg_rating looks more similar to the graph of sqrt(x) rather than having a linear relationship.
We encoded the minutes as quantiles to account for the extreme outlier in minutes that we found.  

*Calories, Fat, Sugar, Protein*:
We extracted nutritional information (amount of calories, fat, sugar, and protein in each recipe) and introduced them as separate quantitative features. We did this because we believe that people are impacted by nutritional information when rating the recipe. We chose the nutritional details that we think get the most attention from people.

We encoded all four of these features using quantiles. This would be helpful to our model because the distribution of calories, fat, sugar, and protein are respectively very spread out and have outliers, so this encoding will mitigate influence by outliers.

### Modeling Algorithm
We are using sklearn's linear regression model. We built our pipeline to have a preprocessing column transformer that performed neccesary encodings during our feature engineering process. 

**Linear Regression Model's Hyperparameters**
For the hyperparameters of our linear regression model, we decided to not test the hyperparameters of copy_X, n_jobs, and positive, because they are either only for the computational performance of the model or are not relevant to our data. For example, the positive hyperparameter is only supported for dense arrays. For, the last hyperparameter, fit_intercept, the optimal value was default=True. This was tested using GridSearchCV with cv=5.

**Transformer's Hyperparameters**
The only other hyperparameters present in our pipeline is n_quantitles, the number of quantiles to encode our values in. We ran a GridSearchCV with cv=5 and got the optimal n_quantiles=10. For each QuantileTransformer we tested the values of [10, 50, 100, None],  

#### Performance
To test our performance, we used a train-test-split from sklearn with a default split proportion of `0.25`. We took the R^2 metric using .score on the model. 

The score we got on the training data set is `0.003718947933712413`.
The score we got on the test data set is `0.003643705985772683`.

Our final model's performance was a clear improvement over our baseline model. 
Our training R^2 is larger: 

> 0.003718947933712413 > 1.508524236615294e-05.

Our testing R^2 is also larger: 

> 0.003643705985772683 > -0.00016719642015305958. 

Since the R^2 metric on both the training and test dataset is larger, this shows that the quality of linear fit in our final model is better and that with our additional features, we explain a greater proportion of variance in average rating. This also shows that the additional features or transformations that we introduced are more relevant indicators of a recipe's average rating than the features we had in our baseline model.

## Fairness Analysis
To test the fairness of our model, we will compare the model's performance on two groups: high and low calories recipes.

The evaluation metric is RMSE so that we can access the accuracy of our model's predictions regarding potential disparities across our two different recipes. 

**Group 1**: Low Calories Recipes

**Group 2**: High Calories Recipes

We differentiated low and high calories recipes by binarizing low if the recipes' calories is less than the mean calories, and high if it is more than the mean calories. 

**Hypothesis**
Null Hypothesis: Model RMSE for low and high calories recipes are roughly the same, with any differences due to chance. (Model is fair!)

Alternative Hypothesis: Model RMSE for low and high calories recipes are different. (Model is unfair)

**Test Statistic**
Absolute difference in mean of average ratings between low and high calorie recipes.

### Conclusion
For this test we will use a significance level of 0.05. Through our permutation test, we got a p-value of 0.0, so p < 0.05, thus our decision is to reject the null hypothesis.

This means that with a significance level of 0.05, it is likely that our model performs differently for low and high calories recipes.























