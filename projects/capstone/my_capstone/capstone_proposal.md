# Machine Learning Engineer Nanodegree
## Capstone Proposal
Simon Jackson
March 12th, 2017

## Proposal
_(approx. 2-3 pages)_

### Domain Background
_(approx. 1-2 paragraphs)_

In this section, provide brief details on the background information of the domain from which the project is proposed. Historical information relevant to the project should be included. It should be clear how or why a problem in the domain can or should be solved. Related academic research should be appropriately cited in this section, including why that research is relevant. Additionally, a discussion of your personal motivation for investigating a particular problem in the domain is encouraged but not required.

### Problem Statement
_(approx. 1 paragraph)_

Imagine you work at Netflix and have added 100 new movies to the service. You'd like to include these new movies in recommendations for people, but who might like them? This project will attempt to solve this problem by investigating methods for predicting the 5-star rating a person will give a new, unrated movie.

Such predictions are directly useful for movie-streaming services like Netflix, who make recommendations based on these predicted ratings. The major challenge is that new movies have not been rated by users, unlike movies that already exist in the service.

The methods investigated are also likley to apply to similar problems faced by other services such as music-streaming services like Spotify (when new music is added) or accomodation services like Booking.com (when new hotels are added).

### Datasets and Inputs

The data used in this project will come from two open-source projects:

- [MovieLens 20M Dataset](https://www.kaggle.com/grouplens/movielens-20m-dataset): Over 20 Million Movie Ratings and Tagging Activities Since 1995
- [IMDB 5000 Movie Dataset](https://www.kaggle.com/deepmatrix/imdb-5000-movie-dataset): 5000+ movie data scraped from IMDB website

The MovieLens dataset contains individual user ratings of movies on a 5-star scale (with 5 being the best and 1 being the lowest). This data set provides the data to be trained and predicted in testing (5-star ratings). It also makes it possible to cluster people based on their movie preferences.

The IMDB dataset contains public information about 5000 movies and includes the following variables:

> "movie_title" "color" "num_critic_for_reviews" "movie_facebook_likes" "duration" "director_name" "director_facebook_likes" "actor_3_name" "actor_3_facebook_likes" "actor_2_name" "actor_2_facebook_likes" "actor_1_name" "actor_1_facebook_likes" "gross" "genres" "num_voted_users" "cast_total_facebook_likes" "facenumber_in_poster" "plot_keywords" "movie_imdb_link" "num_user_for_reviews" "language" "country" "content_rating" "budget" "title_year" "imdb_score" "aspect_ratio"

This dataset can be used to cluster movies on attributes other than user ratings.

Combined, these two data sets can be used to train and test models for predicting the ratings that users will give "new" movies. To tackle the "new" part, all ratings assigned to a subset of movies in the MovieLens dataset will be removed for test purposes.

### Solution Statement

A solution should:

1. Take a person who has rated movies in the MovieLens dataset
2. Take a movie that hasn't appeared in the MovieLens dataset, but has features about it available in the IMDB data set.
3. Return a predicted 5-star rating.

### Benchmark Model

Two benchmark models for estimating the solution (5-star ratings):

- The mean rating of all users for all movies.
- The mean rating of the user for whom a prediction is being made.

### Evaluation Metrics

Given a data set of known movie ratings, models investigating this problem can be evaluated by the accuracy of their predictions. If 5-stars are treated as categorical outcomes, then metrics such as precision, recall, and F1 are appropriate measures of performance. If 5-stars are treated as continuous, then metrics such as *R^2^* are appropriate. Different algorithms will be tested, therefore making it possible that both of these evaluation approaches will be used.

### Project Design
_(approx. 1 page)_

My expected approach will attempt to do the following:

- For any new movie, use unsupervised methods to find similar (rated) movies.
- For a particular user, use unsupervised methods to find similar users and how they would rate the movies being considered.
- Use this information in a supervised manner to make a prediction.

The expected workflow for generating these components will be to iteratively address the following tasks:

- Import data
- Clean data, including:
	- Investigating outliers
	- Handling missing values
	- Normalizing where appropriate
	- one-hot encoding where appropriate
- Feature engineering
- Develop unsupervised method(s) for clustering movies
- Develop unsupervised method(s) for clustering users
- Develop supervised method(s) to make movie prediction
- Assess performance of final model(s) using validation and test sets by:
	- Evaluating performance with relevant metrics (as described earlier)
	- Comparing to benchmark model performance
	- Using k-fold cross validation

** TO BE COMPLETED **


In this final section, summarize a theoretical workflow for approaching a solution given the problem. Provide thorough discussion for what strategies you may consider employing, what analysis of the data might be required before being used, or which algorithms will be considered for your implementation. The workflow and discussion that you provide should align with the qualities of the previous sections. Additionally, you are encouraged to include small visualizations, pseudocode, or diagrams to aid in describing the project design, but it is not required. The discussion should clearly outline your intended workflow of the capstone project.

-----------

**Before submitting your proposal, ask yourself. . .**

- Does the proposal you have written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Solution Statement** and **Project Design**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your proposal?
- Have you properly proofread your proposal to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
