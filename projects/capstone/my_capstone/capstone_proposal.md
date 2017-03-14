# Machine Learning Engineer Nanodegree
## Capstone Proposal
Simon Jackson
March 12th, 2017

## Proposal
_(approx. 2-3 pages)_

### Domain Background
_(approx. 1-2 paragraphs)_

** TO BE COMPLETED **

In this section, provide brief details on the background information of the domain from which the project is proposed. Historical information relevant to the project should be included. It should be clear how or why a problem in the domain can or should be solved. Related academic research should be appropriately cited in this section, including why that research is relevant. Additionally, a discussion of your personal motivation for investigating a particular problem in the domain is encouraged but not required.

### Problem Statement
_(approx. 1 paragraph)_

Imagine you work at Netflix and have added 100 new movies to the service. You'd like to include these new movies in recommendations for people, but who might like them? This project will attempt to solve this problem by investigating methods for predicting the 5-star rating a person will give a new movie.

Such predictions are directly useful for movie-streaming services like Netflix, who make recommendations based on these predicted ratings. The major challenge is that new movies have not been rated by users, unlike movies that already exist in the service.

The methods investigated are also likley to apply to similar problems faced by other services such as music-streaming services like Spotify (when new music is added) or accomodation services like Booking.com (when new hotels are added).

Given a data set of known movie ratings, models can be evaluated by the accuracy of their predictions. If 5-stars are treated as categorical outcomes, then metrics such as precision, recall, and F1 are appropriate measures of performance. If 5-starsare treated as continuous, then metrics such as *R^2^* are appropriate.


** TO BE COMPLETED **

In this section, clearly describe the problem that is to be solved. The problem described should be well defined and should have at least one relevant potential solution. Additionally, describe the problem thoroughly such that it is clear that the problem is quantifiable (the problem can be expressed in mathematical or logical terms) , measurable (the problem can be measured by some metric and clearly observed), and replicable (the problem can be reproduced and occurs more than once).

### Datasets and Inputs
_(approx. 2-3 paragraphs)_

The data used in this project will come from two open-source projects:

- [MovieLens 20M Dataset](https://www.kaggle.com/grouplens/movielens-20m-dataset): Over 20 Million Movie Ratings and Tagging Activities Since 1995
- [IMDB 5000 Movie Dataset](https://www.kaggle.com/deepmatrix/imdb-5000-movie-dataset): 5000+ movie data scraped from IMDB website


The MovieLens dataset contains individual user ratings of movies on a 5-star scale (with 5 being the best and 1 being the lowest).

The IMDB dataset contains public information about 5000 movies and includes the following variables:

> "movie_title" "color" "num_critic_for_reviews" "movie_facebook_likes" "duration" "director_name" "director_facebook_likes" "actor_3_name" "actor_3_facebook_likes" "actor_2_name" "actor_2_facebook_likes" "actor_1_name" "actor_1_facebook_likes" "gross" "genres" "num_voted_users" "cast_total_facebook_likes" "facenumber_in_poster" "plot_keywords" "movie_imdb_link" "num_user_for_reviews" "language" "country" "content_rating" "budget" "title_year" "imdb_score" "aspect_ratio"

The problem addressed here is to determine whether a user's ratings for certain movies can be predicted based on:

i. Their ratings of other movies (contained in the MovieLens dataset)
ii. Public information about that movie (contained in the IMDB dataset)

** TO BE COMPLETED **

In this section, the dataset(s) and/or input(s) being considered for the project should be thoroughly described, such as how they relate to the problem and why they should be used. Information such as how the dataset or input is (was) obtained, and the characteristics of the dataset or input, should be included with relevant references and citations as necessary It should be clear how the dataset(s) or input(s) will be used in the project and whether their use is appropriate given the context of the problem.

### Solution Statement
_(approx. 1 paragraph)_

The datasets will be combined to only consider movies that are included in both data sets.

All movies will be assigned a unit-normalized feature vector. This creates a multidimensional space in which movies that are similar to eachother (eg as determined by cosine similarity) are closer to eachother.

This multidimensional space can be overlayed with ratings (and error margins). For example, we could start with an unbiased model for which all points on the unit-space are intialized to have a rating of 2.5 stars with 2.5-star error margins.

Instead, we'd have a population-level unit-space which is refined based on population-level stats obtained in the IMDB dataset.

For a given user, we could insert their ratings into the unbiased unit-space. An algorithm can then interpolate between 



Unsupervised methods will be used to create similarity metrics / clustering of movies.

A user's ratings will be used to 

** TO BE COMPLETED **

In this section, clearly describe a solution to the problem. The solution should be applicable to the project domain and appropriate for the dataset(s) or input(s) given. Additionally, describe the solution thoroughly such that it is clear that the solution is quantifiable (the solution can be expressed in mathematical or logical terms) , measurable (the solution can be measured by some metric and clearly observed), and replicable (the solution can be reproduced and occurs more than once).

### Benchmark Model
_(approximately 1-2 paragraphs)_

In this section, provide the details for a benchmark model or result that relates to the domain, problem statement, and intended solution. Ideally, the benchmark model or result contextualizes existing methods or known information in the domain and problem given, which could then be objectively compared to the solution. Describe how the benchmark model or result is measurable (can be measured by some metric and clearly observed) with thorough detail.

### Evaluation Metrics
_(approx. 1-2 paragraphs)_

In this section, propose at least one evaluation metric that can be used to quantify the performance of both the benchmark model and the solution model. The evaluation metric(s) you propose should be appropriate given the context of the data, the problem statement, and the intended solution. Describe how the evaluation metric(s) are derived and provide an example of their mathematical representations (if applicable). Complex evaluation metrics should be clearly defined and quantifiable (can be expressed in mathematical or logical terms).

### Project Design
_(approx. 1 page)_

In this final section, summarize a theoretical workflow for approaching a solution given the problem. Provide thorough discussion for what strategies you may consider employing, what analysis of the data might be required before being used, or which algorithms will be considered for your implementation. The workflow and discussion that you provide should align with the qualities of the previous sections. Additionally, you are encouraged to include small visualizations, pseudocode, or diagrams to aid in describing the project design, but it is not required. The discussion should clearly outline your intended workflow of the capstone project.

-----------

**Before submitting your proposal, ask yourself. . .**

- Does the proposal you have written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Solution Statement** and **Project Design**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your proposal?
- Have you properly proofread your proposal to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
