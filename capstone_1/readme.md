# Capstone Project 1: 

# A Reddit troll rapid detection and warning tool

<p align="center">
  <img img src="181108_troll_rs.jpg" width="600"/>
</p>

### Problem:
Reddit allows anyone to participate in discussions in over a million subreddit forums. While Reddit’s format and rules give great freedom to communicate, they also allow participation by trolls and other bad actors who aim to disrupt online communities by posting argumentative, offensive or threatening comments. It‘s possible to identify some of these disruptive posters by their comment histories, but many trolls frequently create new accounts, which makes them hard to track.

This is a particularly difficult problem for Reddit moderators, who are responsible for ensuring that posts and comments meet community guidelines, and that the communities they manage remain positive and not toxic. In very large subreddit communities with political/ideological focus (for example r/politics), keeping track of trolls can be very difficult. I will develop an automated “troll early detection” tool for forum moderators that can monitor comments and report suspicious users before they can threadjack and disrupt otherwise civil discussions. For rapid detection, the tool will rely on comment text rather than posting history and votes, which can fluctuate initially. The troll early detection tool will be useful to moderators as an early warning that a threadjacking might be about to occur, so that they can assess and respond quickly if required.

### Methods:
- Using PRAW (Python Reddit API Wrapper) I will download a large sample of comments from a selection of subreddits with political or ideological topic focus, such as r/politics, r/republicans, r/democrats, etc, where trolls often post. Each comment will represent one text sample, with additional features such as: username, subreddit name, vote value and controversial flag state. Additional features may be derived from parent comments and replies, and an analysis of the user’s comment history. I will use these associated features and ground truthing to develop a classifier model that outputs a score representing the likelihood a comment was made by a troll and may be toxic. This score will then be used as a target variable to train an NLP classifier model using only comment text as the training sample. After a functional model is developed, I will apply it to a reddit bot that a user can subscribe to and configure to monitor selected subreddits for trollish comments, then alert the user.

### Deliverables:
- Jupyter notebook(s) explaining the procedure and model.
- A paper and/or blog post that describes the methodology
- A Reddit bot that monitors commenting in a subreddit and applies the model to alert subscribers when a suspected troll is commenting.
- User documentation.
