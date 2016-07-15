---
layout: resource
title: "Lesson 10 - Videos"
---


<style>
img {max-height:400px;}
</style>

## Unsupervised Learning and Principal Components Analysis 

<center>
<iframe width="560" height="315" src="https://www.youtube.com/embed/ipyxSYXgzjQ?list=PL5-da3qGB5IBC-MneTc9oBZz0C6kNJ-f2" frameborder="0" allowfullscreen></iframe>
</center>

##### Slide 1: 

*Tibshirani:*  Welcome back. Today's lecture is about unsupervised learning. So let's recall what we meant by supervised learning and contrast it with unsupervised learning. So supervised learning-- the word supervised, remember, refers to the fact that there's a target, a label that we are trying to predict from training data. So we have features and we're trying to predict the label and the labels supervises the learning. For example, like if you're in kindergarten, you can think of a kindergarten teacher who shows a child in a class of a bunch of examples of say, Johnny, here's five examples of a house and Lego blocks. Here's five examples of a car. And he tells Johnny that these are houses and these are cars. So he's supervising the learning. And the child then looks and figures out which features of the house make it characteristic of a house and which features of the car make it characteristic of a car. So that's supervised learning. In contrast, unsupervised learning, there's no labels. So a kindergarten teacher would just show Johnny, here's 10 things. And doesn't tell Johnny that these five are cars and these five are houses. And the child then looks at the objects and tries to figure out some patterns. He may realize, these five look similar. I don't know what they're called, but they look similar. So maybe I should group them together. These other five looks similar to each other and I'll put them in a group. So, yes? So this means with supervised we have a $$Y$$ and with unsupervised, we don't. Exactly. So in both we have features, but as Trevor said, in supervised we have a $$Y$$ that we're given in the training data that's a true label. And unsupervised learning is harder in a sense because we don't have the actual labels. So let's move to slide one. Most of the course, as we've seen, focuses on the first exercise, supervised learning, where we have labels. But today we're going to talk about the setting where we don't have labels, unsupervised learning. All we observe are the features, which as before we'll call $$X_1$$ through $$X_p$$. And we want to know how the features relate to each other. 

##### Slide 2: 

*Tibshirani:* So in particular, what are the goals of unsupervised learning? Well, first of all they're not as clear as they are for supervised learning where the object is to predict $$Y$$ from $$X$$. Now we don't have a $$Y$$, so the objectives is a little more fuzzy. For example, we might want to discover subgroups among the observations. Like in my kindergarten example, right, the child may try to discover subgroups among the objects he's seen. We might want to know, is there a good way to view the data? To find the important features or the features that have the most variation over the different objects. So we're going to discuss-- there are lots of methods for unsupervised learning. But in this short lecture we're just going to talk about two of the most important ones. First is *principal components analysis*, which is a tool for viewing data or for pre-processing the features for use later in supervised learning. And the second is *clustering*, which is a group of methods for a class of methods for grouping the objects into different subgroups. So we'll talk about those two methods today. 

##### Slide 3: 

*Tibshirani:* But let's say a little more in general about the challenge of unsupervised learning. As I mentioned, it's a little more fuzzy because there's no simple objective like prediction. There's no $$Y$$ available, so we're not predicting. The objective, as we saw in the previous slides, is a little more fuzzy. But nonetheless unsupervised learning is actually of growing importance. And there's a number of reasons for that. First of all, let's see some examples. An example, as we'll see at the end of the lecture, an actual example is we have breast cancer patients from whom we've measure gene expression using gene chips. And we want to group those patient into subgroups of breast cancer. It turns out these subgroups are actually quite different in terms of their characteristics biologically and their survival as patients. Another example, which is in marketing is to find-- if we have shoppers, and we can record their browsing and purchase histories, we can group or segment the shoppers into different groups. And then maybe they'll be targeted with different kinds of advertising because their behavior are different. Another example which is quite popular is to group movies by the ratings assigned by movie viewers, like thriller, romance, et cetera. 

##### Slide 4: 

*Tibshirani:* The other thing that makes unsupervised learning more and more important is that there's a lot more *unlabeled* data available. And that's because in order to get labels for data, it can be costly or time consuming. 

*Hastie:* There's tons of images on the web. I mean, everybody is loading up images on Google and other places. And mostly they're unlabeled. No one's told us exactly what's in the picture but we have the picture. 

*Tibshirani:* Right, and the point is that kind of information can be collected by machine, the features, the images. But the actual labeling, it often requires human intervention. And that's going to be more costly and time consuming. An example that is movie reviews on the web. A lot of people try to predict, they try to correlate movie reviews with movie quality and group movies. One problem is that if you have a movie written by a human being it's hard to tell by machine whether or not that is actually favorable or not. The movie review could have some sarcasm in it. And for a human reading the paragraph, it's pretty easy to say, ah, that person doesn't like the movie or does like the movie. But for a computer, it's not so easy. So that's an example where finding the actual label can be quite difficult and time consuming. 

##### Slide 5: 

*Tibshirani:* OK, so let's start with the first main method for unsupervised learning, principal components analysis. This goes back probably for the 1930s in statistics, when it was first invented. So PCA a produces a low dimensional representation of a data set. And it finds that a sequence of linear combinations of the variables are features that have maximal variance. And at the same time, they're uncorrelated. So we're going to see, there's a first principal component which has the highest variance across the data. It's a linear combination of the features. And the second component, which is uncorrelated with the first, which has the highest variance under that constraint, et cetera. 

*Hastie:* Just imagine you have tons of variables and many of them are correlated. That can be quite an unmanageable set. What principal components tries to do is pare the set down into some important variables that summarize all the information in the data. 

*Tibshirani:* And that's these principal components. And that can be very useful just as a way of viewing the data. If you have a high dimensional data set, you just want to look at, gee, what's really happening here? What's important? The principal component view is one of the most important ways of displaying the data. And second of all, if you have a lot of features that you do want to use from supervised learning, the principal component summary of those variables could be a good set of new variables to use for supervised learning. 

*Hastie:* Principal components and the techniques related to it is one of the most widely used tools in statistics and data analysis. 

##### Slide 6: 

*Tibshirani:* OK, so let's talk about-- let's actually define principal components. So we have a set of variables, $$X_{1}$$ through $$X_{p}$$. And the principal component, the first principal component $$Z_{1}$$ is a linear combination of those variables.

$$Z_1=\phi_{11}X_1+\phi_{21}+X_2+\dotsb+\phi_{p1}X_p$$

It's a linear combination to find out highest variance across the data set. Now of course we're going to choose weights. It's defined by a set of weights, $$\phi_{1}$$ through $$\phi_{p}$$. But if I was allowed to make those weights as big as I wanted, I could make the variance of $$Z_{1}$$ as big as I wanted. So we need some constraint on the $$\phi$$s. And the natural constraint is that they're normalized.

$$\sum_{j=1}^p\phi_{j1}^2=1$$

So the sum of their squares is 1. That now makes the problem a sensible one, to choose a set of weights that has the highest variance. Those weights are called loadings in some areas of statistics and other social sciences. So the $$\phi{11}$$ through $$\phi{1p}$$ are the low ends of the first principal component. And the principal component of the low end vector is the set of those $$p$$ numbers. 

*Hastie:* If you didn't constrain them, of course, you could just make them much bigger. And that would just make the variance higher. So you need to tie them down. 

##### Slide 7: 

*Tibshirani:* Exactly. So here's an example. 

<center>
<img src=Images/10.1.PNG alt="PCA Example">
</center>
{:refdef: style="text-align:center"}
**Figure 10.1**
{: refdef}

Before we talk about how to actually compute the components, let's just see what the results of the first two components for this data, which is ad spending versus population. So the red points are my data points. I plotted them against these two variables. So $$p$$ is just 2 in this example. The first principal component is given by the projection of the data onto this line. So this direction has the highest variance among all combinations of the features in this data. Correspondingly this has the lowest variance, right below. Well, yeah, that's true, this has the lowest variance. So here's only two components. So I've got the highest variance, the first principal component. The second component is also-- well, it's got the highest variance with the constraining beyond correlated with the first component. Which means this has to have right angles with the first component, there it is. (If you only have two variables, you can only get out two components.) Exactly. Just for illustration we show you two variables. 

##### Slide 8: 

*Tibshirani:* OK, so how do you actually compute the components? Well, now suppose we have our data, which are features $$n \times p$$ at the $$\textbf{X}$$ matrix. And since we're only interested in the variance, we can center the variables to have a mean 0. We don't care about the mean, we're just worried about the variance. So in other words, in particular and specifically we make the column means of $$\textbf{X}$$ zero. And now we want to find the combination of the variables that has the highest variance. Remember, highest variance under the constraint that these loadings have sum of squares 1. Now since the data have been arranged to have a mean 0 in the first step, that means the $$z$$'s have mean 0. And hence the variance of the $$Z$$'s is the sum of their squares. 

##### Slide 9: 

*Tibshirani:* OK, so to continue the computation, we have the $$Z$$'s, which we've defined to be the sum of the loadings times the features. And so now the problem now, we can replace the $$Z$$'s by their expression from the previous slide. 

$$\displaystyle\max_{\phi_{11},\dotsc,\phi_{p1}}\frac{1}{n}\sum_{i=1}^n\left(\sum_{j=1}^p\phi_{j1}x_{ij}\right)^2 \text{ subject to } \sum_{j=1}^p\phi_{j1}^2=1$$

And we want to find the highest variance now, which is this expression subject to the fact that the loadings have to be normalized. So this is now just a computational problem, where the unknowns are these $$\phi$$s. And the optimization can be done by the singular-value decomposition, which is a standard technique in numerical analysis. Which we won't cover, but if you want to read about it yourself, it's very interesting and very important in a lot of areas of statistics. (And it's covered, for example, in our book Elements of Statistical Learning.) So when we've solved this problem and we've got the best loadings, the resulting $$Z$$ values are the first principal component. Which we'll write as $$z_{11}$$ to $$z_{n1}$$. 

*Hastie:* So it's like we've created a new variable now. We had our original $$p$$ variables. Now we've created a new variable, which is the $$Z_{1}$$, which has $$n$$ values just like each of our original variables. 

##### Slide 10: 

*Tibshirani:* Exactly. Now we can think about it geometrically, and we'll go back to the picture in a moment. So the loading vector is a direction of the features that if we project on that direction has the highest variance. And the values of the projected data are the principal component scores. So let's go back to the picture of the two dimensional one (Figure 10.1). So I said this was the first principal component. That means the loading vector is the vector that points in the direction from the middle to the northeast, right? So here it might be something like maybe 1,1, roughly. So that's the combination of the features that we compute using principal components. The actual computed values are the projections onto this line. So we can start, say the origin could be here, we could just measure. Each point, we take its projection onto this line. We measure how far it is away from say, the origin. So $$Z$$'s for these guys would be positive and the $$Z$$'s for these guys would be negative. So we replaced each point by basically how far along this line is it. And that's the first principal component.

## Exploring Principal Components Analysis and Proportion of Variance Explained 

<center>
<iframe width="560" height="315" src="https://www.youtube.com/embed/dbuSGWCgdzw?list=PL5-da3qGB5IBC-MneTc9oBZz0C6kNJ-f2" frameborder="0" allowfullscreen></iframe>
</center>

##### Slide 11: 

*Hastie:* OK. So we've seen how to compute the first principal component. And the description was that it provides a summary of the data. It's got the most variance. Well, you can go further. If you've got $$p$$ variables, you can now ask for a second principal component, which also has large variance. But unless you want to get the same one as the first, you have to put some constraints. So the natural constraint is that the second variable be *uncorrelated* with the first. So it's going to tell you different information about the data. And subject to being uncorrelated, it's going to have its largest variance. So that's a problem which we can phrase in pretty much the same way. 

$$z_{i2}=\phi_{12}x_{i1}+\phi_{22}x_{i2}+\dotsb+\phi_{p2}x_{ip}$$

It's going to be a linear combination. We've got it now indexed by 2 instead of 1. There's a sub 2 there. And subject to the constraint of being uncorrelated, we wanted to maximize the variance. 

##### Slide 12: 

*Hastie:* Now, there's some interesting math that comes with this. It turns out that being uncorrelated also means that the fees that define the principal component as vectors-- those are the loading vectors. So the fees for the second component defines a vector that's orthogonal to the fees for the first component. So it's just a property of the solution. And so we can characterize that lack of correlation as orthogonality of the fees. And the solution turns out also comes from the singular value decomposition of $$\textbf{X}$$. So the second principal component is defined by the second right singular vectors. And there's going to be a third component, a fourth component, and so on that sequentially maximize variance subject to being uncorrelated with all of the previous ones. And it turns out that you can have at most the $$n - 1$$ or $$p$$ principal components. So $$n$$ is the number of data points. Sometimes $$p$$ is bigger than $$n$$. And if that's the case, then we're restricted by $$n$$. But otherwise, we're just restricted by $$p$$, the number of variables. So we get a complete decomposition of the data matrix in terms of its principal components. 

##### Slide 13: 

*Hastie:* So as an illustration, we're going to show you an example using some data on USA arrests. So for each of the 50 states in the United States, we've got the number of arrests per 100,000 residents for each of three crimes-- assault, murder, and rape. And we also have measured the urban population, the percentage of the population in each state living in urban areas. So there are 50 observations. So each of the principal components are going to be end vectors with length 50, because they're new variables. And we'll get at most 4 principal components. And each of them will be described in terms of loading vectors of length $$p$$ equals 4. In this case, we actually standardized the variables before we did the principal components to have mean 0 and standard deviation 1. That's a fairly standard practice. And we'll get back to that in a moment.

##### Slide 14: 

*Hastie:* But let's first look at the results. So here's the results.

<center>
<img src=Images/10.2.PNG alt="USA Arrests PCA plot">
</center>
{:refdef: style="text-align:center"}
**Figure 10.2**
{: refdef}

This picture is called a biplot. And it's a way of displaying the principal components and the loadings in a single plot. So each of the blue, well, they're not points but marks on a picture, is a state. And so their names are there. The origin is at the center over here. So everything's been centered with respect to the origin. And on the plot as well, what we see are the loading vectors. So this plot is the plot of the first principal component along the horizontal axis here and the second principal component on the vertical axis. So we plot the first two principal components, which is fairly standard, because those are the most two important ones. And so in order to understand the principal components, each observation, which is a state, has a score on the principal component. And so some are large and positive. And some are negative. And then to understand these scores, we look at the loadings. And so the loadings are plotted as vectors here, because for each variable, we've got its loading on the first principal component and the second principal component. So that allows us to plot it as a vector in this space. And what we read from this is that the first principal component has got positive loadings, large positive loadings, on rape, assault, and murder. So all of the crimes are positive loadings on the first principal component. So that tells us that this direction on the positive side is loaded up on all three crimes, so roughly about the same amount. And so these are going to be high crime regions. These are going to be low crime regions. And so now, you can look at where the states lie. For example, Florida is sitting right out on the first principal component. So that's a high crime area and so on. So you look at the positions where states project to see how they relate to the principal component. Now, the second principal component seems to get largely loaded on urban population. So this is a high urban population. This is low urban population. And you can even interpret things like rape seems to be on the positive side in high urban populations than in low urban populations. So again, this biplot is a picture that lets you both display the first two principal components and also display the loadings and give you a way of interpreting the picture. (Trevor, what does it mean for a minus 3? What is that?) Well, minus 3 means that you're low down. The actual units are somewhat standardized units. But it means you're on the negative scale in the crime rate. So these are low crime areas. And so some of the extreme low crime areas are-- if I can read the text-- Maine is a low crime area. (New Hampshire, Iowa.) 

##### Slide 15: 

*Hastie:* OK. So here's some details that explain the figure, which we've just gone through. You can read it in your own time. 

##### Slide 16: 

*Hastie:* It's customary also to produce a table of the loadings.

<center>
<img src=Images/10.3.PNG alt="PCA loadings" style="max-width:300px">
</center>
{:refdef: style="text-align:center"}
**Figure 10.3**
{: refdef}

And so this just summarizes what we saw in the picture. So we see that in the first principal component, pretty much equal loadings on murder, assault, and rape. Whereas urban population has somewhat of a lower loading. There's a second principal compartment. Urban populations got a high loading. And murders got quite a high loading as well. Oh, that's a negative loading. But rape was a slightly positive loading. So you can actually look at the numbers. But it's often better and more useful to look at them graphically. 

##### Slide 17: 

*Hastie:* OK. So that's the primary use of principal components to come up with summaries of the data that explain mostly what's going on. But there's another view of principal components. And that's in terms of approximating a cloud of data by a low-dimensional hyperplane. And it turns out these two views are equivalent to each other.

<center>
<img src=Images/10.4.PNG alt="Hyperplane PCA">
</center>
{:refdef: style="text-align:center"}
**Figure 10.4**
{: refdef}

So here's a data set. It's an artificial data set. In the left-hand plot, we see some points. We're visualizing them in three-dimensional space. And what we're shown here is a two-dimensional hyperplane that's meant to approximate those points. Now, the points are colored. But that's just to help us explain what's going on. This seems to be like a clustering. But what we see is that this hyperplane passes through the middle of these points. And we're asking for a hyperplane that gets as close as possible to the points. And what we mean by that is that if we take each point and compute the shortest distance to the hyperplane and sum of the squares of those distances, we want the hyperplane that gets closest to the data in that sense. And it turns out that the hyperplane is defined in terms of the two largest principal components. In other words, the direction vectors that define the hyperplane, if you think of having two-direction vectors in the hyperplane, it'll be orthogonal to each other. Those are also the two-direction vectors of the largest two principal components. And that makes sense. We're in three dimensions here. So we've got three variables. To get closest to the data to find the plane closest to the points, we're asking actually that the projection of the points in the plane are spread out as much as possible. That'll be a characteristic of the solution. In other words, spread out means as much variance as possible. So that gives you an intuition of why the two principal components also characterize the solution. (Now, I'm confused here. We described it being close to points. It sounds to me like least squares regression, which we covered earlier. Is there a difference at all?) That's a good question, Rob. Least squares regression is also finding a hyperplane that gets close to the data in some sense. But there closest is defined just in terms of distance from the hyperplane to $$Y$$, the response, which was a supervising variable. But here we don't have any $$Y$$. So yeah. It's shortest distance of the hyperplane to do $$X$$'s themselves. Could you draw a picture as a scatter plot to show this? Could I draw a scatter plot to show this? The distinction between those two. OK. So for linear regression, we have $$Y$$. And well, I can only draw one $$X$$. Maybe I can draw a second $$X$$ in the picture. So that's $$X_{2}$$, $$X_1$$. This is $$X_{1}$$. And we've got a whole lot of points. And now, we find a hyperplane. But in linear regression, we're focusing just on this distance over here, which is not perpendicular to the hyperplane. It's just distance in the $$Y$$ direction. Whereas in principal components here, there's no special variable that's a $$Y$$. And so we define distance in terms of the perpendicular distance to the hyperplane. How is that picture? (Better than I could have done.) All right. 

##### Slide 18: 

*Hastie:* So this slide here just describes what we've just gone through in detail. 

##### Slide 19: 

*Hastie:* Now, I deferred those points. Scaling of the variable matters. So here's our first two principal components and our biplot again.

<center>
<img src=Images/10.5.PNG alt="PCA Scaling">
</center>
{:refdef: style="text-align:center"}
**Figure 10.5**
{: refdef}

But this was computed when we standardized each of the variables to have unit variance. What if we didn't do that? Well, you get a very different picture then. The picture of the first principal component seems to be mostly assault. Well, it turns out that the variable assault had by far much bigger variance in the original data set that any of the other variables. It's just the units in which it was measured. And so it just dominates the first principal component. And if you have variables that are actually measured in different units, it's quite easy to have one variable that just on its own has way bigger variance than the rest. And it will dominate the principal components. So if you want to avoid the effect of units or just individual variables having a big variance, then you standardize the variables. And in that case, what you're dealing with is variables all measured on an equal scale. And now, you can really see which components add together to give you large variance. 

##### Slide 20: 

*Hastie:* So there's also associated with principal components decomposition of variance. So we start off. We've got our variables. And we can talk about the total variance that's about to be explained, which is the sum of the variance of all the variables.

$$\displaystyle\sum_{j=1}^p\text{Var}(X_j)=\sum_{j=1}^p\frac{1}{n}\sum_{i=1}^nx_{ij}^2$$

And if the variables have been centered to have mean 0, that's given by this expression over here-- 1 over $$n$$ summation $$X_{ij}^2$$. This is going to be the variance of the $$j^{\text{th}}$$ variable. And we sum that over the $$p$$ variables. And the variance of the $$m^{\text{th}}$$ principal component is going to be the sum of the $$z_{im}^2$$.

$$\text{var}(Z_m)=\frac{1}{n}\sum_{i=1}^nz_{im}^2$$

Again, the $$z_i$$'s are going to have mean 0, because the original $$X$$'s at mean 0, they'll inherit that property as well. So this is an expression for the variance of the $$m^{\text{th}}$$ principal component. Now, you can show that the sum of the variances of all the $$X$$'s is the same as the sum of the variances of all the $$Z$$'s. That's just a mathematical property. And so we can talk about the proportion of variance explained by looking at the variance of an individual $$Z$$ relative to the sum of the variances. And that gives you an idea of the importance of each of the components. 

##### Slide 21: 

*Hastie:* And that'll be a quantity that's between 0 and 1, because the sum of the variance of the $$Z$$'s is in the denominator. 

$$\frac{\sum_{i=1}^nz_{im}^2}{\sum_{j=1}^p\sum_{i=1}^nx_{ij}^2}$$
<center>
<img src=Images/10.6.PNG alt="Proportion Variance">
</center>
{:refdef: style="text-align:center"}
**Figure 10.6**
{: refdef}

So this left plot on the crime data shows the proportion of variance explained by each of the principal components. So the first one explains about 0.6 of the variance or 60% of the variance. The second component explains just over 20% of the variance and so on. So these necessarily decrease because, after all, we've been asking for the first principal component to explain the most of the variance. The second one has to be uncorrelated with the first. And so these necessarily have to go down. And sometimes we plot the cumulative percentage of variance explained, which is, in this case, this is the percentage of the first principal component of the total variance. This is the percentage of the first two and so on. And the idea is that we want to know basically when to stop, how many components we need. And so we're going to be happy to stop if we've explained most of the variance. Let's say the first two principal components explain 95% of the variance. We'd be happy to stop there. Here it looks like we need to first three principal components, because there's no real hard and fast rules on how many components you need. So this percentage variance explained is one way of getting a feeling of when you've got most of the variance explained. 

##### Slide 22: 

*Hastie:* So the question that's just come up is how many components should we use? And so there's no real simple answer. One thing you might think is, why not use cross validation? Is that available for this purpose? Well, it's not really. And you might think, why not? We used cross-validation in regression. It was very useful there. Well, we don't have a response here. Cross-validation was used when we had a response. And we could see how well we were predicting the response when we left some data out. But we don't have a response here. So there's no supervising variable to help us. So does that mean cross-validation is completely ruled out? Well, not really. One of the uses of principal components, which we touched on earlier in this course, was in regression. When you had a lot of variables, you could use principal components to reduce the data to a few important variables, i.e., the first few principal components, and use them in the regression instead. Well, now you can use cross-validation, because we do have a response in that case. And so you might decide how many principal components you need to use in the regression. And you could use cross validation to help you decide that. But there you have a supervising response. And as I mentioned in the previous slide, that's sometimes called a scree plot. And you can look for an elbow in the scree plot, which is the percent variance explained plot. So if your percent variance explained look like this, if this was 1 and this was 0, and if the first one was up there, the second one was over there, and then the rest were all down here, you'd say, well, two principal components seems to be explaining a lot. This is a big elbow here. So if you look for elbows in the scree plot, that's often helpful. 

##### Summary:

*Hastie:* So that's the end of principal components. Again, principal components, one of the most useful tools we have in applied statistics. It's used all over the place. And it's got many other applications as well, which we won't touch on here. So it's a very useful tool. In the next session, we're going to talk about the second most useful tool or another very important tool, which is clustering. And we'll tell you about two different methods for clustering.

## K-means Clustering

<center>
<iframe width="560" height="315" src="https://www.youtube.com/embed/aIybuNt9ps4?list=PL5-da3qGB5IBC-MneTc9oBZz0C6kNJ-f2" frameborder="0" allowfullscreen></iframe>
</center>

##### Slide 23: 

*Tibshirani:* Well, welcome back. The second method of the section is clustering, which refers to a broad set of techniques for finding subgroups or clusters in data. So we're trying to segment or partition the data into subgroups that are similar to each other. And we'll have to define what we mean by similar or different in order to do the partitioning. And in a lot of cases, the way we decide to cluster, the way we decide on what's similar or different depends on the context. 

##### Slide 24: 

*Tibshirani:* So how does principal components analysis, which we just discussed, how does it contrast with clustering? Well, PCA, as we've seen, looks for a low dimensional representation or view of the data that explains a good fraction of the variance, right? So we saw the PCA plots. And from it, we derived new variables, which could be used for other methods, like supervised learning. Clustering on the other hand looks for homogeneous subgroups of the observations. So it's not looking for variance but is looking for similarity among observations. 

##### Slide 25: 

*Tibshirani:* For example, if we were trying to do a segmentation of a market, suppose for each of a number of customers we measured things like their income, their occupation, how far do they live to the closest urban area and so forth, and we want to segment them or group them into customers that are similar with regard to these features. Now, why do we want to do that? Because maybe, if they're similar with regard to these features, then the kind of advertising we use for that subgroup will be important. So we use a certain kind of advertising for one subgroup, like maybe young males who have a lot of money in one subgroup. Another subgroup might be-- Like me. Yeah. OK. Another group may be housewives in their '50s who their children have grown up. And they like to travel, for example. So we may want to advertise in a different way to that subgroup. So the task of segmenting that kind of population is a key application of clustering. 

##### Slide 26: 

*Tibshirani:* So we're going to talk about two clustering methods in this section, although there are lots more. In most areas of statistics, there are many, many ways of doing things. But the two most important methods which we'll talk about are *K-means clustering*, in which first we'll predefine $$K$$, the number of clusters. And then we'll see there's a way of grouping the observations into the $$K$$ groups. And then, of course, we'll have to define what $$K$$ is, this number of subgroups. And that's going to be an important and difficult choice. And the second method is called *hierarchical clustering*, in which we don't pre-specify the number of clusters $$K$$. But rather, we group the data in all numbers of clusters. And it's done in a hierarchical agglomerative fashion. And this is nice because we get to see the clustering for all numbers of clusters $$K$$. 

##### Slide 27: 

*Tibshirani:* But let's start with the simplest method-- K-means clustering. Before we describe it, lets see an example of the result of K-means clustering.

<center>
<img src=Images/10.7.PNG alt="K Means Clustering">
</center>
{:refdef: style="text-align:center"}
**Figure 10.7**
{: refdef}

So this is some simulated data. And it's been simulated basically in two groups, which are the upper group. There's two features. And there's a group at the top, some space in between, and a group at the bottom. And now, we apply K-means clustering. And we have to pre-specify $$K$$. And we'll see the procedure in a few slides. But when we pre-specify $$K$$ equals 2 and we run the clustering out of it, it produces the groups indicated by the two colors blue and gold. With $$K$$ equals 2, it has found approximately the right clusters. Although you might argue, well, is this point actually along in the upper cluster rather than the bottom cluster? And that's not something which you can answer just in a quantitative way. That's a subjective call. But in any case, it seems to have found the two clusters pretty correctly. But if I had specified $$K$$ equals 3, it would be forced to find three groups. And the three groups K-means clustering found are indicated here by the green, the blue, and the gold. So what it has done is it's broken up this apparently homogeneous cluster into two clusters. Similarly with $$K$$ equals 4, it finds the blue, the orange, the purple, and the green. So it's broken up this bottom cluster it looks like into three clusters. Although actually it's borrowed some points from the top right there. So you can see that the effect of K-means clustering-- well, first of all, you can see the effect of $$K$$ is really important, because if you choose $$K$$ be too large, it's going to be forced to break up groups like this one, which are fairly homogeneous. 

*Hastie:* What's also interesting, Rob, is it's variables that are responsible for clusters, like, for example, the second variable that we have over here, also tend to have a high variance, because if they separate the data in clusters, there tends to be variance. So there's some connection between principal components and clustering in a more abstract sense. 

##### Slide 28: 

*Tibshirani:* OK. So let's actually drill down in the details of K-means clustering. Well, first of all, we have to define some notation for clusters or sets. So we'll call them $$C_{1}$$ through $$C_{K}$$. And they're sets of indices of the observations. So the indices are 1 through $$n$$. Those are our observations. Each of these $$C$$'s is going to be a subset of 1 through $$n$$. And the subset is going to be a partition of 1 through $$n$$. Now, as we want to get formalistic, we'll say, well, the $$C$$'s, their union, is 1 through $$n$$. So if we can count them all together, they make up 1 through $$n$$. So they're a partition of 1 through $$n$$. And there's no intersection. So there's no overlap between the clusters. So this is just a fancy way of saying we're going to break up the points 1 through $$n$$ and do $$K$$ groups, which are not overlapping, and cover the whole set. OK? And again, if the $$i^{\text{th}}$$ observation is in the $$k^{\text{th}}$$ cluster, then $$i$$ will be a member of the indices for group $$K$$. 

##### Slide 29: 

*Tibshirani:* So again, we want to somehow find the partition $$C_{1}$$ through $$C_{K}$$, which is a good clustering. Well, what do you mean by good clustering? Well, K-means clustering defines good clustering to be one in which the within cluster variation is small. So getting back to this picture, if you asked me, well, divide this into say two groups, well, the notion that K-means is going to use is say, well, I'm going to find two groups so that within each group, like within the gold, the variation is small within the group, so that the points are close together. Similarly for this group, the points are close together. It's a very intuitive definition. So let's call the within-cluster variation of the cluster WCV($$C_{K}$$) for within-cluster variation of $$C_{K}$$. It's the total variation. For example, we can use square distance in two directions. Matter of fact, most the time, we will use squared distance for K-means clustering. So then if we put it all together, if we define the variation within a cluster to be WCV($$C_{K}$$), we want to find-- Well, the total variation adding up overall clusters is here. And we want to find the partitioning, $$C_{1}$$ through $$C_{K}$$, to minimize the total within-cluster variation. So we're going to assign the end data points to $$K$$ clusters so that the total within-cluster variation summed up over the $$K$$ clusters is as small as possible. 

$$\displaystyle\min_{C_1,\dotsc,C_K}\left\{\sum_{k=1}^K\text{WCV}(C_k)\right\}$$

##### Slide 30: 

*Tibshirani:* So I said this in a previous slide, but here is it in detail. We normally define within-cluster variation to be the Euclidean distance, the pair-wise squared distance between each pair of observations in the cluster.

$$\text{WCV}(C_k)=\frac{1}{|C_k|}\displaystyle\sum_{i,i'\in C_k}\sum_{j=1}^p(x_{ij}-x_{i'j})^2$$

Add it up over the $$p$$ feature. So this is in the Euclidean or squared distance between observations $$i$$ and $$i'$$. And we sum it up over all $$i,i'$$ in the cluster. So this is the total pair-wise between every pair in the clusters $$C_{K}$$. And we're going to minimize the total of this over $$K$$, the total variation over all clusters. And here it is.

$$\displaystyle\sum_{C_1,\dotsc,C_K}\left\{\sum_{k=1}^K\frac{1}{|C_k|}\sum_{i,i'\in C_k}\sum_{j=1}^p(x_{ij}-x_{i'j})^2\right\}$$

So here's our optimization problem now. Here's the within-cluster variation. And we're going to find the clustering, $$C_{1}$$ through $$C_{K}$$, that minimizes that. 

##### Slide 31: 

*Tibshirani:* So we have a criteria. But let's actually talk about the K-means algorithm. And then we'll circle back and see why I minimized the objective on the previous slide. So how K-means clustering works. Well, it's got the word means in it. So it's going to use a mean somehow. So it's actually an alternating algorithm that first of all we assign to each observational cluster from 1 to $$K$$. So remember $$K$$ is fixed. We have to decide ahead of time I'm going to pick $$K$$ equals, for example, 2 or $$K$$ equals 3. And we'll worry later about how to choose $$K$$, an important value. But let's fix $$K$$ for the moment. So each observation is assigned to a cluster 1 through $$K$$. And then we have two steps which we alternate back and forth. For each of the $$K$$ clusters, on the one hand, we compute the centroid. That's the average value for each feature of all the points in the cluster. It's the mean in the vertical and horizontal direction of the points in that cluster. So having computed the centroids for each of the clusters, on the other step, we assign each data point to the closest centroid. And then having done that, we have a new set of cluster assignments, $$C_{1}$$ through $$C_{K}$$. We go back. And we compute new centroids. With these new centroids, we complete new assignments. And we alternate back and forth until hopefully this thing settles down. And the solution is the K-means clustering. The solution we want is the assignment of points to the groups. 

##### Jump to Slide 33:

*Tibshirani:* So let's see an example. And then we'll go back and see why that algorithm actually minimizes the objective that we wrote down. So here's an example, actually the same example we had before. 

<center>
<img src=Images/10.8.PNG alt="K Means Example">
</center>
{:refdef: style="text-align:center"}
**Figure 10.8**
{: refdef}

Here's our data. It's unlabeled data. And we've chosen $$K$$ equals 3. So in the first step, we're going to assign points at random to clusters. So we've indicated these by the colors. So each point is assigned to a different color. And you see the grouping is not very good yet. In other words, we're thinking that this is one group and this is another group. But we're nowhere near that clustering at this point. The first step we compute the centroids. So these are the average of the horizontal and vertical direction of all the points in the gold, green, and blue groups. And here the centroids are pretty much on top of each other, because the assignment was random. So there's no real grouping yet. But don't fear. The K-means clustering will work its way to a good solution. So now, we take the centroids. And we find the closest points to each of the centroids. So at each point we ask, are you closest to the green, the purple, or the orange centroid? And now we color the points accordingly. So this is the partitioning step. So even though those first three centroids were pretty much on top of each other, they weren't exactly. And so that defines a fairly nice partition of the data already. It's an almost perfect job. Well, you'll see we'll get to the final iterate. And then given this new assignment, we find the centroids again. By now, the centroids are going to move a lot now, because look where the points are. So for example, there average of these gold points is way up here. So here's the new gold centroid, new purple, new green. Now, the centroids are sitting really right in the middle of the clusters. And the algorithm we're going to continue and make a few refinements. And here's the final K-means solution for this example. (Very intuitive algorithm.) 

##### Slide 32: 

*Tibshirani:* So let's go back. Remember, we had a subjective function, which was the total within-cluster variation we want to minimize, right? We want to find the partitioning that minimizes this within-cluster variation. Well, the question is does this algorithm that we wrote down, does it achieve that? Well, you can actually see that the algorithm will always decrease the value of the objectively at each step. And you might think about why that is. Well, the key to it really is that you can write the pair-wise variation as the variation around the component-wise means. So this is really the key to K-means clustering. Think about it. We really didn't care about the centroids. All we cared about was the clustering. What K-means clustering has done is it's changed this to an equivalent expression involving the centroid. And now, K-means clustering finds both the component $$C_{1}$$ through $$C_{K}$$ as well as the centroids $$\bar{x}$$. And what it does at each step is it minimizes over $$C_{K}$$ on the one hand or minimizes over the centroid on the other hand. And each of those steps is going to decrease this criterion and hence decrease this criterion. 

*Hastie:* So if you have centroids, then when you come to update the assignment of each point, it's going to go to the group for which this distance is smallest for fixed centroids. And on the other hand, if all the assignments are fixed, we know that the sample mean minimizes the sum of squares. And so that's why each of those steps necessarily makes the criterion go down. 

*Tibshirani:* So this is very slick. We started off with a problem involving one set of variables. We added another set of variables, which we think would make it harder but actually makes it easier, because now when you do the joint optimization over both sets of variables, we get to the answer for the ones we care about, the clustering. (Slide 34) So that's just a detail of the previous figure, which we talked about. Sorry. I missed a point here, which is that this algorithm, although it gives you a local minimum, it's not guaranteed to give a global minimum. Why not? Well-(What does that mean, Rob, a local minimum?) Local minimum means that-- well, the point is that you can start the algorithm-- local minimum, in calculus, it means that the derivative of the function is zero. It doesn't mean that it's the lowest point of the function. So if a function is not convex, you can have a place where the derivative or there's a valley, which is flat, but it's not the lowest valley in the whole function. So this algorithm will get to a local minimum. It will get to a valley of the function we're trying to minimize. But it won't be the lowest value necessarily. 

*Hastie:* So we can think of this function that we're trying to optimize as being like a big valley. But it's got lots of little sub-valleys or little ponds or whatever. And any minimum is one of those. And we can get stuck in one of those. 

*Tibshirani:* Right. So in the optimization world, the idea is the idea of a convex function, which is means basically there's only one valley. So if you find a minimum, it's a global minimum. But this function is not convex. In other words, it can go up and down and have more than one valley. And the K-means algorithm will land you in a valley but not necessarily the lowest valley, because the function is not convex. 

##### Slide 35: 

*Tibshirani:* So actually, here's an example. If this is for the same data, here's an example where we start the algorithm from six different starting configurations.

<center>
<img src=Images/10.9.PNG alt="Different Starting Values">
</center>
{:refdef: style="text-align:center"}
**Figure 10.9**
{: refdef}

Remember, a starting configuration was we assigned each point at random to one of the clusters. And this is actually a good thing to do with K-means clustering is since we're not guaranteed to get the global minimum, we start the algorithm at more than one place. And we just examine the value of the criterion at the end in each case. (Remember starting the algorithm was this random assignment of points to the number of clusters you're using.) Right. So when we start the algorithm from different places, we get actually quite different solutions. Don't worry about the fact that the colors are chosen different. Like these are gold, and these are green. The coloring is arbitrary. But the partitioning is quite different. And typically, we pick the lowest value. It looks like we got three different solutions. Three or four different. Three different solutions I guess. 

*Hastie:* The ones we've colored in red at the top all have exactly the same distance. And so they're actually all the same. The colorings are different. But as Rob said, the coloring is arbitrary. 

*Tibshirani:* So these four panels give us the solution with most. (But the top left and the bottom right are actually different solutions again.) Right. OK. So that's K-means clustering. We'll actually talk a little bit at the end how to choose $$K$$, which will also be an issue for hierarchical clustering, which is the topic of our next segment.

## Hierarchical Clustering 

<center>
<iframe width="560" height="315" src="https://www.youtube.com/embed/Tuuc9Y06tAc?list=PL5-da3qGB5IBC-MneTc9oBZz0C6kNJ-f2" frameborder="0" allowfullscreen></iframe>
</center>

##### Slide 37: 

*Tibshirani:* Welcome back. I had to run out and get a haircut and change my shirt. But we're ready to go. So we were just talking about K-means clustering. And you remember on of the things with K-means clustering, one of the challenges, is that you have to choose the number of clusters, $$K$$, which this can be a problem. And a little later, we'll talk about strategies for choosing $$K$$. But in this segment, I will talk about hierarchical clustering, a different kind of clustering, the second major one that we'll talk about. And it has the advantage it doesn't require the choice of $$K$$. As a matter of fact as we'll see, it gives us the clusterings for all $$K$$, all at once. And we'll talk about what's called agglomerative or bottom-up hierarchical clustering, which is the most common kind of hierarchical clustering. There's also top-down hierarchical clustering you might see, but we won't talk about that in the lecture, because the bottom up or agglomerative is by far the most common. And we'll see that the agglomerative clustering starts from the bottom, the individual observations as leaves, and combines them successively in a hierarchy. 

##### Slide 38: 

*Tibshirani:* So let's see the main idea, just in a picture.

<center>
<img src=Images/10.10.PNG alt="Hierarchical Clustering">
</center>
{:refdef: style="text-align:center"}
**Figure 10.10**
{: refdef}

So here's some data with five objects and two features on the horizontal-vertical axis. And what hierarchical clustering does is it successively joins the closest pair of objects. So I'll first of all run through it quickly, and then I'll go back and look at the operations more slowly. So here's the data. And now let's see what hierarchical clustering does. So there's one-- another clustering, another one, and another one. So I did a total of one, two, three, four joins. Let's go back more slowly now and see what happened. So starting with the raw data, it looks for the closest pair of objects in terms of squared distance. And I'll talk a little bit later. We can actually vary that and use a different metric. But usually it's squared distance. So the closest points in the plane here, the closest pair are A and C. It joins those together. Then it looks for the next closest pair. And that is D and E. And then it's going to join the next closest pair. But now the idea of a pair, we're going to join not necessarily always just single observations. We might join an observation with an existing cluster or maybe join two clusters. And we'll have to decide on what we mean by how far apart are two clusters, which we'll talk about a little later in the segment. But we agree at some choice of that metric, and let's see what happens. It joined B with A and C and then, finally, all five points together in one cluster. (So in this last one, it joined two clusters together, the big, red cluster and the green cluster.) 

##### Slide 39: 

*Tibshirani:* So this next slide summarizes now the clustering.

<center>
<img src=Images/10.11.PNG alt="Hierarchical Algorithm">
</center>
{:refdef: style="text-align:center"}
**Figure 10.11**
{: refdef}

Again, you see here on the left the series of joins that we just saw. And now, on the right, is a dendrogram or clustering tree, which is a very nice way of summarizing the clustering. The picture on the left is fine when you have two features. But imagine if you had a hundred features. We'd have to be drawing a hundred dimensional object, which you can't do. The clustering tree on the right is a most useful way of summarizing the clustering. So let's see how it summarizes things. Well first of all, at the bottom, we have the leaves of the tree, the objects, labeled A through E. The first join is indicated here. This is the join of A and C. And the height of the join, this height of this join, is drawn at the height which is distance between the two points involved. So A and C, their squared distance is here. And that's the height at which this is drawn. The next join is D and E, then B with A and C, and finally all the objects together at the top. So the clustering tree gives us a nice summary of the series of joins that we get in the clustering. (But what's it mean, a distance between a point and a cluster?) Well, we'll get to that. Or do you want me to tell you right now? (No, no.) OK. 

##### Slide 40: 

*Tibshirani:* So here is another example. 

<center>
<img src=Images/10.12.PNG alt="Hierarchical Example">
</center>
{:refdef: style="text-align:center"}
**Figure 10.12**
{: refdef}

There's 45 observations, again, in two dimensions. And in reality here, we imagine there's three classes. So I've colored them as such-- green, purple, and gold. But we're not going to use these class labels in the clustering. We want to see, after we do the clustering, where these points end up. 

##### Slide 41: 

*Tibshirani:* So here is the result of hierarchical clustering of the 45 points. 

<center>
<img src=Images/10.13.PNG alt="Hierarchical Example 2">
</center>
{:refdef: style="text-align:center"}
**Figure 10.13**
{: refdef}

And these are all the same dendrogram. But what I've done here is I've cut off the clustering tree at different heights. On the left, I've basically cut it off at the top, so the entire tree is one cluster. That's why I've labeled all the leaves by the same color, green. But here, I've cut the clustering tree off at this height of about 9. It creates two clusters, which I've indicated by red and green. And then finally, here, I've chosen a different cutting height at around 5. It forms three clusters-- the purple, the gold, and the green. And you see the original three classes which I colored actually coincide with these three clusters. So the point is that, by doing the hierarchical clustering and cutting the tree off at an appropriate height, we recover exactly the three classes started with. You also see that, now, what this is showing us is that, with a single operation, giving us this clustering tree, we get a clustering for all $$K$$. If we take the same tree, if we cut it off at the top, we get $$K$$ equals 1. We cut it off at 9. We get $$K$$ equals 2. We cut it off here. We get $$K$$ equals 3. I could cut lower and get more clusters. I could cut it right at the bottom and get a cluster which had a cluster for each observation, which probably wouldn't be very useful. But the point is that we get a spectrum from 1 up to $$n$$ clusters, depending on where we cut, from a single clustering tree. 

##### Slide 45: 

*Tibshirani:* So we talked before about how we had to decide how far apart two clusters were or a point from a cluster. Linkage is what that's called. And now let's talk about the details of the choice of linkage. So on this slide, I've got the different linkages-- **complete, single, average, and centroid**-- and their definitions. But I think it would be easier to go to a blank slide, and I'll have my Khan Academy moment and show you just in pictures what these measures are. 

*Tibshirani:* So a complete linkage, let's first of all imagine our two clusters. And this guy's already a cluster, and over here we have another cluster. And now we want to decide how far apart is this cluster of three from this cluster of two. Complete linkage, first of all, the first kind-- where you look at the farthest distance between any pair where I pick one object from here and one from here. So the farthest distance will be this one. This pair is the farthest apart, so we deem these two clusters to be this distance apart. That's complete linkage. So it's a worst case measure. Single linkage is the other one. It's the best case. We find the closest pair. And for that, it looks like it will be this pair. This pair will be the closest. That's single linkage. And then the average linkage is, as you can imagine, we look at all pairs and take the average distance. So we'd look at the distance between this guy and these two. So how many all together, Trevor? There's six distances. The average of those six pairwise distances would give us the average linkage measure. (Yes, six.) Six. He's quick. So these are the first tree measures. The last one's called centroid linkage, which is not used as commonly. (Pity you don't have different colors in this painting.) I actually do have different colors. Centroid linkage, we take the centroid of each cluster first. That's the average, the middle point. OK. So these points are meant to be the average of these three and this one the average of these two. The centroid linkage is how far apart the centroids are, these two. So that's the third measure. The ones I'd use most commonly are complete and average. Single linkage, if you have a look at this in our book, there's some examples of it. Single linkage has a problem that it tends to produce long, stringy clusters. So it tend to join the one point at a time as a cluster. Whereas complete and average tend to produce trees, which, they're more balanced, more symmetric looking. Central linkage is common in genomics. It has some problems, though, with what are called inversions, which we'll won't discuss but you can read about in the book. So there's four choice of measure with complete and average being the most commonly used. Great-- one picture tells it all, yes. But if you want details, the previous slide has all this again in words. 

##### Slide 46: 

*Tibshirani:* So a few more things to talk about-- we talk about the complete, average and single linkage and centroid in the previous slide. And that was to do with how far apart we deem a pair of clusters from each other. But then we still have to decide, what do we mean by far apart? And we've been using squared distance up to now or Euclidean distance, and that's actually the most commonly used measure. But another one that's used quite commonly is correlation, which is the correlation between the profiles, two observations across their features.

*Hastie:* This is often the case when features are actually measurements at different times. So you can think of each measurement for an individual as a series of points over time-- in a time series, like each of these curves-- and you want to see how similar the patterns are in each of those series. So then correlation measure is suitable in that case. 

*Tibshirani:* So here's an example.

<center>
<img src=Images/10.14.PNG alt="Correlation Measure">
</center>
{:refdef: style="text-align:center"}
**Figure 10.14**
{: refdef}

And we can imagine this index, as Trevor said, could be time. We have three observations-- the purple, the gold, and the green. In terms of squared distance, if we look just across the coordinates, the purple and the gold are very close together, and the green's further apart from those two. And here's the correlation if we just consider the pairs of points at each of these indices. But actually, the green is closer to the gold than it is to the purple, because the green and the gold, they go down and up together. So the point is that correlation doesn't worry about the level but rather the shape. So if shape, in your example, in a particular problem, is more important than the actual level, the correlation is a better choice than Euclidean distance. And also you have to decide whether you want to use correlation or absolute correlation. Sometimes it matters whether things are anticorrelated or positively correlated, and other times you just use the absolute value of a correlation. 

*Hastie:* So those are both common-- I suppose this brings up the point that you can actually come up with any distance that suits you, depending on the problem. If there's a particular distance that makes more sense, you could use that.

*Tibshirani:* Exactly. The only thing you need for hierarchical clustering and, actually, for K-means for that matter, is a choice of distance between pairs of observations. Having decided that, then the algorithms just use that as their input. 

##### Slide 48: 

*Tibshirani:* So some practical issues-- a few more things. Well, scaling of the variables matter. And again, that's the case in K-means as well. So this slide actually refers to both K-means clustering and hierarchical clustering. (And principal components, for that matter.) Exactly-- right. And if the variables are not in the same units and one variable is in a unit that has very large numbers, then if you don't standardize the variables, that variable will dominate either principal components or clustering because the units are such that the squared distance for that variable is much larger. 

<center>
<img src=Images/10.15.PNG alt="Scaling Matters">
</center>
{:refdef: style="text-align:center"}
**Figure 10.15**
{: refdef}

So typically, if variable are not in the same units, one should standardize them-- in other words, make the mean 0 and standard deviation to be 1. If they're in the same units, you have a choice. And it's often good to do it both ways. If they're in the same units, you might want to leave the variables as is and see which ones have more variation. And those will tend to drive the principal components or clustering. In other cases, people will standardize in that situation. So it's good to try it both ways. And then in the case of hierarchical clustering, we talked about these already. Should we use squared distance or correlation for dissimilarity? And the linkage-- choice of linkage. A problem I've alerted to-- I'm not really going to say much more here-- is the choice of $$K$$, the number of clusters. And let me just say, we're not going to talk about it because it's a difficult problem with no agreed upon solution. There's not a standard method we can point you to to say, this is how you choose $$K$$. In our data mining book, The Elements of Statistical Learning, in chapter 13, there's much more detail about this. But typically, the problem, again, is not well solved. And it's usually done in a qualitative way. In a hierarchical clustering, people just tend to look at the result of the clustering tree, see where the biggest drop in the branches is, and they would maybe cut it off there. (Jump to Slide 41) Remember the example? And go back to the clustering-- in this example, if this was actually some data you had, you might look at this and say, it looks like the longest branches are here. Here's where the longest arms are. I'll cut it off here. And then below that, the arms are much shorter. And remember, that's reasonable because the height at which we draw these joins is the squared distance at which the joins occur. So the biggest bang for the buck is occurring in these joints, and then there's less lower down. So that's subjective, but that's typically what people do. They just have a look at this and say, I'm going to cut it off, maybe, at three clusters. (Back to Slide 46) And which features should we use to drive the clustering? You're often given, in some areas of science, a large number of features, and you want to use those features for clustering some objects. You can choose the features you use. And if you change the choice of features, as you can imagine, it's going to change the clustering that you get. That's the end of the segment on hierarchical clustering. In the last segment, we'll see a real example of the application of hierarchical clustering to a study of breast cancer.

## Breast Cancer Example of Hierarchical Clustering 

<center>
<iframe width="560" height="315" src="https://www.youtube.com/embed/yUJcTpWNY_o?list=PL5-da3qGB5IBC-MneTc9oBZz0C6kNJ-f2" frameborder="0" allowfullscreen></iframe>
</center>

##### Slide 49: 

*Tibshirani:* Welcome back. In the last segment of this section, we're going to see an example of hierarchical clustering applied to a study of breast cancer. So this is the last part of the segment. This is an example which actually Trevor and I are both involved in actually 10 years ago now with a post-doc at Stanford in oncology, Therese Sorlie. Therese had measured gene expression from gene chips for about 88 women who had breast cancer, were being treated for breast cancer, and gene expression measurements for about 8,000 genes. So what that means is for each of the 88 patients, there's a quantitative measurement for each of 8,000 genes, which measures how much that gene was expressing, how active it was for that woman. And this is a very common kind of study now where people look at gene expression and try to understand the basis of diseases like breast cancer and figure out whether there are subtypes of the disease which should be treated in a different way. So this is quite large, the amount of data. 88 patients, 8,000 features. The group used average linkage with correlation metric. Again, because it's the case where genes are in the same units in a sense, you're measuring the same units, but the actual level of gene expression wasn't very reliable because it varies across the way it's measured. But what was thought to be more important was the relative expression of different genes for the same patient, so that's why we use correlation metric. And we did hierarchical clustering of the samples, of the 88 patients. Now, when Therese first used the full set of genes, the clustering she got out wasn't satisfactory. Now, what does that mean? Well, it's very subjective, but it wasn't very informative to Therese and her collaborators. So rather than use a subset of the genes, called the intrinsic genes-- this was a way of choosing a more informative subset of genes. And I won't go into detail, except to say the words, in this particular study, these women were given chemotherapy. And there was actually a sample taken before and after for each woman. And gene expression measurements were available before and after. So what Therese did was she defined what called intrinsic genes. So each woman, each gene for each woman, we looked to see which genes had the smallest variation within a woman as opposed to between the 88 women. And the ones with the smallest variation were defined to be intrinsic genes, the 500 genes with the lowest variation. The idea being-- again, this is a biological concept-- was that genes which didn't vary much in a woman before and after chemotherapy, compared to the between-women variation, were thought to be intrinsic to her cell biology. So they are thought to be the ones that could best derive the clustering and separate the women in terms of their biology and maybe their response to treatment. (They varied a lot between women, but little within woman, across the two repeated measures.) 

##### Slide 50: 

*Tibshirani:* So doing that, we got the following clustering.

<center>
<img src=Images/10.16.PNG alt="Breast Cancer Clustering">
</center>
{:refdef: style="text-align:center"}
**Figure 10.16**
{: refdef}

So what do we see here? First of all, here are the 500 or so intrinsic genes. And this is called a heat map. And this is a common display for this kind of data. So what do you see here? Each row of the heat map is a gene. 500 or so genes. Each column is a woman, one of the 88 women. And each pixel is displayed as either green, which is negative-- so gene expression is normalized. So it runs from something like minus five to plus five. So green would be negative, and red is positive. So green means the gene expression for that gene for that woman is lower than average, and red means it's higher than average. And what's been done here is we plot hierarchical clustering to the columns-- that's the women, in the way I just described. In addition, hierarchical clustering was done to the roads, the genes. It was done in both directions. And that's why this picture looks-- it has patches of red and green. Because we've sorted them, basically. In hierarchical clustering, we sorted the observations by the order of the leaves in the tree, both for genes, and for samples. And that's why if we just displayed the data in the order we obtained it, this picture would not look so nice. It would be a checkerboard pattern. It would look very random. Here it looks much more structured because of the clustering's been applied, and the ordering of the leaves is what's been used to reorder the rows and columns. Actually, this kind of display was actually used first at Stanford in the genomics labs around the time the gene chip was invented, which was also done partly here at Stanford. And I think this has become very attractive, just because it's a nice way just to see all the data. If you're given a dataset of 88 observations, women, and 8,000 genes, that's a lot of data just to even look at. And so the first kind of challenge is, how do I just make a display so I can look at all the data and see the gross patterns? And here, it's actually a very effective display. (This is one of the pioneering efforts in the labs of Patrick Brown and David Botstein, where gene expression really started.) So it's kind of sort of funny to think that a pioneering piece of science is actually display, but that's often the case. Some things, some very simple things, which might seem trivial actually have a lot of impact. In this case, just the ability to look at, to arrange and display the data informatively was very useful, and still used a lot today. So here's the full heat map. And then the clustering tree is at the top. This is here, and it's been expanded out here. And it's been divided into one, two, three, four, five, six, seven, eight clusters. The gray is just basically an unknown group. But the other clusters have been labeled by names like normal, basal, ERBB2+, luminal A, luminal B. These names were chosen by Therese and collaborators based on the genes that were expressing in the groups. So now if we look at this picture, what we've taken is the same clusters, and we've just taken subsets of the rows-- that's these five groups-- and these are genes which are expressing highly in one or more of these key groups. Like for example here. This block of c genes is expressing highly in the red group and the blue group. This block of d genes expresses highly in these clusters, et cetera. So then the oncologists will look at this, and they'll try to understand-- so these groups are different with respect to these particular genes. What do these genes do in the cells, and what does it tell us about these sub-groups? 

##### Slide 51:

*Tibshirani:* In particular, onto the last display of this. You look at these subgroups, and you look at their survival of these women, these are called Kaplan-Meier survival curves.

<center>
<img src=Images/10.17.PNG alt="Breast Cancer Survival Curves">
</center>
{:refdef: style="text-align:center"}
**Figure 10.17**-- *Blue: Luminal A, Cyan: Luminal B, Pink: ERBB2+, Red: Basal*
{: refdef}

These women were treated for cancer and followed up to see how they hopefully recovered. Some didn't. And the survival curves of the groups are given here. So for example, the basal group-- I believe the red and the pink-- which groups are those? Basal and ERBB2+ are doing not nearly as well. Their probability of survival is much worse, whereas this group, luminal A, is doing much better. So because their survival's quite different, the scientists really wanted to find out, how are these groups different, and with respect to what genes? And that gives us a clue as to how the diseases might be different in the different groups. So that's example of clustering in a real scientific problem that's of importance.

##### Slide 52: 

*Tibshirani:* So just to wrap up this section now, unsupervised learning was what was the topic. We talked about principal components and clustering. And the important general for understanding the variation in grouping structure, aside of unlabeled data. So they could be useful just by themselves, as we saw, for example, in that last example, or as a preprocessor to choose a linear combination of features for supervised learning. And we also saw that the problem is intrinsically harder than supervised learning because there's no label. There's no gold standard. So you can't use prediction error to figure out how well we're doing. 

*Hastie:* We've just shown you two techniques in these presentations, principal components and clustering, and those are part of a big tool bag of lots of other techniques. Some of them are listed here, like self-organizing maps, independent component analysis, spectral clustering, and many more. Many more of these are covered in our book, Elements of Statistical Learning, in chapter 14. And even beyond that, there are many others as well.
