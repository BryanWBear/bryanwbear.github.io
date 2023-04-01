---
layout: default
title:  "Parsing Anime Reviews with GPT-3"
date:   2023-03-29 23:59:08 -0700
categories: nlp
description: "Using GPT-3 to Understand Why Anime Fans are Hurting"
---

## Introduction

I recently finished watching Shaman King, a Netflix anime that I randomly stumbled upon. Although I wouldn't call it high art, the anime had me crying in the bathroom before school at times, and laughing out loud at others. Some parts are so bad that I felt like they were satirical, but overall I enjoyed the series. I mean, tell me if you've seen this line in any other anime:

![My image Name](/assets/images/shaman.jpg)

Afterwards, I decided to go onto Google to see what other people had to say about the show. I was surprised to see that many people hated it. Take this person's review for example:

```
I tried but I can’t even get passed episode one. Ik it’s a reboot but it’s just soo bad. For starters the old shaman king theme song is ten times better then the new one and anyone who watched the old series will tell u. Second everything feels rush and fast pace. Third the voice actors for the voices  don’t go with the characters at all. Other then that the detail looks amazing but cuz of those three things that I stated I can’t continue to watch it.  Tragic cuz shaman king was my first anime show and I was excited that it was coming on Netflix.But now it’s just a waste 😞😫😩
```

Apparently the Netflix version of Shaman King is a reboot of the original 2001 series (which I have never heard of before this). Many people are upset because they think that the Shaman King from their childhood was way better. However, I also noticed that people who watched the Netflix version without knowledge of original series tended to enjoy the anime a lot:

```
no way y’all are rating this gorgeous anime below 5 stars. plot, amazing. characters, amazing. development, amazing. yall are haters because it’s featured on Netflix but despite your opinions and how yall are simps over one piece, the anime is a whole 10/10. i am genuinely disappointed how y’all are gonna rate this anime lower than it should be. 

note: i didn’t watch the original so i can’t base it off that.
```

<br>
<br>

I felt like this was a good opportunity to learn a bit of GPT-3 prompt engineering. Specifically, we'll be focusing on 2 tasks in this notebook:

1. Filtering out reviews that are potentially biased by nostalgia.
2. Clustering reviews and understanding each cluster.

<br>
<br>

## Setup

First, we need to install the required dependencies:

<br>
<br>


```python
%%capture # suppresses notebook output, delete if you need to debug installs.

!pip install --upgrade openai
!pip install python-dotenv
!pip install tiktoken
```

<br>
<br>

Next, if you haven't done so already, you need to go to https://platform.openai.com/account/api-keys and create an API key for openai. This will allow us to create embeddings and use GPT-3 sentence completion. You may need to link a credit card, or activate a free trial, which gives you 18 dollars of compute credits. 

The API calls used in this article will amount to less than a dollar of usage fees. 

Now you need to create a .env file in the root directory of whatever repository you're running this code in. The .env file only needs to contain the line

```
OPENAI_API_KEY=<YOUR_OPENAI_KEY_HERE_WITH_NO_QUOTES>
```

The next few lines of code load the required libraries and the API key from your .env file. We also specify the models we are using. At this time `text-embedding-ada-002` is by far the best embedding model, and `text-davinci-003` (GPT-3) is the best completion model.

<br>
<br>


```python
import numpy as np
import pandas as pd
import openai
import os 
from dotenv import load_dotenv
import pprint
import tiktoken
import plotly.express as px
from sklearn.decomposition import PCA
import plotly

plotly.offline.init_notebook_mode()

load_dotenv()

EMBEDDING_MODEL = "text-embedding-ada-002"
COMPLETION_MODEL = "text-davinci-003"
openai.api_key = os.getenv('OPENAI_API_KEY')
```

<br>
<br>

## Helper Functions

The following functions get responses from API calls to the embedding model and completion model respectively.

<br>
<br>


```python
def get_embedding(text: list[str], model: str=EMBEDDING_MODEL) -> np.array:
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return np.array([e['embedding'] for e in result['data']])


def send_prompt(prompt, max_tokens=64):
    response = openai.Completion.create(
        engine=COMPLETION_MODEL,
        prompt=prompt,
        temperature=0,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    return response["choices"][0]["text"].replace("\n", "")
```

<br>
<br>

## The Dataset

I collected reviews from Google, Myanimelist, and IMDB. In total, there were only 88 reviews so I just copy pasted them into a word document, then formatted the results into a csv file. Note that Myanimelist and IMDB rate from 1-10, while Google does 1-5, so I converted the 1-10 scale stuff to 1-5 using the following formula: `floor(rating / 2)`.

Since the dataset is so small, I could just read all the reviews and manually categorize them myself. However, GPT-3 helps me do this faster. In addition, the techniques that we use can be extended to a bigger dataset where it's not feasible to read all the reviews manually.

Here's what the dataset looks like. I already ran each review text through the embedding api, and appended the results as columns in the dataframe. Each embedding vector is of dimension 1536, so there are 1536 embedding columns:

<br>
<br>


```python
df = pd.read_csv('../data/shaman_king_reviews.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>reviews</th>
      <th>ratings</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>...</th>
      <th>1526</th>
      <th>1527</th>
      <th>1528</th>
      <th>1529</th>
      <th>1530</th>
      <th>1531</th>
      <th>1532</th>
      <th>1533</th>
      <th>1534</th>
      <th>1535</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>If you want an honest review of this show and ...</td>
      <td>4</td>
      <td>0.010024</td>
      <td>-0.018111</td>
      <td>-0.004465</td>
      <td>-0.015317</td>
      <td>0.008159</td>
      <td>-0.008818</td>
      <td>-0.001967</td>
      <td>-0.062584</td>
      <td>...</td>
      <td>0.002294</td>
      <td>0.010064</td>
      <td>0.030238</td>
      <td>-0.027022</td>
      <td>-0.003921</td>
      <td>0.019113</td>
      <td>-0.008785</td>
      <td>0.011283</td>
      <td>0.002458</td>
      <td>-0.014631</td>
    </tr>
    <tr>
      <th>1</th>
      <td>I (and my brothers) grew up loving the origina...</td>
      <td>3</td>
      <td>-0.006499</td>
      <td>-0.040279</td>
      <td>-0.008809</td>
      <td>-0.021905</td>
      <td>0.006084</td>
      <td>0.014205</td>
      <td>0.008682</td>
      <td>-0.041341</td>
      <td>...</td>
      <td>0.010833</td>
      <td>-0.013528</td>
      <td>-0.009406</td>
      <td>-0.013807</td>
      <td>-0.024906</td>
      <td>0.027070</td>
      <td>-0.020936</td>
      <td>0.007295</td>
      <td>-0.002458</td>
      <td>-0.027030</td>
    </tr>
    <tr>
      <th>2</th>
      <td>I can’t lie this has to be one of the most dis...</td>
      <td>1</td>
      <td>0.004583</td>
      <td>-0.018756</td>
      <td>-0.006495</td>
      <td>-0.005203</td>
      <td>0.010277</td>
      <td>0.010461</td>
      <td>0.000595</td>
      <td>-0.041682</td>
      <td>...</td>
      <td>0.012888</td>
      <td>0.005033</td>
      <td>0.018374</td>
      <td>-0.035330</td>
      <td>-0.010754</td>
      <td>0.031732</td>
      <td>-0.004096</td>
      <td>-0.012233</td>
      <td>-0.004154</td>
      <td>-0.010720</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Enjoyed watching season 1. Brought back alot o...</td>
      <td>4</td>
      <td>0.005704</td>
      <td>-0.034056</td>
      <td>-0.008109</td>
      <td>-0.025170</td>
      <td>-0.005753</td>
      <td>0.012702</td>
      <td>0.014689</td>
      <td>-0.038003</td>
      <td>...</td>
      <td>0.023693</td>
      <td>0.004812</td>
      <td>0.008592</td>
      <td>-0.022791</td>
      <td>-0.015839</td>
      <td>0.006789</td>
      <td>-0.013800</td>
      <td>-0.020165</td>
      <td>-0.013696</td>
      <td>-0.006145</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Based on the first 10 episodes, I would defini...</td>
      <td>3</td>
      <td>0.011579</td>
      <td>-0.037938</td>
      <td>0.007227</td>
      <td>-0.009057</td>
      <td>0.007973</td>
      <td>-0.010501</td>
      <td>-0.014874</td>
      <td>-0.055266</td>
      <td>...</td>
      <td>0.021315</td>
      <td>0.006688</td>
      <td>0.037585</td>
      <td>-0.023972</td>
      <td>-0.002610</td>
      <td>0.015227</td>
      <td>-0.004478</td>
      <td>0.000454</td>
      <td>-0.002464</td>
      <td>-0.011254</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 1538 columns</p>
</div>



<br>
<br>

## Filtering Reviews

First, let's see if we can filter out nostalgia-biased reviews by searching for whether people mention the original series being better. We can frame this to GPT-3 as a classification problem as follows:

<br>
<br>


```python
def get_filter_prompt(review):
    return f'Say "yes" if the following Shaman king review mentions that the original series is better. \
        Say "no" the review does not mention the original series, and "unsure" if there is not enough information. \n\nShaman King Review:\n"""\n{review}\n"""'

cond = df.reviews.apply(get_filter_prompt).apply(send_prompt)
```

<br>
<br>

Let's take a look at the distribution of predicted classes:

<br>
<br>


```python
cond.value_counts()
```




    Unsure    38
    No        25
    Yes       11
    No.        7
    unsure     7
    Name: reviews, dtype: int64



<br>
<br>

Sometimes GPT-3 will predict either `No` or `No.`, which is kind of annoying but not a big deal. If we were being more thorough, we could now read through the "Unsure / unsure" category to see if there is more signal for nostalgia bias that we can glean, then write another classification prompt. However, for now let's just look at the reviews marked "Yes".

<br>
<br>


```python
pp = pprint.PrettyPrinter(width=180, compact=True)

for i in range(4):
    print(f'Review {i}: ')

    sample_review = df[cond == 'Yes'].reviews.values[i]
    pp.pprint(sample_review)
    print('------------------')
```

    Review 0: 
    ('I (and my brothers) grew up loving the original series (2001 version) and we still love to watch the show (20 years later! it was/is that good). I watched the English Dub '
     'version- so the song, the acting and the humour was very high level and justice to the characters and story. The show references and jokes still get used in our daily '
     'lives...... the original was THAT GOOD! I have watched the new version and was excited to hear that alot of the original actors are a part of the remake and do a great job of '
     'performing these characters. However, the voice actors for Yoh, Len(Ren) and Zeke (Hao) are female and they do a great job at performing these characters. But when my ears hear '
     'them as females and eyes see male characters, my mind is unable to accept and there is a lingering annoyance throughout watching this. So it takes away from the greatness of '
     'the show. My brothers also felt the same so it cant just be me left with this annoyance. The remake is more fast paced which suits the modern day viewer who has so much content '
     'to choose these days. It is more gory/dark and there are name changes (minor) and plot changes (minor) in comparison. No new intro in English (minor). The remake is worth '
     'watching, new viewers wont have the original to compare it to. So far only seen the first 13 episodes and will be watching the rest when it comes out. I have marked this down '
     'because i have watched the original which is slower and so you get time to like the the characters. Also the mental breakdown with the voices conflicting with the visual. Hope '
     'that male voice actors can be roped in for the future episodes. The original is still better! they should have completed the original!')
    ------------------
    Review 1: 
    ('I tried but I can’t even get passed episode one. Ik it’s a reboot but it’s just soo bad. For starters the old shaman king theme song is ten times better then the new one and '
     'anyone who watched the old series will tell u. Second everything feels rush and fast pace. Third the voice actors for the voices  don’t go with the characters at all. Other '
     'then that the detail looks amazing but cuz of those three things that I stated I can’t continue to watch it.  Tragic cuz shaman king was my first anime show and I was excited '
     'that it was coming on Netflix.But now it’s just a waste 😞😫😩')
    ------------------
    Review 2: 
    ("Utter garbage. 1- The original theme song was better 2- The story line's been screwed up 3- The character personalities have changed 4- Voices aren't the same 5- They revealed "
     "Zeke to be Yoh's brother in EP-1. WTH. 6- Cutting corners. If they were going to do a remake, they should've done it properly.")
    ------------------
    Review 3: 
    ('Bruh I used to love the old anime so much more. It saddens me that this newer version is a lot faster and the voice acting is subpar (both dub and sub). Sometimes the chosen '
     'background music for the scene is absolute garbage and at this point I want to cry this should definitely be worked on before new watchers talk about this series')
    ------------------


<br>
<br>

Wow, I need some water after tasting all the salt in these reviews. GPT-3 seems to be doing a good job of finding people who think that the original series is better. I read through all the reviews and there is one that it sometimes gets wrong:

```
... It's inevitable that a lot of people will try to compare this to the original anime from 2001 and, while I'm sure that nostalgia is a factor and "
 'a lot of people will prefer the old one based solely on that, I can say that I definitely think this one is better... 
```

 This reviewer is being highly empathetic of the fact that nostalgia bias might be a thing that affects reviews, but GPT-3 mistakes it as thinking that the original series is better. So there are some edge cases that GPT-3 can't handle yet. For the most part though, it seems to be doing a great job.

 After filtering these reviews out, the overall rating of the show increases:

 <br>
<br>


```python
print(f'original ratings: {df.ratings.mean()}')
print(f'ratings with some filtered out nostalgia: {df[cond != "Yes"].ratings.mean()}')
```

    original ratings: 2.8636363636363638
    ratings with some filtered out nostalgia: 3.012987012987013


<br>
<br>

## Clustering

Clustering doesn't have as much of a direct use case as filtering out nostalgia-biased reviews, but I was very curious about how good the GPT-3 embedding vectors are. The following function is useful for checking how long your inputs are. The max token length that the embedding model will take is ~8000.

<br>
<br>


```python
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

df.reviews.apply(lambda review: num_tokens_from_string(review, "cl100k_base")).max()
```




    1562



<br>
<br>

All of our reviews are shorter than the max limit, so there's nothing to worry about. Since I already embedded the reviews beforehand, let's take a PCA and see what happens. The following plot projects the reviews down to 3 dimensions using PCA, and colors each point by its rating.

<br>
<br>


```python
NUMERIC_COLS = [col for col in df.columns if col not in ['reviews', 'ratings']]
```


```python
pca = PCA(n_components=3)
vis_pca = pca.fit_transform(df[NUMERIC_COLS].values)
vis_df_pca = pd.DataFrame({'x': vis_pca[:,0], 'y': vis_pca[:,1], 'z': vis_pca[:,2], 'ratings': df.ratings})


print(f'Amount of variance explained: {pca.explained_variance_ratio_.sum()}')
fig = px.scatter_3d(vis_df_pca, x='x', y='y', z='z', color='ratings')
```

    Amount of variance explained: 0.20827198227505478

{% include shaman_king_1.html %}

<br>
<br>

There's a lot of variation in the data that's not explained by the first 3 principal components. But still, there's a remarkable amount of stratification between good and bad reviews. We notice that good reviews are clumped together, bad reviews are clumped together, and the neutral reviews (3) sit between them. 

Let's see if we can gain a better understanding of the reviews. First, we'll ask GPT-3 to summarize each review as follows:

<br>
<br>


```python
def get_summarize_prompt(review):
    return f'Summarize the following Shaman King review, \
        focusing on the reviewer\'s sentiment, as well as comparisons to the original show if any exist in the review. \n\nShaman King Review:\n"""\n{review}\n"""'
```


```python
summarize_prompts = df.reviews.apply(get_summarize_prompt)
```


```python
summarized_reviews = summarize_prompts.apply(send_prompt)
pp.pprint(summarized_reviews.iloc[0])
```

    ('The reviewer has mixed feelings about the new Shaman King series, noting that the animation is impressive but the pacing and voice changes make it hard to swallow for returning '
     'viewers. They suggest that the show is good for new viewers, but that the original series was better overall. They rate the voiceover, acting, animation')


<br>
<br>

Now we get the embeddings.

<br>
<br>


```python
summarized_embeddings = get_embedding(list(summarized_reviews.values), EMBEDDING_MODEL)
```

<br>
<br>

Now we do PCA on the summarized reviews. We can see that there's even better stratification than before, and the amount of explained variance is a little more than before. This makes sense because GPT-3 summarizes the reviews using similar prose and words. 

<br>
<br>


```python
from sklearn.decomposition import PCA

pca = PCA(n_components=3)
vis_pca_sum = pca.fit_transform(summarized_embeddings)
vis_df_pca_sum = pd.DataFrame({'x': vis_pca_sum[:,0], 'y': vis_pca_sum[:,1], 'z': vis_pca_sum[:,2], 'ratings': df.ratings})
print(f'Amount of variance explained: {pca.explained_variance_ratio_.sum()}')

fig = px.scatter_3d(vis_df_pca_sum, x='x', y='y', z='z', color='ratings')
```

    Amount of variance explained: 0.24312936506302338

{% include shaman_king_2.html %}

<br>
<br>

Now let's summarize the summarized reviews. The reason why we couldn't do this with the original reviews was because of the token length. But as we can see, the summarized reviews are much shorter: 

<br>
<br>


```python
# the summarized reviews are much shorter

vis_df_pca_sum['sum_reviews'] = summarized_reviews
summarized_reviews.apply(lambda string: num_tokens_from_string(string, "cl100k_base")).max()
```




    68



<br>
<br>

Let's divide the summarized reviews into two groups, $x < 0.02$ and $x \geq 0.02$, then summarize the summaries using GPT-3. This sounds kind of stupid, but actually works:

<br>
<br>


```python
vis_df_pca_sum['sum_reviews'] = summarized_reviews
```


```python
negatives = "\n".join(vis_df_pca_sum[vis_df_pca_sum['x'] < -0.02]['sum_reviews'].values)

prompt = f'Summarize the following Shaman King review summaries. \n\nShaman King review summaries:\n"""\n{negatives}\n"""'
response = send_prompt(prompt, max_tokens=240)

print("response: ")
pp.pprint(response)
```

    response: 
    ('The reviewer has a negative sentiment towards the remake of Shaman King, comparing it unfavorably to the original series. They criticize the rushed pacing, lack of important '
     'dialogue, and childish art style, as well as the cringy voice acting. They express disappointment that the manga classic was not faithfully adapted, and suggest viewers watch '
     'the original instead. They also mention that some of the more gruesome aspects and moments that made the show special have been cut, which they feel detracts from the overall '
     'experience. They criticize the decision to combine two episodes into one and skipping important character information, as well as the changes made to the characters, such as '
     'Anna. They also suggest that the show is trying to combine elements of different types of shonen anime, which may be detracting from the original show. They criticize the '
     'soundtrack, action sequences, and storytelling, and suggest that the reboot would have been better if it had been done by Studio Bones. They also criticize the moralistic '
     'vignettes and characters, and find the moral of revenge to be unrealistic and childish. They suggest that viewers watch the original show instead of the remake.')



```python
positives = "\n".join(vis_df_pca_sum[vis_df_pca_sum['x'] >= -0.02]['sum_reviews'].values)

prompt = f'Summarize the following Shaman King review summaries. \n\nShaman King review summaries:\n"""\n{positives}\n"""'
response = send_prompt(prompt, max_tokens=240)

print("response: ")
pp.pprint(response)
```

    response: 
    ('The reviewer has a generally positive sentiment towards the new Shaman King series, noting that it is good for new viewers and that the English dub version is of high quality. '
     'They appreciate the faster pacing and improved fight scenes, but suggest that the relationship between Yoh and Anna should be stronger and that there should be more female main '
     'characters. They also suggest that the original opening song and spirit form transformation soundtrack should have been kept, and that the animation and pacing could be better. '
     'Comparisons to the original show are made, with the reviewer noting that the remake is closer to the manga and that the characters are more badass. They also appreciate the '
     'messages of the show and the exploration of different cultures.')


<br>
<br>

## Conclusion

These summaries generally coincide with what I read from the positive and negative reviews. Both negative and positive reviews commented on the faster pacing and different animation style from the original. However, the positive reviews were okay with overlooking these aspects, while the negative reviews fixated on them. The only contradiction between the summaries is that the negative summary claims that the original was more faithful to the manga, while the positive summary claims that the remake was more faithful to the manga. I don't actually know the answer, since I didn't watch the original, but it shows that GPT-3 is getting some things wrong here.

Overall, even if I had a much bigger dataset and couldn't sift through reviews by hand, I'd be pretty confident in GPT-3 to produce sensible filterings and clusterings.


