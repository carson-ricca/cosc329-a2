# COSC-329: Assignment #2

### Run the Assignment:

- Ensure you are in the root of the repository.
- Run the assignment using `python a2.py`.
- After running the assignment the project structure will resemble the structure outlined below.

### Structure/Output:

- `data`
    - `cosine_similarity`
        - This directory contains the cosine similarity matrix as a csv file.
    - `dendrogram`
        - This directory contains the dendrogram image as a png.
        - The dendrogram points are as follows:
            - 0-13 represents k1-k14.
            - 14-27 represents t1-t14.
    - `processed`
        - This directory contains text files of the data after going through preprocessing.
    - `raw`
        - This directory contains the raw data files.
    - `tf_idf_output`
        - This directory contains the document vector for each file as a csv.
    - `stop_words.txt`
        - This file contains the stop_words used in preprocessing.

### Dendrogram Analysis

- It appears that similar to the example diagram in the assignment description, my data also split into two clusters.
  Although in my case it was point 17 (t3), that ended up in its own cluster.
- When looking at the dendrogram we can also see that within the larger cluster, there is a pretty clear separation
  between two groups. Based on the points that are in the two groups it appears that some texts were successfully
  clustered to others within the same category.
  - Unfortunately, it doesn't look like the clustering did a good job of separating the two types of files we had in the beginning.
  - Running stemming more than once may help to increase the effectiveness of the clustering algorithm.
  - Removing terms in the document vector that have a very low score may also help improve the clustering.