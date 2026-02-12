Stylometric Analysis with Normalized SPI (Tuccinardi Method)
Overview

This project implements a stylometric analysis workflow based on character n-grams and the Tuccinardi SPI (Stylistic Profile Index) similarity measure.

The program is designed to evaluate whether a doubtful text stylistically aligns with a corpus of certain texts attributed to a given author.

The methodology combines:

Character n-gram extraction

L1 normalization

Similarity computation (SPI)

Z-score standardization

Statistical evaluation using either:

Student’s t distribution (small sample size)

Normal distribution (larger sample size)

The output includes numerical results and graphical visualizations to assist interpretation.

Methodological Foundations
1. Character n-grams

The program extracts character n-grams (typically 3-grams or 4-grams) from each text fragment.

3-grams capture general linguistic habits (morphological patterns, orthographic tendencies).

4-grams capture more specific stylistic sequences (recurrent formulae, syntactic patterns).

The choice of n affects sensitivity and variance in the results.

2. L1 Normalization

Each text is represented as a frequency vector of n-grams.

L1 normalization transforms raw counts into proportions:

𝑣
𝑖
′
=
𝑣
𝑖
∑
𝑣
𝑖
v
i
′
	​

=
∑v
i
	​

v
i
	​

	​


This ensures:

Each text is treated as a distribution of stylistic habits.

Text length does not distort similarity.

Comparisons are based on stylistic profile rather than volume.

3. SPI Similarity (Tuccinardi)

The SPI index measures similarity between normalized frequency vectors.

The result is a similarity score for:

Each certain text vs. the rest of the certain corpus

The doubtful text vs. the certain corpus

This produces a distribution of similarity values representing the author’s stylistic baseline.

Statistical Evaluation
1. Z-score Standardization

The similarity of the doubtful text is evaluated relative to the distribution of similarities among the certain texts:

𝑧
=
𝑥
−
𝜇
𝜎
z=
σ
x−μ
	​


Where:

𝑥
x = similarity of doubtful text

𝜇
μ = mean similarity among certain texts

𝜎
σ = standard deviation

The z-score indicates how many standard deviations the doubtful text lies from the authorial norm.

2. Choice of Statistical Distribution

Depending on the number of fragments (n):

Small sample (typically n ≤ 30) → Student’s t distribution

Larger sample → Normal distribution

This adjustment allows proper estimation of statistical significance.

Graphical Output

The program generates visualizations such as:

Distribution plots (histogram + density curve)

Boxplots of similarity values

Z-score distribution plot

These graphs allow visual inspection of:

Central tendency

Dispersion

Outlier position of the doubtful text

Interpretation of Results
Step 1: Evaluate Central Tendency

Examine the mean similarity among certain texts.

A high and compact mean suggests a consistent stylistic profile.

Step 2: Examine Dispersion

The standard deviation reflects stylistic variability within the author’s corpus.

Low dispersion → strong stylistic cohesion

High dispersion → internal stylistic heterogeneity

This directly affects the reliability of classification.

Step 3: Locate the Doubtful Text

Interpret the z-score:

Z-score range	Interpretation
|z| < 1	Fully compatible with authorial norm
1 ≤ |z| < 2	Slight deviation, stylistically plausible
2 ≤ |z| < 3	Significant deviation
|z| ≥ 3	Strong stylistic divergence
Step 4: Consider Genre and Register

A deviation does not automatically imply different authorship.

Differences may reflect:

Genre shift (didactic vs. rhetorical)

Chronological development

Intended audience

Transmission history

Interpretation must therefore combine statistical evidence with philological judgment.

Comparative Use of Different n-grams

Running the analysis with multiple n-gram sizes (e.g., 3 and 4) increases robustness.

If both analyses converge → strong evidence.

If results diverge → possible genre or register variation.

If deviation appears only with larger n-grams → surface stylistic variation rather than deep linguistic divergence.

Intended Use

This tool is designed for:

Authorship attribution studies

Internal consistency analysis within a corpus

Evaluation of doubtful works

Quantitative support for philological hypotheses

It does not replace traditional philological analysis, but provides statistically grounded stylistic evidence to inform it.

Conceptual Summary

The program models each text as a probabilistic distribution of micro-stylistic features.

Authorship compatibility is assessed by measuring whether the doubtful text behaves statistically like the known works of the author.

The central interpretive question is not:

“Is the similarity high?”

but rather:

“Is the doubtful text statistically indistinguishable from the authorial distribution?”