# Customer-Segmentation-using-K-means-Clustering
Today’s business runs based on the ability to captivate the customers with the products, but with such a large raft of products leave the customers puzzled, what to buy and what to not and also the companies are doubtful about what section of customers to target to sell their products. This is where machine learning comes in handy, various algorithms are applied for unraveling the hidden patterns in the data for better decision making for the future. This eludes the concept of which segment to target is made unequivocal by applying segmentation. The process of segmenting the customers with similar behaviors into the same segment and with different patterns into different segments is called customer segmentation. In this paper, 3 different clustering algorithms (k-Means, Agglomerative, and Meanshift) are been implemented to segment the customers and finally compare the results of clusters obtained from the algorithms. A python program has been developed and the program is been trained by applying a standard scaler onto a dataset having two features of 200 training samples taken from the local retail shop. Both the features are the mean of the amount of shopping by customers and average of the customer’s visit into the shop annually.

5.1 LIBRARY AND DEPENDENCY IMPORTATIONS
At first, the required libraries and dependencies were imported for further use in the Jupyter
notebook.
5.1.1 NUMPY
NumPy is the fundamental package for scientific computing in Python. It is a Python library that
provides a multidimensional array object, various derived objects (such as masked arrays and
matrices), and an assortment of routines for fast operations on arrays, including mathematical,
logical, shape manipulation, sorting, selecting, I/O, discrete Fourier transforms, basic linear
algebra, basic statistical operations, random simulation and much more.
At the core of the NumPy package, is the ndarray object. This encapsulates n-dimensional
arrays of homogeneous data types, with many operations being performed in compiled code for
performance.
There are several important differences between NumPy arrays and the standard Python
sequences:
1. NumPy arrays have a fixed size at creation, unlike Python lists (which can grow dynamically). Changing the size of an ndarray will create a new array and delete the original.
2. The elements in a NumPy array are all required to be of the same data type, and thus will
be the same size in memory. The exception: one can have arrays of (Python, including
NumPy) objects, thereby allowing for arrays of different sized elements.
3. NumPy arrays facilitate advanced mathematical and other types of operations on large
numbers of data. Typically, such operations are executed more efficiently and with less
code than is possible using Python’s built-in sequences.
4. A growing plethora of scientific and mathematical Python-based packages are using NumPy
arrays; though these typically support Python-sequence input, they convert such input to
NumPy arrays prior to processing, and they often output NumPy arrays. In other words,
in order to efficiently use much (perhaps even most) of today’s scientific/mathematical
Python-based software, just knowing how to use Python’s built-in sequence types is insufficient - one also needs to know how to use NumPy arrays.
The following code can be written to import the library in the Jupyter Notebook:
import numpy as np
22
5.1.2 PANDAS
Pandas is a software library written for the Python programming language for data manipulation
and analysis. In particular, it offers data structures and operations for manipulating numerical
tables and time series. It is free software released under the three-clause BSD license. The
name is derived from the term "panel data", an econometrics term for data sets that include
observations over multiple time periods for the same individuals. Its name is a play on the
phrase "Python data analysis" itself. Wes McKinney started building what would become pandas
at AQR Capital while he was a researcher there from 2007 to 2010.
Pandas is mainly used for data analysis. Pandas allows importing data from various file formats such as comma-separated values, JSON, SQL, Microsoft Excel. Pandas allows various
data manipulation operations such as merging, reshaping, selecting, as well as data cleaning,
and data wrangling features.
The following are the features of the Pandas Library:
1. DataFrame object for data manipulation with integrated indexing.
2. Tools for reading and writing data between in-memory data structures and different file
formats.
3. Data alignment and integrated handling of missing data.
4. Reshaping and pivoting of data sets.
5. Label-based slicing, fancy indexing, and subsetting of large data sets.
6. Data structure column insertion and deletion.
7. Group by engine allowing split-apply-combine operations on data sets.
8. Data set merging and joining.
9. Hierarchical axis indexing to work with high-dimensional data in a lower-dimensional
data structure.
10. Time series-functionality: Date range generation and frequency conversions, moving window statistics, moving window linear regressions, date shifting and lagging.
11. Provides data filtration.
The library is highly optimized for performance with critical code path written in Cython.
The following code can be written to import the library in the Jupyter Notebook:
import pandas as pd
23
5.1.3 MATPLOTLIB
Matplotlib is a plotting library for the Python programming language and its numerical mathematics extension NumPy. It provides an object-oriented API for embedding plots into applications using general-purpose GUI toolkits like Tkinter, wxPython, Qt, or GTK. There is also
a procedural "pylab" interface based on a state machine (like OpenGL), designed to closely
resemble that of MATLAB, though its use is discouraged. SciPy makes use of Matplotlib.
Matplotlib was originally written by John D. Hunter. Since then it has an active development
community and is distributed under a BSD-style license. Michael Droettboom was nominated as
matplotlib’s lead developer shortly before John Hunter’s death in August 2012 and was further
joined by Thomas Caswell.
Matplotlib 2.0.x supports Python versions 2.7 through 3.10. Python 3 support started with
Matplotlib 1.2. Matplotlib 1.4 is the last version to support Python 2.6. Matplotlib has pledged
not to support Python 2 past 2020 by signing the Python 3 Statement.
matplotlib.pyplot is a plotting library used for 2D graphics in python programming language.
It can be used in python scripts, shell, web application servers and other graphical user interface
toolkits.
Matplotlib is not a part of the Standard Libraries which is installed by default when Python,
there are several toolkits which are available that extend python matplotlib functionality. Some
of them are separate downloads, others can be shipped with the matplotlib source code but have
external dependencies.
One of the greatest benefits of visualization is that it allows us visual access to huge amounts
of data in easily digestible visuals. Matplotlib consists of several plots like line, bar, scatter,
histogram etc.
The following code can be written to import the library in the Jupyter Notebook:
import matplotlib.pyplot as plt
5.1.4 SEABORN
Seaborn is a data visualization library built on top of matplotlib and closely integrated with
pandas data structures in Python. Visualization is the central part of Seaborn which helps in
exploration and understanding of data.
Seaborn offers the following functionalities:
1. Dataset oriented API to determine the relationship between variables.
2. Automatic estimation and plotting of linear regression plots.
3. It supports high-level abstractions for multi-plot grids.
4. Visualizing univariate and bivariate distribution.
24
Using Seaborn we can plot wide varieties of plots like:
1. Distribution Plots
2. Pie Chart Bar Chart
3. Scatter Plots
4. Pair Plots
5. Heat maps
The following code can be written to import the library in the Jupyter Notebook:
import seaborn as sns
5.1.5 SCIKIT LEARN
Scikit-learn (formerly scikits.learn and also known as sklearn) is a free software machine learning library for the Python programming language. It features various classification, regression
and clustering algorithms including support vector machines, random forests, gradient boosting,
k-means and DBSCAN, and is designed to interoperate with the Python numerical and scientific
libraries NumPy and SciPy.
The scikit-learn project started as scikits.learn, a Google Summer of Code project by David
Cournapeau. Its name stems from the notion that it is a "SciKit" (SciPy Toolkit), a separatelydeveloped and distributed third-party extension to SciPy. The original codebase was later rewritten by other developers. In 2010 Fabian Pedregosa, Gael Varoquaux, Alexandre Gramfort and
Vincent Michel, all from the French Institute for Research in Computer Science and Automation in Rocquencourt, France, took leadership of the project and made the first public release on
February the 1st 2010. Of the various scikits, scikit-learn as well as scikit-image were described
as "well-maintained and popular" in November 2012. Scikit-learn is one of the most popular
machine learning libraries on GitHub.
Scikit-learn is largely written in Python, and uses NumPy extensively for high-performance
linear algebra and array operations. Furthermore, some core algorithms are written in Cython to
improve performance. Support vector machines are implemented by a Cython wrapper around
LIBSVM; logistic regression and linear support vector machines by a similar wrapper around
LIBLINEAR. In such cases, extending these methods with Python may not be possible.
Scikit-learn is a free machine learning library for Python. It features various algorithms like
support vector machine, random forests, and k-neighbours, and it also supports Python numerical and scientific libraries like NumPy and SciPy.
The following code can be written to import the library in the Jupyter Notebook:
from sklearn.cluster import KMeans
