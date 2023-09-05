# ERA Metric Pair Evaluation

System for calculating selection criteria for available metric pairs for conducting Expressive Range Analysis (ERA). This system implements the approach that formed the basis oc [PAPER LINK] that will be presented at the Foundations of Digital Games conference this April (http://fdg2023.org/). 

System processed phenotypic metrics for sets of Super Mario levels, calculated using the system found at http://github.com/KrellFace/mario_metric_extraction, and based on the Mario AI Framework: http://github.com/amidos2006/Mario-AI-Framework. 

The selection criteria calculated are:

* Mutual Correlation: Level of correlation between the values of a pair of metrics, calculated using Spearman's Rho
* Alternative Metric Corelation: Average level of best correlation found between metrics in pair and all other metrics, calculated using average of best Spearman's Rho for the metric pair
* Average Fitness: Average fitness across a grid defined by the values of the metric pairs

This system is a proof of concept of the metric selection criteria approach.

## Basic Operation

main_MetricEval.py is the class for conducting an evaluation run. The file to process is specified at 'input_to_use', and there are a choice of two files to use. One with the full evaluation data on all levels in the Mario AI Framework, and one with only 10 levels from each generator.

'out_folder' specifies where the output will be stored. The output here are .csvs of the selectrion criteria values for every metric pair, and a .csv of their relative ranks. Fitness heatmaps for every metric pair are also produced.

buck_cnt specifies the granularity of the grid that will be used for calculating average fitness, as well as generating the heatmaps.

main_GenERAScatter.py is a standalone class for producing scatterplots for every level in a specified file, with their location specified with the values they have for specified metrics with variables 'metric1' and 'metric2'.

