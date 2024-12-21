# Cloud HPC Instance Ranking using Multi-Criteria Decision Analysis (MCDA)

This implementation applies a multi-criteria decision analysis (MCDA) framework for evaluating and ranking cloud HPC instances using the Technique for Order Preference by Similarity to Ideal Solution (TOPSIS) method. It integrates various statistical techniques, including sensitivity analysis, bootstrapping, and the Friedman test, to ensure the robustness and reliability of the rankings.

The ranking process evaluates cloud HPC instances from major cloud platforms such as AWS, Google Cloud Platform, Microsoft Azure, and Oracle Cloud Infrastructure, based on key parameters: physical CPU cores, total memory, memory per core, network bandwidth, and on-demand hourly cost. The framework provides a systematic approach to selecting the most suitable cloud HPC instance for high-performance workloads, considering both performance and cost factors.

This implementation allows users to replicate the methodology, perform sensitivity analysis to assess the stability of rankings, apply bootstrapping for uncertainty quantification, and validate results using the Friedman test for statistical significance. It serves as an effective tool for decision-making in cloud HPC instance selection.
