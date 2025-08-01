# InfraKG





### Interactive Dashboard for Further Infrastructure Analysis
To support large-scale analysis of computational infrastructure usage in scientific research, we developed an interactive web-based dashboard using Streamlit. The dashboard provides visual analytics across multiple dimensions, including hardware, software, cloud platforms, collaboration networks, and research topics. It enables exploration of entity-level trends such as GPU and memory configurations, cloud service adoption, institutional collaborations, and topic-wise usage patterns, facilitating a comprehensive understanding of infrastructure usage in the research ecosystem. 

Here the URL: [https://app-data-analysis-hfphssdxfnqy8dzvugamzs.streamlit.app/](https://app-data-analysis-6vuaylmux3jbbvyb6tgmcx.streamlit.app/)
### 1. Geographic Analysis
### 2. Organizational Insights
### 3. Publication Metrics
### 4. Cloud Platform Analytics
### 5. Hardware Analysis
### 6. Collaboration Networks
### 7. Software Ecosystem
### 8. Research Topics
For the research topic analysis, we applied Latent Dirichlet Allocation (LDA) to a corpus of 85,000 publications. We used the titles and abstracts of the papers as input for topic modeling. Prior to applying LDA, we used the Gensim library and coherence scores to determine the optimal number of topics.

**Note:** If any figure appears distorted or does not fit well on the screen, users can adjust the display size using predefined options (e.g., Compact, Standard) to improve readability and layout.
