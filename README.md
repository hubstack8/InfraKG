# InfraKG: Infrastructure Knowledge Graph
## 1. Neo4j Knowledge Graph Setup

This repository contains a Docker-based setup for running Neo4j with a pre-existing knowledge graph dump.

### Prerequisites

- Docker
- Docker Compose
- Git

### Project Structure

```
.
├── data/                 # Neo4j data directory (mounted as volume)
├── dump/                 # Contains the knowledge graph dump file
│   └── your-kg.dump     # Neo4j database dump (uncompressed)
├── imports/              # Import directory for Neo4j
├── plugins/              # Neo4j plugins directory
├── docker-compose.yml    # Docker Compose configuration
├── load-dump.sh         # Script to load the dump into Neo4j
└── README.md            # This file
```

### Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/hubstack8/InfraKG.git
cd InfraKG
```

### 2. Make the Script Executable

```bash
chmod +x load-dump.sh
```

### 3. Run the Setup

Execute the load script to start Neo4j and load your knowledge graph:

```bash
./load-dump.sh
```

The script will:
- Start the Neo4j container using Docker Compose
- Wait for Neo4j to initialize
- Load the dump file into the Neo4j database
- Restart the container to ensure proper loading

### 4. Access Neo4j

Once the setup is complete, you can access Neo4j through:

- **Neo4j Browser**: http://localhost:7474
- **Bolt Protocol**: bolt://localhost:7687

### 5. Login Credentials

- **Username**: `neo4j`
- **Password**: `test1234`

## Configuration Details

### Docker Compose Services

- **Neo4j Version**: 5.1.0
- **Container Name**: kg-neo4j
- **Ports**:
  - 7474: Neo4j Browser interface
  - 7687: Bolt protocol for database connections
- **Volumes**:
  - `./data` → `/data` (database files)
  - `./imports` → `/import` (import directory)
  - `./plugins` → `/plugins` (plugins directory)






# 2. Interactive Dashboard for Further Infrastructure Analysis (Streamlit)
To support large-scale analysis of computational infrastructure usage in scientific research, we developed an interactive web-based dashboard using Streamlit. The dashboard provides visual analytics across multiple dimensions, including hardware, software, cloud platforms, collaboration networks, and research topics. It enables exploration of entity-level trends such as GPU and memory configurations, cloud service adoption, institutional collaborations, and topic-wise usage patterns, facilitating a comprehensive understanding of infrastructure usage in the research ecosystem. 

Here the URL: [https://app-data-analysis-hfphssdxfnqy8dzvugamzs.streamlit.app/](https://app-data-analysis-6vuaylmux3jbbvyb6tgmcx.streamlit.app/)
#### 1. Geographic Analysis
#### 2. Organizational Insights
#### 3. Publication Metrics
#### 4. Cloud Platform Analytics
#### 5. Hardware Analysis
#### 6. Collaboration Networks
#### 7. Software Ecosystem
#### 8. Research Topics
For the research topic analysis, we applied Latent Dirichlet Allocation (LDA) to a corpus of 85,000 publications. We used the titles and abstracts of the papers as input for topic modeling. Prior to applying LDA, we used the Gensim library and coherence scores to determine the optimal number of topics.

**Note:** If any figure appears distorted or does not fit well on the screen, users can adjust the display size using predefined options (e.g., Compact, Standard) to improve readability and layout.
