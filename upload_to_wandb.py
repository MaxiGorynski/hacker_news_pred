import wandb

# Step 1: Log in and initialize
wandb.login()
run = wandb.init(project="hack-news-predict", job_type="model-upload")

# Step 2: Create artifact and add the file
artifact = wandb.Artifact("hacker-news-model", type="model")
artifact.add_file("best_predictor_model.pth")
artifact.add_file("HN_Corpus_Model_Weights.txt")

# Step 3: Log the artifact
run.log_artifact(artifact)
run.finish()