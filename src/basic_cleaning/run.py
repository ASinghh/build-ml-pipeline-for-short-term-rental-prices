#!/usr/bin/env python
"""
[An example of a step using MLflow and Weights & Biases]: Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd
import os


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    ######################
    # YOUR CODE HERE     #
    ######################

    #get data file from wandb
    logger.info(f'Downloading artifact from wandb {args.input_artifact}')
    artifact_local_path = run.use_artifact(args.input_artifact).file()  

    # Load Data from the file
    logger.info("reading data from the csv")

    #create pandas dataframe
    df = pd.read_csv(artifact_local_path)

    # Use input args to do data preprocessing
    logger.info('remove outliers and modify datetime')
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()
    df['last_review'] = pd.to_datetime(df['last_review'])

    # Removing non NYC data points
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()

    # Save dataframe as clean_sample.csv
    logger.info(f'Saving Dataframe {args.output_artifact}')
    df.to_csv('clean_sample.csv', index=False)

    # Create clean_sample.csv artifact
    logger.info('Creating artifact')
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )

    artifact.add_file(local_path='clean_sample.csv')

    # Upload the artifact to wandb website
    logger.info(f'Log artifact {args.output_artifact}')
    run.log_artifact(artifact)

    # Finish Run
    os.remove(args.output_artifact)
    run.finish()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")


    parser.add_argument(
        "--input_artifact", 
        type=str,
        help= "name of input artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="name of the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="type of output file",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="description of output",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="minimum price limit for outlier",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="maximum price limit for outlier",
        required=True
    )


    args = parser.parse_args()

    go(args)
