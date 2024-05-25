import yaml
import os
import json
import logging
import time


def file_exists(file_path):
    is_exist = os.path.exists(file_path)
    if is_exist:
        logging.info(f"File is exis for {file_path}")
        return True

def read_yaml(path_to_yaml:str) -> dict:
    with open(path_to_yaml) as yaml_file:
        content = yaml.safe_load(yaml_file)
    logging.info(f"yaml  file: {path_to_yaml} loaded successfully")
    return content

def create_dirs(dirs: list):
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)
        logging.info(f"Directory is created at {dir}")