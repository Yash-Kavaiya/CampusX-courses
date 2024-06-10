# Session 3 - Data to Model Pipeline through DVC

|                                 Github                                          |
|---------------------------------------------------------------------------------|
|           https://github.com/Himanshu-1703/nyc-taxi                             |
|  https://www.kaggle.com/code/himanshuarora1703/feature-engineering-on-taxi-data |

### What is DVC?

DVC, or Data Version Control, is an open-source version control system for data science and machine learning projects. It is designed to handle large datasets and machine learning models, making it easier to manage changes, reproduce experiments, and collaborate effectively. DVC extends the capabilities of Git, which is traditionally used for versioning code, to handle large files and data.

### Key Features of DVC:

1. **Data Management:** Handles large datasets and machine learning models efficiently.
2. **Version Control:** Tracks changes in data and models, similar to how Git tracks changes in code.
3. **Reproducibility:** Ensures that experiments can be reproduced with the same results.
4. **Collaboration:** Facilitates collaboration among team members by sharing data and models efficiently.
5. **Pipeline Management:** Manages complex machine learning pipelines, ensuring each step can be reproduced.

### How to Use DVC

#### Installation

Before using DVC, ensure you have Git installed. Then, install DVC using pip:

```bash
pip install dvc
```

#### Initializing DVC in a Project

1. **Initialize Git Repository:**

   ```bash
   git init
   ```

2. **Initialize DVC:**

   ```bash
   dvc init
   ```

This sets up a `.dvc` directory and a configuration file, preparing your project for DVC.

#### Tracking Data Files

To track a dataset (e.g., `data/data.csv`), use the `dvc add` command:

```bash
dvc add data/data.csv
```

This creates a `.dvc` file (e.g., `data/data.csv.dvc`) which contains metadata about the data file. Commit the changes to Git:

```bash
git add data/data.csv.dvc .gitignore
git commit -m "Add dataset"
```

#### Remote Storage Configuration

DVC supports various remote storage options like AWS S3, Google Drive, Azure Blob Storage, etc. To set up remote storage, use:

```bash
dvc remote add -d myremote s3://mybucket/path
```

Push the data to remote storage:

```bash
dvc push
```

#### Managing ML Pipelines

DVC allows you to define pipelines to track the sequence of steps in a machine learning workflow. Create a pipeline stage with:

```bash
dvc run -n stage_name -d input_file -o output_file command_to_run
```

For example, if you have a preprocessing script:

```bash
dvc run -n preprocess -d data/raw.csv -o data/preprocessed.csv python preprocess.py
```

DVC creates a `dvc.yaml` file describing the pipeline.

#### Reproducing Experiments

To reproduce the pipeline, use:

```bash
dvc repro
```

This reruns the necessary steps to regenerate outputs.

### Why We Need to Use DVC

1. **Data and Model Versioning:**
   - DVC provides a way to version control large datasets and models, allowing you to keep track of changes and revert to previous versions if needed.

2. **Reproducibility:**
   - Ensures that machine learning experiments are reproducible by tracking the exact versions of data, code, and parameters used.

3. **Collaboration:**
   - Facilitates collaboration in data science teams by enabling easy sharing of datasets and models without the need to include large files in the Git repository.

4. **Scalability:**
   - Handles large files and datasets efficiently, which is crucial for real-world machine learning projects where data can be substantial.

5. **Pipeline Management:**
   - Simplifies the management of complex ML pipelines, ensuring that each step is properly tracked and reproducible.

6. **Integration with Existing Tools:**
   - Integrates seamlessly with Git and other tools commonly used in data science workflows, making it easy to adopt without significant changes to existing processes.

### Example Workflow

1. **Initialize Project:**

   ```bash
   git init
   dvc init
   ```

2. **Track Data:**

   ```bash
   dvc add data/dataset.csv
   git add data/dataset.csv.dvc .gitignore
   git commit -m "Add dataset"
   ```

3. **Set Up Remote Storage:**

   ```bash
   dvc remote add -d myremote s3://mybucket/path
   dvc push
   ```

4. **Define Pipeline:**

   ```bash
   dvc run -n preprocess -d data/dataset.csv -o data/preprocessed.csv python preprocess.py
   ```

5. **Reproduce Pipeline:**

   ```bash
   dvc repro
   ```

By following these steps, you can ensure that your data science projects are well-managed, reproducible, and collaborative.

| Feature                        | Git                                          | DVC                                                 |
|--------------------------------|----------------------------------------------|-----------------------------------------------------|
| Primary Use                    | Version control for code                     | Version control for data and machine learning models|
| Handling Large Files           | Not efficient                                | Efficient, designed for large files                 |
| Data Storage                   | Local repository                             | Supports remote storage (S3, GDrive, Azure, etc.)   |
| Versioning                     | Tracks changes in code files                 | Tracks changes in data, models, and pipelines       |
| Reproducibility                | Limited to code                              | Ensures reproducibility of entire ML experiments    |
| Collaboration                  | Code collaboration                           | Data and model collaboration                        |
| Integration                    | Works with code repositories                 | Integrates with Git, extends functionality          |
| Pipeline Management            | Manual                                       | Automated with `dvc.yaml` and `dvc run`             |
| Metadata                       | Stores code metadata                         | Stores metadata for data and pipelines              |
| Handling Binary Files          | Poor                                         | Efficient, designed to handle binaries              |
| Remote Storage Support         | Requires external tools (like LFS)           | Built-in support                                    |
| Commands                       | `git add`, `git commit`, `git push`          | `dvc add`, `dvc push`, `dvc repro`                  |
| Learning Curve                 | Steeper for beginners                        | Moderate, especially for those familiar with Git    |
| Installation                   | `git` command line tool                      | `pip install dvc`                                   |
| Dependency Management          | Code dependencies                            | Data, model, and code dependencies                  |
| Branching and Merging          | Efficient                                    | Efficient when combined with Git                    |

| Command                 | Description                                                                                  |
|-------------------------|----------------------------------------------------------------------------------------------|
| `dvc init`              | Initializes a new DVC project in the current directory.                                      |
| `dvc add <file>`        | Adds a data file or directory to DVC for tracking.                                           |
| `dvc remote add <name> <url>` | Adds a remote storage location for storing data and models.                              |
| `dvc remote modify <name> <option> <value>` | Modifies the settings of an existing remote storage.                              |
| `dvc push`              | Uploads tracked files to the remote storage.                                                 |
| `dvc pull`              | Downloads tracked files from the remote storage.                                             |
| `dvc fetch`             | Downloads files from the remote storage without placing them in the workspace.               |
| `dvc status`            | Shows the status of data files, indicating if they are up to date.                           |
| `dvc repro`             | Reproduces the pipeline by running necessary stages.                                         |
| `dvc run -n <name> -d <dependencies> -o <outputs> <command>` | Defines a pipeline stage with specified dependencies and outputs, running a command. |
| `dvc pipeline show`     | Displays the pipeline graph.                                                                 |
| `dvc pipeline list`     | Lists all the stages in the pipeline.                                                        |
| `dvc config`            | Manages DVC configuration settings.                                                          |
| `dvc checkout`          | Restores files in the workspace from the cache or remote storage.                            |
| `dvc commit`            | Saves changes to tracked files in the DVC cache.                                             |
| `dvc diff`              | Shows the differences between commits in data and pipeline stages.                           |
| `dvc lock`              | Locks the specified stage to prevent it from being automatically reproduced.                 |
| `dvc unlock`            | Unlocks the specified stage, allowing it to be reproduced.                                   |
| `dvc metrics show`      | Displays metrics tracked in the project.                                                     |
| `dvc metrics diff`      | Shows the difference in metrics between two commits or branches.                             |
| `dvc params diff`       | Shows the difference in parameters between two commits or branches.                          |
| `dvc params show`       | Displays the parameters tracked in the project.                                              |
| `dvc gc`                | Garbage collects unused files from cache and remote storage.                                 |
| `dvc import <url>`      | Imports a file or directory from an external DVC repository.                                 |
| `dvc import-url <url>`  | Downloads a file or directory from a URL and tracks it with DVC.                             |
| `dvc move <source> <destination>` | Moves a tracked file or directory, updating DVC metadata accordingly.                 |
| `dvc remove <file>`     | Removes a tracked file or directory from DVC.                                                |
| `dvc unprotect <file>`  | Unprotects a file or directory, making it writable.                                          |

### `.gitkeep` Files

`.gitkeep` is not an official Git feature but rather a convention used by developers. It is typically used to ensure that empty directories are tracked by Git. Git does not track empty directories by default, so adding a `.gitkeep` file inside an empty directory ensures that the directory is included in the Git repository. The `.gitkeep` file itself is an empty file and serves no purpose other than to force Git to track the directory.

#### Example Use:
1. Create an empty directory:
   ```bash
   mkdir empty_directory
   ```
2. Add a `.gitkeep` file inside the directory:
   ```bash
   touch empty_directory/.gitkeep
   ```
3. Add and commit the directory to Git:
   ```bash
   git add empty_directory/.gitkeep
   git commit -m "Add empty directory with .gitkeep"
   ```

### `.` Files (Dotfiles)

Dotfiles are hidden configuration files on Unix-like operating systems. They are named with a leading dot (`.`), which makes them hidden by default in file directory listings. These files often store settings and preferences for various applications and shell environments.

#### Common Dotfiles:
- **`.bashrc` / `.bash_profile` / `.zshrc`:** Configuration files for Bash and Zsh shells.
- **`.gitconfig`:** User-specific Git configuration.
- **`.vimrc`:** Configuration for the Vim text editor.
- **`.profile`:** User-specific environment and startup programs.
- **`.env`:** Environment variables for various applications and scripts.

#### Example Use of Dotfiles:
1. **Setting Up a Custom Git Configuration:**
   Create or edit the `.gitconfig` file in your home directory to customize Git settings:
   ```ini
   [user]
       name = Your Name
       email = you@example.com
   [core]
       editor = vim
   ```

2. **Customizing Shell Behavior:**
   Add custom aliases and functions to your `.bashrc` or `.zshrc` file:
   ```bash
   alias ll='ls -la'
   export PATH=$PATH:~/custom/scripts
   ```

### Differences and Uses:

- **Purpose:**
  - `.gitkeep` is used to ensure empty directories are tracked by Git.
  - Dotfiles (files starting with a dot) are used to configure and customize the behavior of applications and shell environments.

- **Visibility:**
  - Both `.gitkeep` and dotfiles are hidden by default due to the leading dot.

- **Context:**
  - `.gitkeep` is specific to Git repositories and is a workaround for Git's inability to track empty directories.
  - Dotfiles are a broader concept used across Unix-like systems for configuration purposes.

Understanding these files and their purposes helps in managing both Git repositories and Unix-like system configurations effectively.

