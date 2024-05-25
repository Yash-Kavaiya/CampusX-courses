# Session 1 - Introduction to MLOPs and Version Control

**MLops (Machine Learning Operations)**

MLops is a practice that combines machine learning (ML) with software engineering practices to automate the development, deployment, and management of ML models in production. It ensures a smooth and efficient lifecycle for your models, from initial development to ongoing monitoring and improvement.

Here are some key aspects of MLops:

- **Continuous Integration/Continuous Delivery (CI/CD):** Automates the building, testing, and deployment of ML models, enabling frequent updates and faster delivery cycles.
- **Model Versioning:** Tracks changes made to models, data, and code, allowing you to rollback to previous versions if necessary and ensuring reproducibility.
- **Experiment Tracking:** Logs and stores details about different model training runs, making it easier to compare results, identify the best performing models, and understand how changes impact performance.
- **Monitoring:** Tracks the performance of deployed models in production, identifying potential issues and alerting teams when performance degrades.
- **Governance:** Establishes guidelines and best practices for managing ML models throughout their lifecycle, promoting consistency, quality, and security.

**Scalability**

Scalability refers to a system's ability to handle increasing amounts of data, users, or requests without significant performance degradation. In ML, scalability is crucial for:

- **Training on Larger Datasets:** As the amount of data available for training grows, the system needs to be able to handle it efficiently. Distributed training techniques are often used to leverage multiple machines or cloud resources.
- **Serving Models in Production:** Models deployed in production might receive a high volume of requests. The system needs to be able to scale to meet peak demands while maintaining low latency (response time). Scalable infrastructure like cloud platforms can be employed.

**Reusability**

Reusability is the ability to use code, models, or components in multiple projects. This saves time and effort, promotes consistency, and reduces errors:

- **Modular Code:** Breaking down code into well-defined functions and classes makes it easier to reuse in different projects.
- **Pre-trained Models:** Reusing pre-trained models as a starting point for new tasks can significantly reduce training time and improve initial performance.
- **Standardized Data Pipelines:** Designing data pipelines with reusability in mind allows you to easily apply them to new datasets with minimal modifications.

**Automation**

Automation involves using tools and scripts to automate repetitive tasks in the ML workflow. This streamlines the process, reduces manual errors, and improves efficiency:

- **Automated Data Preprocessing:** Automating data cleaning, normalization, and feature engineering tasks frees up data scientists to focus on more strategic work.
- **Automated Model Training:** Scripting the model training process allows you to easily reproduce results, run hyperparameter tuning, and iterate on models quickly.
- **Automated Model Deployment:** Automating model deployment eliminates the need for manual configuration and ensures consistent deployments across different environments.

**Version Control**

Version control systems (VCS) like Git are essential for tracking changes to code, data, and models throughout the ML lifecycle:

- **Model Lineage:** Provides a history of changes made to models, which models were derived from which ones, and what data they were trained on. This is crucial for debugging issues and understanding model behavior.
- **Reproducibility:** Allows you to recreate the exact environment and conditions under which a model was trained, ensuring results can be replicated.
- **Rollback Capability:** Enables you to revert to previous versions of models or data in case of problems during deployment or experimentation.

By effectively incorporating these practices, you can streamline your ML development process, improve model performance, and ensure reliable and scalable deployments in production.

In machine learning (ML), version control for models, data, and code is paramount for ensuring reproducibility, maintaining a clear history of your work, and facilitating collaboration. Here's a breakdown of how version control helps with each element:

**Model Version Control**

* **Tracks Changes:** Keeps a record of modifications made to the model architecture, hyperparameters, and training process. This allows you to understand how changes have impacted performance and revert to previous versions if needed.
* **Reproducible Results:** Enables you to recreate the exact model trained on a specific version of the data using a specific codebase. This is crucial for scientific rigor and debugging.
* **Experiment Comparison:** Version control helps you compare different models trained with varying configurations. You can easily identify the best-performing model based on version information.

**Data Version Control**

* **Lineage Tracking:** Tracks the origin and transformations applied to your data. This is vital for understanding how data quality impacts model performance and identifying potential biases.
* **Reproducible Training:** Ensures you're using the same exact data for training in different experiments or model iterations. This consistency is essential for fair comparisons and reliable results.
* **Rollback Capability:** Allows you to revert to a previous version of the data if issues arise after preprocessing or cleaning. This can save time and effort spent on retraining models.

**Code Version Control**

* **Collaboration:** Enables multiple data scientists to work on the same project simultaneously without conflicts. Version control tracks changes and allows merging of code effectively.
* **Bug Tracking:** Helps pinpoint the exact code changes that introduced bugs, streamlining the debugging process. You can revert to a previous version of the codebase to isolate the issue.
* **Improved Efficiency:** Version control promotes modular code design, as different parts of the code can be tracked and reused independently.

**Tools for Version Control**

* **Git:** The most popular version control system, used for managing code changes. It can also be extended to track models and data through tools like DVC (Data Version Control).
* **MLflow:** An open-source platform for managing the ML lifecycle, including model versioning and experiment tracking.
* **Neptune.ai:** A cloud-based platform for ML lifecycle management, offering versioning for models, code, and data.

By implementing version control for models, data, and code, you gain greater control and transparency in your ML projects. This fosters collaboration, facilitates debugging, and ensures reproducible results, leading to more robust and reliable ML pipelines.

These commands are part of the command-line interface (CLI) found in Windows environments like Command Prompt or PowerShell. Here's a breakdown of what each one does:

**cd (change directory)**

* **Purpose:** Navigates between directories (folders) in the file system.
* **Usage:** `cd <directory_name>`
  * Replace `<directory_name>` with the name of the directory you want to move to.
  * Examples:
    * `cd Desktop` - Moves you to the "Desktop" directory.
    * `cd ..` - Moves you to the parent directory of the current one.
    * `cd /` - Moves you to the root directory (top level of the file system).

**dir (directory)**

* **Purpose:** Lists the contents of the current directory.
* **Usage:** `dir` (standalone command)
  * You can optionally add switches for additional information:
    * `/a` - Shows all files, including hidden ones.
    * `/w` - Shows filenames in a wide format.
    * `/s` - Lists all files recursively, including subdirectories.
  * Examples:
    * `dir` - Lists all files and folders in the current directory.
    * `dir /a` - Lists all files, including hidden ones.

**cp (copy)**

* **Purpose:** Copies files or directories from one location to another.
* **Usage:** `cp <source> <destination>`
  * Replace `<source>` with the path to the file or directory you want to copy.
  * Replace `<destination>` with the path where you want to create the copy.
    * If the destination is a directory, the copied file will have the same name as the source.
  * You can optionally use wildcards like `*` or `?` to copy multiple files.
  * Examples:
    * `cp report.txt Documents` - Copies "report.txt" to the "Documents" directory.
    * `cp *.jpg Pictures` - Copies all files with the ".jpg" extension to the "Pictures" directory.

**Additional Notes:**

* These commands are case-sensitive, so `Cd` is not the same as `cd`.
* You can use Tab key for autocompletion of file and directory names.
* For more advanced usage and options for these commands, you can refer to the official Microsoft documentation by typing `help cd` or `help dir` in the Command Prompt.

## Absolute vs. Relative Paths in Markdown

| Feature | Absolute Path | Relative Path |
|---|---|---|
| **Definition** | A complete path that specifies the location of a file or directory from the root directory (top level) of the file system.  | A path that specifies the location of a file or directory relative to the current working directory or the location of the Markdown file. |
| **Example** |  * `C:\Users\JohnDoe\Documents\report.md` (Windows)  <br> * `/home/user/projects/report.md` (Linux/macOS) |  * `./images/chart.png` (current directory) <br> * `../data/data.csv` (one directory up) |
| **Markdown Syntax**  |  * Not commonly used in Markdown due to portability issues.  |  * Used with image references, links to other Markdown files, or including external files.  |
| **Advantages** |  * Unambiguous, always points to the exact location regardless of the current working directory. |  * More concise and portable, especially when files are organized within the same project or directory structure. |
| **Disadvantages** |  * Less portable, breaks if the file structure changes or the Markdown file is moved to a different location. |  * Relies on the current working directory or the location of the Markdown file, can lead to broken links if not used carefully. |
| **Best Use Cases** |  * Referencing system-wide resources (rare in Markdown). |  * Linking to images, other Markdown files, or including external files within the same project or directory structure. |


**Additional Notes:**

* In Markdown, relative paths are typically used within the same document or within a related set of Markdown files.
* You can use forward slashes (`/`) in both absolute and relative paths on all operating systems.
* For maximum portability, consider structuring your Markdown files and related resources (images, data) within a well-defined directory structure and use relative paths within that structure.  

I'd be glad to explain `pathlib` and `os` for file system interactions in Python:

**pathlib**

- **Introduction:** The `pathlib` module, introduced in Python 3.4, offers a modern and object-oriented approach to working with file paths. It provides a more intuitive and user-friendly way to handle file system operations compared to the traditional `os` module.

- **Key Features:**
    - **Path Objects:** `pathlib` represents paths as objects with attributes and methods. This allows for cleaner and more Pythonic code.
    - **Platform Independence:** It handles path separators (`/` for Unix-like systems, `\` for Windows) automatically, ensuring code portability.
    - **Rich Functionality:** Provides methods for common path operations like joining, splitting, resolving, checking existence, getting file size, and more.
    - **Error Handling:** Raises exceptions for invalid paths or operations, making it easier to catch and handle errors.

- **Example:**

```python
from pathlib import Path

# Create a Path object
data_dir = Path("data")

# Check if the directory exists
if data_dir.exists():
    print("data directory exists")

# Create a new file (if it doesn't exist)
report_file = data_dir / "report.txt"
report_file.touch()  # Creates an empty file if it doesn't exist

# Get the absolute path of the report file
report_path = report_file.absolute()
print(report_path)
```

**os**

- **Introduction:** The `os` module has been part of the Python standard library since its early days. It provides various functions for interacting with the operating system, including file system manipulation.

- **Key Functions:**
    - `os.path.join(*paths)`: Joins multiple path components into a single path string.
    - `os.path.exists(path)`: Checks if a path exists.
    - `os.makedirs(path, exist_ok=True)`: Creates nested directories (raises an error by default if the directory already exists; `exist_ok=True` prevents the error).
    - `os.listdir(path)`: Lists files and directories within a directory.
    - `os.remove(path)`: Removes a file.
    - `os.rename(src, dst)`: Renames a file or directory.
    - Many more functions for advanced operations.

- **Example:**

```python
import os

# Create directories (if they don't exist)
data_dir = os.path.join("data", "processed")
os.makedirs(data_dir, exist_ok=True)

# List files in the current directory
files = os.listdir(".")
print(files)

# Remove a file (if it exists)
report_file = os.path.join("reports", "old_report.txt")
if os.path.exists(report_file):
    os.remove(report_file)
```

**Choosing Between `pathlib` and `os`**

- In general, **`pathlib` is recommended for most file system operations** in modern Python due to its object-oriented approach, platform independence, rich functionality, and clear error handling.
- Use `os` when:
    - You need low-level control over specific operating system functionality not readily available in `pathlib`.
    - You're working with older Python versions that don't have `pathlib`.
    - You're already familiar with `os` and prefer its style.

By understanding the strengths and use cases of both `pathlib` and `os`, you can effectively handle file system tasks in your Python applications.

In the three-digit version number (1.2.1), each digit conventionally represents a different level of change or update made to the software:

**1. Major Version (X):**

- This signifies significant changes that may introduce **breaking changes** to the API (Application Programming Interface) or functionality. Breaking changes means that existing code written for previous versions might no longer work as expected with the new major version.
- Upgrading from a lower major version (e.g., 0.x.y) to a higher major version (e.g., 1.x.y) often requires code modifications to adapt to the new API or functionality.
- Major version increments typically indicate major overhauls, feature additions, or rewrites.

**2. Minor Version (Y):**

- This represents new features or **bug fixes** that are generally **backwards compatible** with the previous major version. Backwards compatibility implies that existing code written for the previous minor version (e.g., 1.1.x) should still work without requiring changes in the new minor version (e.g., 1.2.x).
- Minor version increments usually introduce new functionalities, enhancements, or non-breaking bug fixes.

**3. Patch Version (Z):**

- This signifies minor bug fixes or improvements that do not introduce breaking changes. It's the most frequent type of update.
- Patch versions are for addressing bugs, security vulnerabilities, or minor improvements without altering existing functionalities.

**Example Breakdown (Version 1.2.1):**

- In the version 1.2.1, a major change (possibly introducing breaking changes) likely occurred at version 1.0.0.
- Since then, there have been two minor version updates (1.1.x and 1.2.x) that introduced new features or bug fixes while likely maintaining backwards compatibility.
- Finally, the patch version 1.2.1 indicates a minor bug fix or improvement within the 1.2.x minor version series.

**Key Points:**

- The specific meaning and significance of each version number can vary depending on the software and its development team's versioning practices. However, the general convention described above is widely used.
- It's always recommended to consult the software's documentation or release notes to understand the exact changes introduced in each version update.

When you encounter a version number like 1.2.1, it's typically following the Semantic Versioning (SemVer) convention, especially in the context of software development. Semantic Versioning is a versioning scheme for using meaningful version numbers to convey the nature of changes between releases. Each of the three digits in a SemVer version number, separated by dots, has a specific meaning:

- **MAJOR version** when you make incompatible API changes,
- **MINOR version** when you add functionality in a backwards compatible manner, and
- **PATCH version** when you make backwards compatible bug fixes.

### Major Version `1`
The first digit is the **MAJOR** version number. It indicates significant changes to the software, which are not backward compatible. This means that software using an older major version might not work correctly if it's updated to a newer major version without some modification to accommodate the changes. For example, moving from version `1.2.1` to `2.0.0` would indicate a major overhaul that could break compatibility with version 1.x codebases.

### Minor Version `2`
The second digit is the **MINOR** version number. It represents backward-compatible improvements or additions to the software. These changes add functionality but do not break the software's existing API or existing functionalities. For instance, upgrading from version `1.2.1` to `1.3.0` would add new features or improvements, but it wouldn't require any changes to how you use the software.

### Patch Version `1`
The third digit is the **PATCH** version number. This is incremented when backward-compatible bug fixes are introduced. These fixes address problems that do not alter the software's functionality or existing API but rather correct errors to ensure the software operates as intended. So, if a software moves from `1.2.1` to `1.2.2`, it means that some bugs have been fixed, but no new features have been added, and no existing features have been changed in a way that would break backward compatibility.

### Beyond 1.2.1
When you're looking at versions above `1.2.1`, you're essentially looking at increments in these three numbers based on the type of changes that have been made according to the rules of Semantic Versioning. For example:

- `1.2.2` would indicate a patch has been made to fix bugs.
- `1.3.0` would indicate new features have been added without breaking backward compatibility.
- `2.0.0` would indicate a major change that may not be backward compatible.

This versioning system is widely adopted in software development because it helps developers understand the potential impact of updating a dependency based on the version number alone.

Git is a distributed version control system (DVCS) that is widely used in software development to track and manage changes to codebases. It was created by Linus Torvalds in 2005 to support the development of the Linux kernel. Git's primary goal is to enable multiple developers to work on the same project efficiently, allowing for non-linear development through branches, and ensuring data integrity and speed. Here's a detailed breakdown of its key concepts and features:

### Core Concepts

- **Repository (Repo):** A repository is a directory or storage space where your projects can live. It can be local to a folder on your computer, or it can be a storage space on GitHub or another online host. The repository contains all of the project's files and stores each file's revision history.

- **Commit:** A commit is an individual change to a file (or set of files). It's like a snapshot of your entire repository at a specific point in time. Each commit has a unique ID (also known as a "SHA" or "hash") that allows you to keep record of what changes were made, by whom, and when.

- **Branch:** A branch in Git is simply a lightweight movable pointer to one of these commits. The default branch name in Git is `master`. As you start making commits, you're given a master branch that points to the last commit you made. You can create new branches from existing commits to work on new features or bug fixes independently from the main project (master branch).

- **Merge:** When you've completed work on a branch and it's ready to be integrated into the main project (or another branch), you merge those changes. This involves combining the changes made in one branch with another, which can be done automatically by Git if there are no conflicting changes, or may require manual intervention and conflict resolution.

### Key Features

- **Distributed Development:** Git gives every developer a local copy of the entire development history, and changes are copied from one repository to another. This allows for easy branching and merging and provides developers with the ability to work offline.

- **Data Integrity:** Git uses a data model that ensures the cryptographic integrity of every part of your project. Each file and commit is checksummed and retrieved by its checksum at the time of checkout, making it impossible to change the contents of any file or directory without Git knowing about it.

- **Speed:** Git is designed to handle projects of any size with speed and efficiency. Operations like commits, branching, merging, and comparing past versions are all optimized to be as fast as possible.

- **Non-linear Development:** Git supports rapid branching and merging and includes tools for visualizing and navigating a non-linear development history. This enables a workflow that can adapt to the project's needs, such as feature branches, topic branches, and forked workflows.

- **Staging Area:** Git has an intermediate area known as the "staging area" or "index," where commits can be formatted and reviewed before completing the commit.

- **Free and Open Source:** Git is released under the GNU General Public License version 2.0, which means it is freely available for anyone to use, copy, modify, and distribute.

### Common Commands

- `git init`: Initializes a new Git repository.
- `git clone [url]`: Creates a local copy of a remote repository.
- `git add [file]`: Stages a file for commit.
- `git commit -m "[message]"`: Commits the staged snapshot to the project history.
- `git status`: Lists the status of working directory and staging area.
- `git branch`: Lists existing branches and allows you to create new ones.
- `git merge [branch]`: Merges specified branch into the current branch.
- `git pull`: Fetches changes from the remote repository and merges them into the current branch.
- `git push`: Pushes local branch commits to the remote repository branch.

Git has become the standard for version control in software development, thanks to its robust features, efficiency, and flexibility in managing complex projects with multiple contributors.

The `.gitignore` file is a crucial component in Git repositories, used to specify intentionally untracked files that Git should ignore. Files already tracked by Git are not affected by `.gitignore`; it only affects untracked files. By using `.gitignore`, developers can prevent certain files and directories from being added to the version control system, which is particularly useful for excluding temporary, private, or machine-specific files that don't need to be shared with others (e.g., build outputs, temporary files created by development tools, sensitive information).

Here's a breakdown of its key aspects:

### Purpose
- **Exclude specific files:** Useful for excluding files that are generated during runtime or compilation, such as `.log`, `.tmp`, or binary files.
- **Exclude directories:** Entire directories can be ignored, which is handy for node modules, build directories, or virtual environments.
- **Keep the repository clean:** Helps in maintaining a clean project structure without unnecessary files.
- **Improve performance:** Reduces the load on Git, making operations like `git status`, `git add`, and `git commit` faster by not processing ignored files.

### How It Works
- **Patterns:** The `.gitignore` file uses globbing patterns to match file names. For example, `*.log` ignores all files with the `.log` extension, and `build/` ignores all files within the `build` directory.
- **Location:** A `.gitignore` file can be placed in the repository's root directory or in any subdirectory to apply its rules recursively within that directory.
- **Multiple `.gitignore` files:** Different `.gitignore` files can be used in different directories for more granular control.
- **Negation:** By prefixing a pattern with an exclamation mark (`!`), files can be included again. For example, `!important.log` would track a file named `important.log` even if `*.log` files are ignored.

### Common Use Cases
- Ignoring compilation output directories (e.g., `/bin`, `/out`, `/target`).
- Ignoring package directories (e.g., `/node_modules`, `/venv`).
- Ignoring IDE and editor configuration files (e.g., `.idea/`, `*.sublime-workspace`).
- Ignoring system files (e.g., `.DS_Store` on macOS, `Thumbs.db` on Windows).
- Ignoring environment configuration files that may contain sensitive information.

### Example `.gitignore` File
```
# Ignore all .log files
*.log

# Ignore specific directories
node_modules/
build/

# Ignore all .txt files in the doc/ directory
doc/*.txt

# But do not ignore this specific file
!doc/README.txt
```

The `.gitignore` file is a powerful tool for managing the files included in a Git repository, ensuring that only relevant source code and resources are versioned and shared.


### 1. `git init`
- **Purpose:** Initializes a new Git repository. This command creates a new `.git` directory in your project, which holds all of the necessary metadata for the repo. Running this command sets up the necessary Git environment.
- **Usage:** `git init`

### 2. `git clone [url]`
- **Purpose:** Creates a local copy of a remote repository. This command is used when you want to get a copy of an existing Git repo to work on your machine.
- **Usage:** `git clone https://github.com/user/repo.git`

### 3. `git add [file]`
- **Purpose:** Adds files to the staging area. It tells Git that you want to include updates to a particular file(s) in the next commit. However, `git add` doesn't really affect the repository in any significant way—changes are not actually recorded until you commit them.
- **Usage:** `git add <file>` or `git add .` to add all files

### 4. `git commit -m "[message]"`
- **Purpose:** Records or snapshots the file permanently in the version history with a descriptive message. The `-m` option allows you to add a commit message inline.
- **Usage:** `git commit -m "Fixed the bug causing system crash"`

### 5. `git status`
- **Purpose:** Shows the status of changes as untracked, modified, or staged. It gives you an overview of what's going on in your repository, including which changes have been staged, which haven't, and which files aren't being tracked by Git.
- **Usage:** `git status`

### 6. `git branch`
- **Purpose:** Lists all the local branches in the current repository or creates a new branch. It's a critical part of Git's branch-based workflow.
- **Usage:** `git branch` to list branches or `git branch <branch-name>` to create a new branch

### 7. `git checkout [branch-name]`
- **Purpose:** Switches branches or restores working tree files. It's used to switch from one branch to another or to checkout files or commits.
- **Usage:** `git checkout <branch-name>` or `git checkout -b <new-branch>` to create and switch to a new branch

### 8. `git merge [branch]`
- **Purpose:** Merges the specified branch’s history into the current branch. This is usually used to combine changes from a feature branch back into the main branch.
- **Usage:** `git merge <branch-name>`

### 9. `git pull`
- **Purpose:** Fetches and merges changes on the remote server to your working directory. It's a combination of `git fetch` followed by `git merge`.
- **Usage:** `git pull origin <branch-name>`

### 10. `git push [alias] [branch]`
- **Purpose:** Sends local branch commits to the remote repository branch. It's used to share your changes with others.
- **Usage:** `git push origin <branch-name>`

The `HEAD` in Git refers to the current pointer indicating the latest commit in the branch that is checked out in your working directory. It acts as a reference to the last commit made on the current branch, and it moves forward with each new commit you make.

Here are some key points about `HEAD`:

- **Current Snapshot**: `HEAD` represents the most recent commit of the branch you are working on. When you switch branches with `git checkout`, the `HEAD` revision changes to point to the tip of the new branch.

- **Detached HEAD State**: Normally, `HEAD` points to a branch reference, but it can also point directly to a commit. This is known as a "detached HEAD" state and can happen when you check out a specific commit rather than a branch. In this state, if you make commits, they won't belong to any branch and can be difficult to find later unless you create a new branch starting from those commits.

- **Navigating History**: You can use `HEAD` to navigate the commit history. For example, `HEAD~1` refers to the commit before the current one, `HEAD~2` refers to two commits before the current one, and so on. This is useful for operations like resetting to a previous state (`git reset HEAD~1`) or checking out a previous commit (`git checkout HEAD~1`).

- **Role in Branching and Merging**: When you create a new branch, Git sets the `HEAD` to point to the latest commit of the new branch. When you merge branches, `HEAD` will point to the merge commit if the merge is successful.

Understanding `HEAD` is crucial for navigating Git repositories, understanding the current state of your project, and performing operations like commits, branches, and merges effectively.

Branching in Git is a powerful feature that allows developers to diverge from the main line of development and work independently without affecting the main project. This is particularly useful for developing new features, fixing bugs, or experimenting with new ideas in a contained environment. Here's a detailed breakdown of branching in Git:

### What is a Branch?

In Git, a branch represents an independent line of development. Branches serve as pointers to a specific commit in the repository's history. The default branch in most Git repositories is called `master` or `main`, but you can create as many branches as you need.

### Why Use Branches?

- **Isolation:** Branches provide a way to work on different tasks simultaneously without interfering with each other. Each branch is isolated, so changes made in one branch do not affect others.
- **Collaboration:** Multiple developers can work on different features simultaneously by using separate branches. This enhances team collaboration and speeds up development.
- **Safety:** Branches allow you to experiment with new ideas in a safe environment. If an experiment fails, you can easily switch back to the main branch without impacting the project.

### How to Work with Branches

#### Creating a Branch

To create a new branch, use the `git branch` command followed by the name of the new branch:

```bash
e72921e4-a1f0-4488-86f5-a5eef799910d

git branch <branch-name>

```

#### Switching Branches

To switch to an existing branch, use the `git checkout` command:

```bash
git checkout <branch-name>
```

In newer versions of Git, you can also use `git switch`:

```bash
git switch <branch-name>
```

#### Listing Branches

To list all branches in your repository, use:

```bash
git branch
```

The current branch will be marked with an asterisk (*).

#### Merging Branches

When you've completed work on a branch and want to integrate it into another branch (e.g., merging a feature branch into the main branch), you use the `git merge` command:

```bash
git checkout main
git merge <branch-name>
```

This command merges the specified branch into the current branch.

#### Deleting Branches

After merging a branch, you might want to delete the old branch to clean up your repository. To do this, use:

```bash
git branch -d <branch-name>
```

Use `-D` instead of `-d` to force deletion if the branch has unmerged changes.

### Best Practices

- **Branch Naming:** Use descriptive names for your branches to make it clear what they are for (e.g., `feature/add-login`, `bugfix/fix-login-error`).
- **Regular Commits:** Commit your changes regularly to keep a detailed history and make it easier to identify and undo changes if needed.
- **Frequent Merges:** Regularly merge changes from the main branch into your feature branches to minimize merge conflicts.

Branching is a core concept in Git that, when used effectively, can significantly enhance the development workflow, making it more flexible and collaborative.

