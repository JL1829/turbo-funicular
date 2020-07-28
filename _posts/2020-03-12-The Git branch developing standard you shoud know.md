---
toc: true
description: Some example of Git branch develop standard coder should follow
categories: [git]
comments: true
---

# The Git branches developing standard you should know

![git](/images/git.png)

Git is by far the most popular source code management tool. In order to standardize development, keep the code commit record and git branch structure clear, and facilitate subsequent maintenance. Coder should bear in mind there's some specification of branch naming, commit messages formating, and what & when to merge different branches. 
In this blog, we are going to talk about the following topic: 

* Branch Mangement
    * Naming convention
    * Routine Task
* Message Specification
    * Commit messages

## Branch Management
### Branch Naming
#### Master Branch
* `master` is the major branch, it's also the production deployment branch, stablity is the uppermost important. 
* Normally `master` branch is merged by `develop` and `hotfix` branch, modfying code directly into the `master` branch is not allowed at all time. 

#### Develop branch
* `develop` branch is the branch for further developing, keep updated all the time and merging the code when bugs was fixed. 
* When developing some new feature, there will be a `feature` branch breakout from `develop` branch in order to provide a "clean" environment for new feature developing. 

#### Feature branch
* Based on `develop` branch, create `feature` branch
* Naming: different features is named after the `feature/`, such as `feature/preprocessing_module`, `feature/unsupervised_module`

#### Release branch
* Release is a pre-launch branch. During the release and testing phase, the release branch code is used as a benchmark for testing.

>When a group of feature development is completed, it will first be merged into the develop branch. When entering the test, a release branch will be created.
>
>If there are bugs that need to be fixed during the test, the developers will fix and submit them directly in the release branch.
>
>After the test is completed, merge the release branch into the master and develop branches. At this time, the master is the latest code and is used for going online.

#### Hotfix branch
* Branch naming: Started with `hotfix/`, and its naming rules are similar to feature branches, it build for bug fixing.
* When there's emergency, the bug need to be fixed in time. Use the `master` branch as the baseline and create a `hotfix` branch. After the debug is completed, it need to be merged it into the `master` branch and `develop` branch.

### Routine Task
#### New Function developing
```shell
(dev)$: git checkout -b feature/xxx            # create feature from dev
(feature/xxx)$: vi nlp_train.py                # develop
(feature/xxx)$: git add xxx
(feature/xxx)$: git commit -m 'commit comment'
(dev)$: git merge feature/xxx --no-ff          # merge feature into dev
```

#### Bug fix
```shell
(master)$: git checkout -b hotfix/xxx         # create hotfix from master
(hotfix/xxx)$: vi nlp_train.py                # bug fix
(hotfix/xxx)$: git add xxx
(hotfix/xxx)$: git commit -m 'commit comment'
(master)$: git merge hotfix/xxx --no-ff       # merge hotfix into master
(dev)$: git merge hotfix/xxx --no-ff          # merge hotfix into dev
```

#### Testing
```shell
(release)$: git merge dev --no-ff             # merge dev into release
```

#### Bring it into production
```shell
(master)$: git merge release --no-ff          # merge release into master
(master)$: git tag -a v0.1 -m 'Version name'  # name the version and tag
```

Above process in one picture
![branches](/images/branches.jpg)

## Message Specification
>In a team collaboration project, developers often need to submit some code to fix bugs or implement new features. The files in the project, what functions are implemented, and what problems are solved are gradually forgotten, and finally you need to waste time reading the code. But writing good formatting commit messages helps us, and it also reflects whether a developer is a good collaborator.

**A good formatting commit messages can have this benefits:**
* Speed up the review process
* Help the team to write up Release Note
* Let the team know what and why this features were added and how the bugs were fixed. 

Currently there's different version of commit messages, the **Angular** is the most accepted version, such as: 
![commit_messages](/images/commit_messages.jpg)

### Commit messages format
Please refer to [Angular Git Commit Guidelines](https://github.com/angular/angular.js/blob/master/DEVELOPERS.md#-git-commit-guidelines)
```shell
<type>: <subject>
<BLANK LINE>
<body>
<BLANK LINE>
<footer>
```

* **Type**: what's the type of this commit, is `bugfix` or `docs` or `style` etc.
* **Scope**: the affecting range of this commit.
* **Subject**: Describe the main idea of this commit. 
* **Body**: Detail explaination of this commit, such as motivation of this commit
* **Footer**: Is there any related issue or break change?

#### Different Types: 
* **feat**: New Features
* **fix**: Bug fix
* **docs**: Documentation modification
* **style**: Changing coding style, such as whitespace, indentation. 
* **refactor**: Refactor the code, but did not change the logic
* **perf**: adding some new code for performance test
* **test**: Adding test code
