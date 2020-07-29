---
layout: page
title: About Me
permalink: /about/
---

```python
class README(object):
    """Doc String PlaceHolder"""

    def __init__(self, username='JL1829', year=2020):
        self.username = username
        self.FullName = 'Johnny(ZHIPING) Lu'
        self.education = {
            'Machine Learning': ['Machine Learning', 'Stanford Online',
                                 'Deep Learning', 'Coursera',
                                 'Mathematic for Machine Learning', 'Imperial College London'],
            'Master': ['MSc of Technology and Intelligent System', 'NUS'], 
        }
    
    def doing(self, now=2020):
        today = self.year
        
        if now == today:
            experience = self.employment['Lead Consulting Engineer']
            return """
            I am a Lead Consulting Engineer for Machine Learning and Data Science, current
            working project:
             - Customer purchase value prediction
             - Customer sentimental anaylser for review
             - Using Autoencoder for Computer Network Anomaly Detection.
            """
            learning = self.education.get('Master')
            
        if abs(now - today) > 4:
            experience = self.employment['System Engineer']
            return """
            I am a system engineer with around 10 years of solution design and 
            customer facing experiences, designing solution based on current
            product, as well as developing customized product for particular requirement."""
            
        if now > today:
            goal = self.employment['Machine Learning Engineer', 'Data Scientist']
            return """
            I am eager to explore the bigger world of Machine Learning and Data Science.
            **Open for Opportunity**
            """
            
        else:
            return """
            ### Hi there ~~
            """

me = README(2020)
```


## Experiences

### Allied Telesis
- **Lead Consulting Engineerr**, (Machine Learning, Data Science).
*Singapore, Jun/2017 -- Current*

    - Leading Data Science and System Engineer team to develop products that measurably and efficiently improve sales and top-line growth.
    - Interfacing with customers to receive valuable product feedback.
    - Driving strategy and vision for products by translating research, customer insights, and data discovery into innovative solutions for customers. 
    - Actively collaborating with global IT, Architectures, Infrastructure, and Sales teams to deploy products.
    - Develop End-to-End Data Science, Machine Learning Project using: 
        - MySQL, Scikit-Learn, NumPy, Pandas, PySpark, TensorFlow 2, Keras
        - LightGBM, XGBoost, SpaCy, NLTK

- **Regional System Engineer** 
*Singapore, May/2016 -- Jun/2017*
    - Response to business development, including leads generating, roadshow/seminar conducting, solution designing, quoting/bidding, solution deploying, and after sales servicing. 
    - Promote Allied Telesis SDN, Service Cloud solution; initiative in the region, managing the technical team to provide high reliability services. 
    - Managing key account, provide advisory of the option for new technology adoption. 


### NETGEAR Inc.
- **Regional System Engineer**
*Singapore, Jun/2013 -- May/2016*
    - Act as a constant performer, teaming with Regional Sales Director, perform annually business growth. 
    - Provide consultation service to downstream partner & end user, including basic infra network design and deployment, integration with virtualization, customized solution for individual customers. 

- **Regional System Engineer**
*Guangzhou, China, Nov/2012 -- Oct/2013*
    - Design, develop and carry out technical solutions 
    - Establish and develop technical marketing objectives and goals 
    - Analyze and interpret marketing trends concerning technical products or services


## Professional Certification
* [Machine Learning](https://www.coursera.org/account/accomplishments/verify/M9QF62R25BUR) from [Stanford University Online](https://online.stanford.edu). *Oct/2018*
* [Mathematics for Machine Learning: Linear Algebra](https://www.coursera.org/account/accomplishments/verify/NW6MCXFBX47G), from [Imperial College London](https://www.imperial.ac.uk). *Feb/2019*
* [Mathematics for Machine Learning: Multivariate Calculus](https://www.coursera.org/account/accomplishments/verify/X64UR4U3AQFA), from from [Imperial College London](https://www.imperial.ac.uk). *Mar/2019*
* [Neural Network and Deep Learning](https://www.coursera.org/account/accomplishments/certificate/HY9746KQFZDD), from [deeplearning.ai](https://deeplearning.ai). *Mar/2019*
* [Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization](https://www.coursera.org/account/accomplishments/certificate/KRJRFJKGF4RE), from [deeplearning.ai](https://deeplearning.ai). *Apr/2019*
* [Convolutional Neural Networks](https://www.coursera.org/account/accomplishments/certificate/64Q5LSDGCVNN), from [deeplearning.ai](https://deeplearning.ai). *May/2019*
* [Sequance Models](https://www.coursera.org/account/accomplishments/certificate/7DVELPA6RWFP), from [deeplearning.ai](https://deeplearning.ai). *Jul/2019*


## Education Background

### National University of Singapore
> Master of Science - MS, Technology & Intelligent System
* Machine Reasoning
* Reasoning Systems
* Cognitive Systems
* Problem Solving using Pattern Recognition
* Intelligent Sensing and Sense Making
* Pattern Recognition and Machine Learning Systems
* Text Analytics
* New Media and Sentiment Mining
* Text Processing using Machine Learning
* Conversational UIs
* Vision Systems
* Spatial Reasoning from Sensor Data
* Real Time Audio-Visual Sensing and Sense Making

### Royal Holloway, University of London
> Bachelor of Science (B.S.), Marketing/Marketing Management, General
* MN2041K Managerial Accounting
* MN2061K Marketing Management
* MN2155K Asia Pacific Businesses
* MN2165K The Global Economy
* MN22201K Strategic Management
* MN3215K Asia Pacific Multinationals in Europe
* MN3455K Advertising And Promotion in Brand Marketing
* MN3495K Clusters, Small Business and International Competition
* MN3555K E-Commerce
* MN3035K Marketing Research
* MN3055K Consumer Behaviour
* MN3301K Modern Business in Comparative Perspective

### Guangzhou Civil Aviation College
> Associate's degree, Electrical and Electronics Engineering
* Further Mathematics
* College English
* Circuit Analysis
* Analog Electronic Technology
* Digital Electronic Technology
* Single-chip Microcomputer Design & Develop
* The C Programming Language
* Computer Network
* Digital Communication Theory
* Stored Program Control & Mobile Communication Theory


