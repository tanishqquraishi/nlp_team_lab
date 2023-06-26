# **POS Tagger**

 * data.py:
  Read a file path, create token-POS pairs, extract features for tokens in a sentence.

  * perceptron.py: Contains model specs, parameters, returns predicted tags and scores
  
  
*  evaluation.py: Results in confusion matrix, per tag / class error, F1 scores, micro and macro scores.
  
 * features.py: Features desgined based on the Onto Notes dataset. 
  
 * main.py: Launch to train the model, prints training stages and evaluations after model is trained.
  
# **Goal**

  Build a POS tagger from scratch using a perceptron. 

# **Technologies**
  Python 
  
# **Launch**

  Run main.py to train the perceptron on a dataset. 
  
# **Project Status**

  Currently building domain adaptation techniques after the completion of the baseline.
  
# **Sources**

  	DATA: Weischedel, Ralph, et al. OntoNotes Release 5.0 LDC2013T19. Web Download. Philadelphia: Linguistic Data Consortium, 2013.
   
# **Note**

  All .txt files contain logs on train, dev and test set results after running the model. We also marked up some of the logs to find and track patterns of poor performance.
