# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 16:06:58 2022

@author: tareq
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#from matplotlib import pyplot as plt

from sklearn import linear_model
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
import warnings


# Control console printing settings, like displaying more rows and remove warnings from the console
pd.set_option('display.max_columns', None) 
pd.set_option('display.max_rows', 200)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'center')
pd.set_option('display.precision', 3)
warnings.filterwarnings("ignore")
#####################################


 
#This function cleans the data and prepare it to be used in the algorith
def matchesDataCleansing(match_df):
    
  #Correct teams' names encoding and updating teams' names
  match_df = match_df.replace(["1. FC KÃ¶ln"], "1. FC Köln")
  match_df = match_df.replace(["Bor. MÃ¶nchengladbach"], "Borussia Mönchengladbach")
  match_df = match_df.replace(["TSV 1860 MÃ¼nchen"], "TSV 1860 München")
  match_df = match_df.replace(["PreuÃŸen MÃ¼nster"], "SC Preußen Münster")
  match_df = match_df.replace(["1. FC SaarbrÃ¼cken"], "1. FC Saarbrücken")
  match_df = match_df.replace(["1. FC NÃ¼rnberg"], "1. FC Nürnberg")
  match_df = match_df.replace(["Bayern MÃ¼nchen"], "FC Bayern München")
  match_df = match_df.replace(["Fortuna DÃ¼sseldorf"], "Fortuna Düsseldorf")
  match_df = match_df.replace(["Rot-WeiÃŸ Oberhausen"], "Rot-Weiß Oberhausen")
  match_df = match_df.replace(["Fortuna KÃ¶ln"], "SC Fortuna Köln")
  match_df = match_df.replace(["SpVgg Blau-WeiÃŸ 90 Berlin"], "Blau-Weiß 1890 Berlin")
  match_df = match_df.replace(["SC Freiburg"], "Sport-Club Freiburg")
  match_df = match_df.replace(["Werder Bremen"], "SV Werder Bremen")
  match_df = match_df.replace(["VfL Bochum"], "VfL Bochum 1848")
  match_df = match_df.replace(["Fortuna DÃ¼sseldorf"], "Fortuna Düsseldorf")
  match_df = match_df.replace(["Arminia Bielefeld"], "DSC Arminia Bielefeld") 
  match_df = match_df.replace(["SpVgg Greuther FÃ¼rth"], "SpVgg Greuther Fürth")
  match_df = match_df.replace(["SpVgg Greuther FÃ¼rth"], "SpVgg Greuther Fürth")
  match_df = match_df.replace(["1. FC NÃ¼rnberg"], "1. FC Nürnberg")
  match_df = match_df.replace(["Hansa Rostock"], "F.C. Hansa Rostock")
  match_df = match_df.replace(["1. FC Dynamo Dresden"], "SG Dynamo Dresden")
  match_df = match_df.replace(["SV Waldhof Mannheim"], "SV Waldhof Mannheim 07")
  match_df = match_df.replace(["TSV 1860 MÃ¼nchen"], "TSV 1860 München")
  match_df = match_df.replace(["Borussia Dortmund"], "Borussia Dortmund II")
  match_df = match_df.replace(["SC Freiburg"], "SC Freiburg II")
  
  #Unifying the number of points given to the winning team before and after 1995 
  #We adopt that each winner is given 3 points, not 2
  match_df['PointsGuest'] = np.where((match_df['SeasonFrom'] <= 1995) & (match_df['PointsGuest'] == 2), 3, match_df['PointsGuest'])
  match_df['PointsHome'] = np.where((match_df['SeasonFrom'] <= 1995) & (match_df['PointsHome'] == 2), 3, match_df['PointsHome'])
  return match_df
  


# This function read the FIFA rankings and assign them to each team in the historical matches dataset
def addRankingToTeams(match_df):
    #Read the ranking
    ranking_df = pd.read_excel("GermanLeagueData/Teams_Ranking.xlsx")
    
    #Initializing two columns that store the rankings for each team
    match_df = match_df.assign(HomeTeamRanking=0)
    match_df = match_df.assign(AwayTeamRanking=0)
       
    #Assign FIFA overall ranking to each team
    for i in ranking_df.index: 
        match_df.loc[match_df["Home"] == ranking_df["NAME"][i], "HomeTeamRanking"] = ranking_df["OVERALL"][i]
        match_df.loc[match_df["Guest"] == ranking_df["NAME"][i], "AwayTeamRanking"] = ranking_df["OVERALL"][i]
    
    #If a team doesn't have ranking, remove it
    #We essentially do this because teams that have no up-to-date FIFA ranking have, for sure, stopped operating 
    #And that is why no further rankings have been published for them 
    match_df = match_df[match_df["HomeTeamRanking"] != 0]
    match_df = match_df[match_df["AwayTeamRanking"] != 0]
    
    
    #Calculate possible feature to be used in the algorithm, including RankDifference, AverageRank, PointsDifference and ScoreDifference
    match_df["RankDifference"] = match_df["HomeTeamRanking"] - match_df["AwayTeamRanking"]
    match_df["AverageRank"] = (match_df["HomeTeamRanking"] + match_df["AwayTeamRanking"]) / 2
    match_df["PointsDifference"] = match_df["PointsHome"] - match_df["PointsGuest"]
    match_df["ScoreDifference"] = match_df["Score90Home"] - match_df["Score90Guest"]
    
    #Here we add another attribute "is_won" to indicate if the home team wins or not
    match_df['is_won'] = match_df['ScoreDifference'] > 0
    
    #Write the output onto the hard drive
    match_df.to_excel("GermanLeagueData/output.xlsx") 
    return match_df





#This function contains the algorithms we used in this project
#The function predicts the winner for each matchin the leage and finally assigns points for each winning team
def dataSetTraining(match_df, seasonStartYear):
    # Load matches data that belong to seasons before the passed season "seasonStartYear" 
    match_df = match_df.loc[match_df["SeasonFrom"] <= seasonStartYear]
    # For the matches in the passed season "seasonStartYear", we assume that all matches did not take place yet so that we calculate their results, and that is done by setting "is_played" = 0
    match_df.loc[match_df["SeasonFrom"] == seasonStartYear, "is_played"] = 0 
    # X, y represent the features used in the algorithm, where x is the training features and y is the target feature
    X, y = match_df.loc[match_df["is_played"] != 0,['AverageRank', 'RankDifference']], match_df.loc[match_df["is_played"] != 0,['is_won']]
    # train_test_split() splits the dataset into training and test datasets 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
    
    # RandomForestClassifier is the class we use for Random Forest algorithm, where n_estimators is the maximum number of trees in the forest
    random_forest = RandomForestClassifier(n_estimators=100)
    # the fit() function builds our model  
    random_forest.fit(X_train, y_train)    
    # the function export_graphviz only renders the Random Forest tree into a .dot file, which can be visualized in the following website https://dreampuf.github.io/GraphvizOnline/
    export_graphviz(random_forest.estimators_[0], out_file='tree.dot', 
                feature_names = ['AverageRank', 'RankDifference'],
                class_names = True,
                rounded = True, proportion = False, 
                precision = 2, filled = True)
    
    
    # The score function tests the accuracy of the result using the testing dataset against the training dataset, and returns a precentage of accuracy
    accuracy_random_forest = round(random_forest.score(X_test, y_test) * 100, 2)
    print("accuracy_random_forest")
    print(accuracy_random_forest)
    
    
    
    
    # Y_pred = random_forest.predict(X_test)  
    # random_forest.score(X_train, y_train)
    # print(acc_random_forest)
    # print("classess_")
    # print(random_forest.classes_) 
    # from sklearn import metrics
    # Model Accuracy, how often is the classifier correct?
    # print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    # mean_squared_error(Y_pred, ytest)
    
    
    
    
    # Build and test Logistic Regression algorithm
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    lg_pred = logreg.predict(X_test)
    accuracy_logisticRegression = round(logreg.score(X_test, y_test) * 100, 2)
    print("accuracy_logisticRegression")
    print(accuracy_logisticRegression)
    
    # Build and test Support Vector Machines algorithm
    svc = SVC(probability=True)
    svc.fit(X_train, y_train)
    svm_pred = svc.predict(X_test)
    accuracy_svc = round(svc.score(X_test, y_test) * 100, 2)
    print("accuracy_svc")
    print(accuracy_svc)
    
    # Build and test KNN algorithm
    knn = KNeighborsClassifier(n_neighbors = 3)
    knn.fit(X_train, y_train)
    Y_pred = knn.predict(X_test)
    accuracy_knn = round(knn.score(X_test, y_test) * 100, 2)
    print("accuracy_knn")
    print(accuracy_knn)
    
    # Build and test Gaussian Naive Bayes algorithm
    gaussian = GaussianNB()
    gaussian.fit(X_train, y_train)
    gnb_pred = gaussian.predict(X_test)
    accuracy_gaussian = round(gaussian.score(X_test, y_test) * 100, 2)
    print("accuracy_gaussian")
    print(accuracy_gaussian)
    
    # Build and test Decision Tree algorithm
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X_train, y_train)
    Y_pred = decision_tree.predict(X_test)
    accuracy_decision_tree = round(decision_tree.score(X_test, y_test) * 100, 2)
    print("accuracy_decision_tree")
    print(accuracy_decision_tree)
    
    
     
    
    # Here we initialize the new columns to store the results in them
    # log_home_winning_probability stores the probability that the home team wins using Logistic Regression algorithm
    # LogPointsHome is the number of points the home team gets using Logistic Regression algorithm
    # LogPointsGuest is the number of points the away team gets using Logistic Regression algorithm
    # We did the same thing for the rest of the algorithms
    match_df['log_home_winning_probability'] = 0 
    match_df['LogPointsHome'] = match_df["PointsHome"]
    match_df['LogPointsGuest'] = match_df["PointsGuest"]
    
    match_df['svc_home_win_prob'] = 0 
    match_df['SvcPointsHome'] = match_df["PointsHome"]
    match_df['SvcPointsGuest'] = match_df["PointsGuest"] 
    
    match_df['knn_home_win_prob'] = 0 
    match_df['KnnPointsHome'] = match_df["PointsHome"] 
    match_df['KnnPointsGuest'] = match_df["PointsGuest"]
    
    match_df['gaussian_home_win_prob'] = 0 
    match_df['GaussianPointsHome'] = match_df["PointsHome"]
    match_df['GaussianPointsGuest'] = match_df["PointsGuest"] 
    
    match_df['decision_tree_home_win_prob'] = 0 
    match_df['DecisionTreePointsHome'] = match_df["PointsHome"]
    match_df['DecisionTreePointsGuest'] = match_df["PointsGuest"] 
    
    match_df['random_forest_home_win_prob'] = 0 
    match_df['RandomForestPointsHome'] = match_df["PointsHome"]
    match_df['RandomForestPointsGuest'] = match_df["PointsGuest"] 
     
    
    
    # The following margin is used to determine Draw matches
    # if the winning probability is in the range [0.5 - 0.05, 0.5 + 0.05] then the result is draw
    margin = 0.05
 
    # iterate over to-be-predicted matches and predict the result
    for i in match_df.index: 
        if(match_df["is_played"][i] == 0): # Only matches that did not take place yet
            home = match_df["Home"][i]
            away = match_df["Guest"][i] 
            # Create a row for each match
            row = pd.DataFrame(np.array([[np.nan, np.nan]]), columns=X_train.columns) 
            # Get the home and away teams rankingsa and points
            #home_rank   = match_df["HomeTeamRanking"][i]
            #home_points = match_df["PointsHome"][i]
            #opp_rank    = match_df["AwayTeamRanking"][i]
            #opp_points  = match_df["PointsGuest"][i] 
            
            # Initializing the features 
            row['AverageRank'] = match_df["AverageRank"][i]
            row['RankDifference'] = match_df["RankDifference"][i] 
             
            # Model Output
            # For each algorith, we get two results, the probability of winning and losing for the home team
            log_home_win_prob               = logreg.predict_proba(row)[:,1][0]
            svc_home_win_prob               = svc.predict_proba(row)[:,1][0]
            knn_home_win_prob               = knn.predict_proba(row)[:,1][0]
            gaussian_home_win_prob          = gaussian.predict_proba(row)[:,1][0]
            decision_tree_home_win_prob     = decision_tree.predict_proba(row)[:,1][0]
            random_forest_home_win_prob     = random_forest.predict_proba(row)[:,1][0] 
             
            
            # Store the winning prediction results for each algorithm
            match_df["log_home_winning_probability"][i] = log_home_win_prob
            match_df["svc_home_win_prob"][i]            = svc_home_win_prob 
            match_df["knn_home_win_prob"][i]            = knn_home_win_prob
            match_df["gaussian_home_win_prob"][i]       = gaussian_home_win_prob
            match_df["decision_tree_home_win_prob"][i]  = decision_tree_home_win_prob
            match_df["random_forest_home_win_prob"][i]  = random_forest_home_win_prob
            
            
            # Determining Win / Draw / Lose based on the winning probability using Logistic Regression (log_home_win_prob)
            points = 0
            if log_home_win_prob <= 0.5 - margin:
                #in this case, away team wins, so it receives 3 points
                match_df["LogPointsHome"][i] = 0
                match_df["LogPointsGuest"][i] = 3 
            elif ((log_home_win_prob > 0.5 - margin) & (log_home_win_prob < 0.5 + margin)):
                #draw, each teams receives 1 point
                match_df["LogPointsHome"][i] = 1
                match_df["LogPointsGuest"][i] = 1 
            elif log_home_win_prob >= 0.5 + margin:
                #in this case, home team wins
                match_df["LogPointsHome"][i] = 3
                match_df["LogPointsGuest"][i] = 0 
                
                
                
            # Determining Win / Draw / Lose based on the winning probability using SVC (svc_home_win_prob)
            points = 0
            if svc_home_win_prob <= 0.5 - margin:
                #in this case, away team wins, so it receives 3 points
                match_df["SvcPointsHome"][i] = 0
                match_df["SvcPointsGuest"][i] = 3 
            elif ((svc_home_win_prob > 0.5 - margin) & (svc_home_win_prob < 0.5 + margin)):
                #draw, each teams receives 1 point
                match_df["SvcPointsHome"][i] = 1
                match_df["SvcPointsGuest"][i] = 1 
            elif svc_home_win_prob >= 0.5 + margin:
                #in this case, home team wins, so it receives 3 points
                match_df["SvcPointsHome"][i] = 3
                match_df["SvcPointsGuest"][i] = 0 
                
                
                
            # Determining Win / Draw / Lose based on the winning probability using KNN (knn_home_win_prob)
            points = 0
            if knn_home_win_prob <= 0.5 - margin:
                #in this case, away team wins, so it receives 3 points
                match_df["KnnPointsHome"][i] = 0
                match_df["KnnPointsGuest"][i] = 3 
            elif ((knn_home_win_prob > 0.5 - margin) & (knn_home_win_prob < 0.5 + margin)):
                #draw, each team receives 1 point
                match_df["KnnPointsHome"][i] = 1
                match_df["KnnPointsGuest"][i] = 1 
            elif knn_home_win_prob >= 0.5 + margin:
                #in this case, home team wins, so it receives 3 points
                match_df["KnnPointsHome"][i] = 3
                match_df["KnnPointsGuest"][i] = 0 
                
            
            # Determining Win / Draw / Lose based on the winning probability using Gaussian algorithm (gaussian_home_win_prob)
            points = 0
            if gaussian_home_win_prob <= 0.5 - margin:
                #in this case, away team wins, so it receives 3 points
                match_df["GaussianPointsHome"][i] = 0
                match_df["GaussianPointsGuest"][i] = 3 
            elif ((gaussian_home_win_prob > 0.5 - margin) & (gaussian_home_win_prob < 0.5 + margin)):
                #draw, each team receives 1 point
                match_df["GaussianPointsHome"][i] = 1
                match_df["GaussianPointsGuest"][i] = 1 
            elif gaussian_home_win_prob >= 0.5 + margin:
                #in this case, home team wins, so it receives 3 points
                match_df["GaussianPointsHome"][i] = 3
                match_df["GaussianPointsGuest"][i] = 0 
                
                
            # Determining Win / Draw / Lose based on the winning probability using Decision Tree algorithm (decision_tree_home_win_prob)
            points = 0
            if decision_tree_home_win_prob <= 0.5 - margin:
                #in this case, away team wins, so it receives 3 points
                match_df["DecisionTreePointsHome"][i] = 0
                match_df["DecisionTreePointsGuest"][i] = 3 
            elif ((decision_tree_home_win_prob > 0.5 - margin) & (decision_tree_home_win_prob < 0.5 + margin)):
                #draw, each team receives 1 point
                match_df["DecisionTreePointsHome"][i] = 1
                match_df["DecisionTreePointsGuest"][i] = 1 
            elif decision_tree_home_win_prob >= 0.5 + margin:
                #in this case, home team wins, so it receives 3 points
                match_df["DecisionTreePointsHome"][i] = 3
                match_df["DecisionTreePointsGuest"][i] = 0 
                
                 
            
            # Determining Win / Draw / Lose based on the winning probability using Random Forest algorithm (random_forest_home_win_prob)
            points = 0
            if random_forest_home_win_prob <= 0.5 - margin:
                #in this case, away team wins, so it receives 3 points
                match_df["RandomForestPointsHome"][i] = 0
                match_df["RandomForestPointsGuest"][i] = 3 
            elif ((random_forest_home_win_prob > 0.5 - margin) & (random_forest_home_win_prob < 0.5 + margin)):
                #draw, each team receives 1 point
                match_df["RandomForestPointsHome"][i] = 1
                match_df["RandomForestPointsGuest"][i] = 1 
            elif random_forest_home_win_prob >= 0.5 + margin:
                #in this case, home team wins, so it receives 3 points
                match_df["RandomForestPointsHome"][i] = 3
                match_df["RandomForestPointsGuest"][i] = 0 
                
    # write the results onto the hard disk
    match_df.to_excel("GermanLeagueData/output.xlsx") 
    return match_df


# This function simply prints an ordered list of the teams and their predicted and real number of points, starting from the winner
def getOrderedTeams(seasonStartYear):
    # Read the results
    match_df = pd.read_excel("GermanLeagueData/output.xlsx")
    # Get the required season's data, specified in "seasonStartYear" parameter
    seasonData = match_df.loc[match_df["SeasonFrom"] == seasonStartYear]  
    # Initialize a new dataframe to store the result in, and create three columns, TeamName, ActualPoints and TotalPoints
    OrderedTeamsDataFrame = pd.DataFrame()
    OrderedTeamsDataFrame = OrderedTeamsDataFrame.assign(TeamName="")
    OrderedTeamsDataFrame = OrderedTeamsDataFrame.assign(ActualPoints=0)
    OrderedTeamsDataFrame = OrderedTeamsDataFrame.assign(TotalPoints=0) 
    OrderedTeamsDataFrame = OrderedTeamsDataFrame.set_index(['TeamName'])
    
    # Assign teams names to the newly created dataframe
    for _match in seasonData.index:  
        OrderedTeamsDataFrame = pd.concat(
                [OrderedTeamsDataFrame, 
                  pd.DataFrame(
                      { 
                        "TeamName": seasonData["Home"][_match],
                        "ActualPoints":0,
                        "TotalPoints":0
                      } , index=[seasonData["Home"][_match]]
                  )
                  ]
            )
        OrderedTeamsDataFrame = pd.concat(
                [OrderedTeamsDataFrame, 
                  pd.DataFrame(
                      { 
                        "TeamName": seasonData["Guest"][_match],
                        "ActualPoints":0,
                        "TotalPoints":0
                      } , index=[seasonData["Guest"][_match]]
                  )
                  ]
            ) 
    print(OrderedTeamsDataFrame)  
    # If some neames are repeated, delete the copies and keep one copy only
    OrderedTeamsDataFrame = OrderedTeamsDataFrame.drop_duplicates() 
    # Predicted number of points
    numberOfPoints = 0  
    # Actual number of points
    actualNumberOfPoints = 0
    
    # Just iterate over the teams in this season and sum their number of points
    for _team in OrderedTeamsDataFrame.index:  
        for _match in seasonData.index:  
            if(seasonData["Home"][_match] == OrderedTeamsDataFrame["TeamName"][_team]):
                numberOfPoints = numberOfPoints + seasonData["RandomForestPointsHome"][_match]
                actualNumberOfPoints = actualNumberOfPoints + seasonData["PointsHome"][_match]
            if(seasonData["Guest"][_match] == OrderedTeamsDataFrame["TeamName"][_team]):
                numberOfPoints = numberOfPoints + seasonData["RandomForestPointsGuest"][_match]
                actualNumberOfPoints = actualNumberOfPoints + seasonData["PointsGuest"][_match]
        OrderedTeamsDataFrame["TotalPoints"][_team] = numberOfPoints
        OrderedTeamsDataFrame["ActualPoints"][_team] = actualNumberOfPoints
        # Sort the values based on the number of points received
        OrderedTeamsDataFrame = OrderedTeamsDataFrame.sort_values(by=["TotalPoints"], ascending=False)
        # resetting the values for the next iteration
        numberOfPoints = 0
        actualNumberOfPoints = 0
    #Write the result onto the hard disk
    OrderedTeamsDataFrame.to_excel("GermanLeagueData/OrderedTeamsDataFrame.xlsx") 
    print(OrderedTeamsDataFrame)
     
    
    
    
##### The code runs starting from this point
##### First, we read the data and then call our functions    
##### Reading the data
##### Please put all excel files with the same folder where the code is stored so that you don't get an error when running the code
match_df = pd.read_excel("GermanLeagueData/All_matches_history.xlsx") 
match_df = matchesDataCleansing(match_df)
match_df = addRankingToTeams(match_df) 
# Here you can specify any season you want. For testing, we specified 2021 season
match_df = dataSetTraining(match_df, 2021)
getOrderedTeams(2021)




