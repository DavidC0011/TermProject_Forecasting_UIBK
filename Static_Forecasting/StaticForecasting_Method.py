# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 16:06:58 2022

@author: tareq
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt

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


pd.set_option('display.max_columns', None) 
pd.set_option('display.max_rows', 200)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'center')
pd.set_option('display.precision', 3)


def addRankingToTeams(match_df):
    ranking_df = pd.read_excel("../GermanLeagueData/Teams_Ranking.xlsx")
    
    #Initializing rankings
    match_df = match_df.assign(HomeTeamRanking=0)
    match_df = match_df.assign(AwayTeamRanking=0)
       
    for i in ranking_df.index: 
        match_df.loc[match_df["Home"] == ranking_df["NAME"][i], "HomeTeamRanking"] = ranking_df["OVERALL"][i]
        match_df.loc[match_df["Guest"] == ranking_df["NAME"][i], "AwayTeamRanking"] = ranking_df["OVERALL"][i]
    
    match_df = match_df[match_df["HomeTeamRanking"] != 0]
    match_df = match_df[match_df["AwayTeamRanking"] != 0]
    
    match_df["RankDifference"] = match_df["HomeTeamRanking"] - match_df["AwayTeamRanking"]
    match_df["AverageRank"] = (match_df["HomeTeamRanking"] + match_df["AwayTeamRanking"]) / 2
    match_df["PointsDifference"] = match_df["PointsHome"] - match_df["PointsGuest"]
    match_df["ScoreDifference"] = match_df["Score90Home"] - match_df["Score90Guest"]
    match_df['is_won'] = match_df['ScoreDifference'] > 0 # Take draw as lost
    
    match_df.to_excel("../GermanLeagueData/output.xlsx") 
    return match_df

def matchesDataCleansing(match_df):
    
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
  
   
  match_df['PointsGuest'] = np.where((match_df['SeasonFrom'] <= 1995) & (match_df['PointsGuest'] == 2), 3, match_df['PointsGuest'])
  match_df['PointsHome'] = np.where((match_df['SeasonFrom'] <= 1995) & (match_df['PointsHome'] == 2), 3, match_df['PointsHome'])
  return match_df
  
   
  
def getWinner(match_df, seasonStartYear): 
      seasonData = match_df.loc[match_df["SeasonFrom"] == seasonStartYear]  
      ranking_df = pd.read_excel("../GermanLeagueData/Teams_Ranking.xlsx")
      ranking_df["Season"] = seasonStartYear
      
      for _team in ranking_df.index:  
          print("===========================")
          print(_team)
          print("===========================")
          LogRegressionPointsNumber = 0 
          SvcPointsNumber           = 0
          KnnPointsNumber           = 0
          GaussianPointsNumber      = 0
          DecisionTreePointsNumber  = 0
          RandomForestPointsNumber  = 0
          
          
          for _index in seasonData.index: 
              if(seasonData["Home"][_index] == ranking_df["NAME"][_team]): 
                  
                  LogRegressionPointsNumber = LogRegressionPointsNumber + seasonData["LogPointsHome"][_index]
                  SvcPointsNumber           = SvcPointsNumber           + seasonData["SvcPointsHome"][_index]
                  KnnPointsNumber           = KnnPointsNumber           + seasonData["KnnPointsHome"][_index]
                  GaussianPointsNumber      = GaussianPointsNumber      + seasonData["GaussianPointsHome"][_index]
                  DecisionTreePointsNumber  = DecisionTreePointsNumber  + seasonData["DecisionTreePointsHome"][_index]
                  RandomForestPointsNumber  = RandomForestPointsNumber  + seasonData["RandomForestPointsHome"][_index]
                   
              elif(seasonData["Guest"][_index] == ranking_df["NAME"][_team]): 
                  
                  LogRegressionPointsNumber = LogRegressionPointsNumber + seasonData["LogPointsGuest"][_index]     
                  SvcPointsNumber           = SvcPointsNumber           + seasonData["SvcPointsGuest"][_index]
                  KnnPointsNumber           = KnnPointsNumber           + seasonData["KnnPointsGuest"][_index]
                  GaussianPointsNumber      = GaussianPointsNumber      + seasonData["GaussianPointsGuest"][_index]
                  DecisionTreePointsNumber  = DecisionTreePointsNumber  + seasonData["DecisionTreePointsGuest"][_index]
                  RandomForestPointsNumber  = RandomForestPointsNumber  + seasonData["RandomForestPointsGuest"][_index]
                    
          #saving the results
          ranking_df.loc[ranking_df["NAME"] == ranking_df["NAME"][_team], "LogRegressionTotalPoints"]      = LogRegressionPointsNumber
          ranking_df.loc[ranking_df["NAME"] == ranking_df["NAME"][_team], "SvcTotalPoints"]                = SvcPointsNumber
          ranking_df.loc[ranking_df["NAME"] == ranking_df["NAME"][_team], "KnnTotalPointsNumber"]          = KnnPointsNumber
          ranking_df.loc[ranking_df["NAME"] == ranking_df["NAME"][_team], "GaussianTotalPointsNumber"]     = GaussianPointsNumber
          ranking_df.loc[ranking_df["NAME"] == ranking_df["NAME"][_team], "DecisionTreeTotalPointsNumber"] = DecisionTreePointsNumber
          ranking_df.loc[ranking_df["NAME"] == ranking_df["NAME"][_team], "RandomForestTotalPointsNumber"] = RandomForestPointsNumber
           
      #ranking_df.to_excel("../GermanLeagueData/Teams_Ranking.xlsx") 
      #ranking_df = pd.read_excel("../GermanLeagueData/Teams_Ranking.xlsx")
      #print(match_df["Home"].unique())
      #print(match_df["Home"].value_counts())
      #print(match_df["Home"].value_counts(normalize=True)) 
      ranking_df.to_excel("../GermanLeagueData/Teams_Ranking.xlsx") 
      ranking_df = pd.read_excel("../GermanLeagueData/Teams_Ranking.xlsx")
      
      print("LogRegressionTotalPoints winner")
      maxPoints = ranking_df["LogRegressionTotalPoints"].max()
      print(ranking_df.loc[ranking_df["LogRegressionTotalPoints"] == maxPoints]) 
      #if(len(ranking_df.loc[ranking_df["LogRegressionTotalPoints"] == maxPoints])):
      #   drawTeams = ranking_df.loc[ranking_df["LogRegressionTotalPoints"] == maxPoints]
         
         
      print("SvcTotalPoints winner")
      maxPoints = ranking_df["SvcTotalPoints"].max()
      print(ranking_df.loc[ranking_df["SvcTotalPoints"] == maxPoints]) 
      
      print("KnnTotalPointsNumber winner")
      maxPoints = ranking_df["KnnTotalPointsNumber"].max()
      print(ranking_df.loc[ranking_df["KnnTotalPointsNumber"] == maxPoints]) 
       
      print("GaussianTotalPointsNumber winner")
      maxPoints = ranking_df["GaussianTotalPointsNumber"].max()
      print(ranking_df.loc[ranking_df["GaussianTotalPointsNumber"] == maxPoints]) 
       
      print("DecisionTreeTotalPointsNumber winner")
      maxPoints = ranking_df["DecisionTreeTotalPointsNumber"].max()
      print(ranking_df.loc[ranking_df["DecisionTreeTotalPointsNumber"] == maxPoints]) 
        
      print("RandomForestTotalPointsNumber winner")
      maxPoints = ranking_df["RandomForestTotalPointsNumber"].max()
      print(ranking_df.loc[ranking_df["RandomForestTotalPointsNumber"] == maxPoints]) 
        
      ranking_df.to_excel("../GermanLeagueData/Teams_Ranking.xlsx")   
        
def dataSetTraining(match_df):
     
    print(match_df)
    X, y = match_df.loc[:,['AverageRank', 'RankDifference']], match_df['is_won']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   
    # Logistic Regression 
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    lg_pred = logreg.predict(X_test)
    acc_log = round(logreg.score(X_test, y_test) * 100, 2)
    print(acc_log)
    
    # Support Vector Machines 
    svc = SVC(probability=True)
    svc.fit(X_train, y_train)
    svm_pred = svc.predict(X_test)
    acc_svc = round(svc.score(X_test, y_test) * 100, 2)
    print(acc_svc)
    
    # KNN
    knn = KNeighborsClassifier(n_neighbors = 3)
    knn.fit(X_train, y_train)
    Y_pred = knn.predict(X_test)
    acc_knn = round(knn.score(X_test, y_test) * 100, 2)
    print(acc_knn)
    
    # Gaussian Naive Bayes
    gaussian = GaussianNB()
    gaussian.fit(X_train, y_train)
    gnb_pred = gaussian.predict(X_test)
    acc_gaussian = round(gaussian.score(X_test, y_test) * 100, 2)
    print(acc_gaussian)
    
    # Decision Tree
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X_train, y_train)
    Y_pred = decision_tree.predict(X_test)
    acc_decision_tree = round(decision_tree.score(X_test, y_test) * 100, 2)
    print(acc_decision_tree)
    
    # Random Forest
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(X_train, y_train)
    Y_pred = random_forest.predict(X_test)
    random_forest.score(X_train, y_train)
    acc_random_forest = round(random_forest.score(X_test, y_test) * 100, 2)
    print(acc_random_forest)

    model = logreg
    print(match_df)
     
    #print((match_df.loc[match_df["HomeTeamRanking"] == 0]))
    
     
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
    
    # Define a small margin when we safer to predict draw then win
    margin = 0.05

    for i in match_df.index: 
        if(match_df["is_played"][i] == 0):
            home = match_df["Home"][i]
            away = match_df["Guest"][i]
            
            # Create a row for each match
            row = pd.DataFrame(np.array([[np.nan, np.nan]]), columns=X_train.columns)
            
            home_rank   = match_df["HomeTeamRanking"][i]
            home_points = match_df["PointsHome"][i]
            opp_rank    = match_df["AwayTeamRanking"][i]
            opp_points  = match_df["PointsGuest"][i] 
            
            row['AverageRank'] = match_df["AverageRank"][i]
            row['RankDifference'] = match_df["RankDifference"][i]
            #row['PointsDifference'] = prediction_df["PointsDifference"][i]
             
            # Model Output
            log_home_win_prob               = model.predict_proba(row)[:,1][0]
            svc_home_win_prob               = svc.predict_proba(row)[:,1][0]
            knn_home_win_prob               = knn.predict_proba(row)[:,1][0]
            gaussian_home_win_prob          = gaussian.predict_proba(row)[:,1][0]
            decision_tree_home_win_prob     = decision_tree.predict_proba(row)[:,1][0]
            random_forest_home_win_prob     = random_forest.predict_proba(row)[:,1][0]
            
            print(home)
            print(log_home_win_prob)
            match_df["log_home_winning_probability"][i] = log_home_win_prob
            match_df["svc_home_win_prob"][i]            = svc_home_win_prob 
            match_df["knn_home_win_prob"][i]            = knn_home_win_prob
            match_df["gaussian_home_win_prob"][i]       = gaussian_home_win_prob
            match_df["decision_tree_home_win_prob"][i]  = decision_tree_home_win_prob
            match_df["random_forest_home_win_prob"][i]  = random_forest_home_win_prob
            
            # Determining Win / Draw / Lose based on log_home_win_prob
            points = 0
            if log_home_win_prob <= 0.5 - margin:
                #in this case, away team wins
                match_df["LogPointsHome"][i] = 0
                match_df["LogPointsGuest"][i] = 3 
            elif ((log_home_win_prob > 0.5 - margin) & (log_home_win_prob < 0.5 + margin)):
                #draw
                match_df["LogPointsHome"][i] = 1
                match_df["LogPointsGuest"][i] = 1 
            elif log_home_win_prob >= 0.5 + margin:
                #in this case, home team wins
                match_df["LogPointsHome"][i] = 3
                match_df["LogPointsGuest"][i] = 0 
                
                
                
            # Determining Win / Draw / Lose based on svc_home_win_prob
            points = 0
            if svc_home_win_prob <= 0.5 - margin:
                #in this case, away team wins
                match_df["SvcPointsHome"][i] = 0
                match_df["SvcPointsGuest"][i] = 3 
            elif ((svc_home_win_prob > 0.5 - margin) & (svc_home_win_prob < 0.5 + margin)):
                #draw
                match_df["SvcPointsHome"][i] = 1
                match_df["SvcPointsGuest"][i] = 1 
            elif svc_home_win_prob >= 0.5 + margin:
                #in this case, home team wins
                match_df["SvcPointsHome"][i] = 3
                match_df["SvcPointsGuest"][i] = 0 
                
                
                
            # Determining Win / Draw / Lose based on knn_home_win_prob
            points = 0
            if knn_home_win_prob <= 0.5 - margin:
                #in this case, away team wins
                match_df["KnnPointsHome"][i] = 0
                match_df["KnnPointsGuest"][i] = 3 
            elif ((knn_home_win_prob > 0.5 - margin) & (knn_home_win_prob < 0.5 + margin)):
                #draw
                match_df["KnnPointsHome"][i] = 1
                match_df["KnnPointsGuest"][i] = 1 
            elif knn_home_win_prob >= 0.5 + margin:
                #in this case, home team wins
                match_df["KnnPointsHome"][i] = 3
                match_df["KnnPointsGuest"][i] = 0 
                
            
            # Determining Win / Draw / Lose based on gaussian_home_win_prob
            points = 0
            if gaussian_home_win_prob <= 0.5 - margin:
                #in this case, away team wins
                match_df["GaussianPointsHome"][i] = 0
                match_df["GaussianPointsGuest"][i] = 3 
            elif ((gaussian_home_win_prob > 0.5 - margin) & (gaussian_home_win_prob < 0.5 + margin)):
                #draw
                match_df["GaussianPointsHome"][i] = 1
                match_df["GaussianPointsGuest"][i] = 1 
            elif gaussian_home_win_prob >= 0.5 + margin:
                #in this case, home team wins
                match_df["GaussianPointsHome"][i] = 3
                match_df["GaussianPointsGuest"][i] = 0 
                
                
            # Determining Win / Draw / Lose based on decision_tree_home_win_prob
            points = 0
            if decision_tree_home_win_prob <= 0.5 - margin:
                #in this case, away team wins
                match_df["DecisionTreePointsHome"][i] = 0
                match_df["DecisionTreePointsGuest"][i] = 3 
            elif ((decision_tree_home_win_prob > 0.5 - margin) & (decision_tree_home_win_prob < 0.5 + margin)):
                #draw
                match_df["DecisionTreePointsHome"][i] = 1
                match_df["DecisionTreePointsGuest"][i] = 1 
            elif decision_tree_home_win_prob >= 0.5 + margin:
                #in this case, home team wins
                match_df["DecisionTreePointsHome"][i] = 3
                match_df["DecisionTreePointsGuest"][i] = 0 
                
                
            
            
            # Determining Win / Draw / Lose based on random_forest_home_win_prob
            points = 0
            if random_forest_home_win_prob <= 0.5 - margin:
                #in this case, away team wins
                match_df["RandomForestPointsHome"][i] = 0
                match_df["RandomForestPointsGuest"][i] = 3 
            elif ((random_forest_home_win_prob > 0.5 - margin) & (random_forest_home_win_prob < 0.5 + margin)):
                #draw
                match_df["RandomForestPointsHome"][i] = 1
                match_df["RandomForestPointsGuest"][i] = 1 
            elif random_forest_home_win_prob >= 0.5 + margin:
                #in this case, home team wins
                match_df["RandomForestPointsHome"][i] = 3
                match_df["RandomForestPointsGuest"][i] = 0 
                

    match_df.to_excel("../GermanLeagueData/output.xlsx") 
    
     
##### Reading the data
match_df = pd.read_excel("../GermanLeagueData/All_matches_history.xlsx")

match_df = matchesDataCleansing(match_df)
match_df = addRankingToTeams(match_df)

dataSetTraining(match_df)
getWinner(match_df, 2022)



