# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 15:37:35 2019

@author: rahul
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize

def preprocessing():
        dataset1=pd.read_csv("Building_Ownership_Use.csv")
        dataset2=pd.read_csv("Building_Structure.csv")
        dataset3=pd.read_csv("train.csv")
        
        #Merging datasets into a single dataset
        print("columns in Ownership::::",dataset1.columns)
        print("columns in structure::::",dataset2.columns)
        print("columns in train::::",dataset3.columns)
        
        dataset_building_data=pd.merge(dataset1,dataset2,on="building_id")
        dataset_building_data=dataset_building_data.drop(["district_id_y"],axis=1)  
        dataset_building_data=dataset_building_data.drop(["ward_id_y"],axis=1)  
        dataset_building_data=dataset_building_data.drop(["vdcmun_id_y"],axis=1)
        dataset_building_data=dataset_building_data.drop(["vdcmun_id_x"],axis=1)
        dataset_building_data=dataset_building_data.drop(["ward_id_x"],axis=1)
        
        dataset_final=pd.merge(dataset_building_data,dataset3,on="building_id")
        
        #moving the dependant variable to the last column
        dataset_final=dataset_final.drop(["vdcmun_id"],axis=1)
        cols_to_move = ['damage_grade']
        new_cols = np.hstack((dataset_final.columns.difference(cols_to_move), cols_to_move))
        dataset_final = dataset_final.reindex(columns=new_cols)
        dataset_final=dataset_final.drop(["building_id"],axis=1)
        dataset_final=dataset_final.drop(["district_id_x"],axis=1)
        dataset_final=dataset_final.drop(["district_id"],axis=1)
        
        
        #save dataset 
        dataset_final.to_csv("final_dataset.csv")
    
    
        print("no. of columns",len(dataset_final.columns))
        print("columns in final dataset:",dataset_final.columns)
        

       
        total = dataset_final.isnull().sum().sort_values(ascending=False)
        percent = (dataset_final.isnull().sum()/dataset_final.isnull().count()).sort_values(ascending=False)
        missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        
        dataset_final['count_families'].fillna(dataset_final['count_families'].mode()[0],inplace=True)
        
        print(dataset_final['has_repair_started'].value_counts())
        dataset_final['has_repair_started'].fillna(dataset_final['has_repair_started'].mode()[0],inplace=True)
        
        print(dataset_final.columns.hasnans)
        
        Y = {'Grade 1': 1, 'Grade 2': 2, 'Grade 3': 3, 'Grade 4': 4, 'Grade 5': 5}
        dataset_final['damage_grade'].replace(Y, inplace = True)
        dataset_final['damage_grade'].unique()
        
        
        
        
        
        list1=list(dataset_final.columns.values)
        list2=[]
        i=0
        for cols in list1:
            list2.append((i,cols))
            i=i+1
       

        listcat= dataset_final.select_dtypes('object').nunique()      
        #Encoding of cateogorical variables
        dataset_f = dataset_final.copy()
        lb=LabelBinarizer()

        #The following indexes are the cateogircal variables present in the dataset
        for i in [1,2,6,7,41,42,43,44,46,47]:
            j=list2[i][1]
            print(j)
            
            lb_results = lb.fit_transform(dataset_f[j])
            lb_results_df = pd.DataFrame(lb_results, columns=lb.classes_)
            print(lb_results_df.head())
            dataset_f = pd.concat([dataset_f, lb_results_df], axis=1)
            
        print("Removing orginal cateogorical variables...")
        for i in [1,2,6,7,41,42,43,44,46,47]:
             j=list2[i][1]
             print(j)
             dataset_f=dataset_f.drop([j],axis=1)
         
        print(dataset_f.columns.hasnans)     
       
          
#        list3=list(dataset_f.columns.values)
##        list4=[]
##        i=0
##        for cols in list3 :
##            list4.append((i,cols))
##            i=i+1    
        dataset_f= dataset_f.loc[:,~dataset_f.columns.duplicated()]
        
        dataset_f['count_floors_change'] = (dataset_f['count_floors_post_eq']/dataset_f['count_floors_pre_eq'])
        dataset_f['height_ft_change'] = (dataset_f['height_ft_post_eq']/dataset_f['height_ft_pre_eq'])
        dataset_f.drop(['count_floors_post_eq', 'height_ft_post_eq'], axis=1, inplace=True)

        
        cols_to_move = ['damage_grade']
        new_cols = np.hstack((dataset_f.columns.difference(cols_to_move), cols_to_move))
        dataset_f = dataset_f.reindex(columns=new_cols)
        list3=list(dataset_f.columns.values)
        #dataset_f.to_csv("dataset_after_preprocess_.csv")
        colcat = dataset_f.dtypes
        #Splitting dataset into independant and dependant variables
        x=dataset_f.iloc[:,0:85].values
        y=dataset_f.iloc[:,85].values
        list4=[]
        for i in range(85):
            list4.append(np.amax(x[:,i]))
        sc_x=normalize(x[:,[47,48,82,83,84]])
        x=np.delete(x,[47,48,82,83,84],1)
        x=np.append(x,sc_x,axis=1) 
        
        if True in (np.isnan(x)):
            print("pre process")
        list5=[]    
        for i in range(0,85):
            list5.append(np.amax(x[:,i]))
        
        
        
        
#        from sklearn.ensemble import ExtraTreesClassifier
#        import matplotlib.pyplot as plt
#        model = ExtraTreesClassifier()
#        model.fit(x,y)
#        print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#        #plot graph of feature importances for better visualization
#        feat_importances = pd.Series(model.feature_importances_, index=list3[:-5])
#        feat_importances.nlargest(20).plot(kind='barh')
#        plt.show()
#        
#        list5=list(feat_importances.nlargest(20).index)     
#        fts=[]
#        for features in list5:
#            if features not in fts:
#                fts.append(features)
#        dataset_fii=dataset_f[fts]
#        dataset_fii = dataset_fii.loc[:,~dataset_fii.columns.duplicated()]
#        list6=[]
#        for i in range(0,len(dataset_fii.columns)):
#            list6.append(dataset_f.columns.get_loc(dataset_fii.columns[i]))
#       
#        x_f=x[:,list6]
#        
        #Splitting data into train and test
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
        return x_train,x_test,y_train,y_test 
        

                                                                             
          