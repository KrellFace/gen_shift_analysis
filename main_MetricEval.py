from datetime import *
from enum import Enum, auto
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import math
import os.path
import random
import seaborn as sns; sns.set_theme()
from sklearn import preprocessing
from sklearn.manifold import TSNE
import metricEvalMethods as methods
import marioFunc as marioFunc
import isovistFunc as isoF
import ERAScatterGeneration as scatterGen
import dbscan as db

from sklearn.decomposition import PCA

#MARIO METRIC FILES
short_mario_inputFile = 'inputdata/FullData_10LevelsPerGen.csv'
main_mario_inputFile = 'inputdata/FullData_AllLevels.csv'

#ISOVIST FILES
isovist_fulldata = 'inputdata/Isovist Raw Data.xlsx'
assembled_isovist_data = 'inputdata/all_maps_isovist_combined_data.csv'
isovist_pca_data = 'inputdata/Isovist_PCA_Data.csv'
test_matched_pairs = 'inputdata/Base_Settlement_1_MatchedPairs.csv'
test_matched_pairs_1000 = 'inputdata/TEST_Base_Settlement_1_MatchedPairs_1000.csv'
highres_pca_data = 'inputdata/highres_pca_data.csv'

basemap_settlement6_matched_pairs = 'inputdata/BaseMap_Settlement_6_MatchedPairs.csv'

hybrid_only_pca_data = 'inputdata/hybrid_only_pcadata.csv'
volcano_only_pca_data = 'inputdata/volcano_only_pcadata.csv'

#CONFIG
buck_cnt = 20
out_folder = 'Volcano_Settlement6'


overall_start_time = 0


def sort_datapoints(input_df, map_name_a, map_name_b, x_axis_col, y_axis_col, loop_limit):

    #First, sort df to only contain the values determining x and y coords
    #coord_df = input_df[[x_axis_col,y_axis_col]].copy

    #Create map filtered dfs 
    map_a_df =  input_df.loc[(input_df['MapName'] == map_name_a)]
    map_b_df =  input_df.loc[(input_df['MapName'] == map_name_b)]

    #Filter the maps to coord columns only
    map_a_coord_df = map_a_df[['Isovist_ID', x_axis_col,y_axis_col]].copy()
    map_b_coord_df = map_b_df[['Isovist_ID', x_axis_col,y_axis_col]].copy()

    map_a_coord_df['MapAppearance'] = 'Both'
    map_b_coord_df['MapAppearance'] = 'Both'


    points_evaled = 0
    aloops = 0
    bloops = 0

    for index_a, row_a in map_a_coord_df.iterrows():
        proximal_b_found = False
        
        for index_b, row_b in map_b_coord_df.iterrows():
            points_evaled+=1
            aloops+=1
            if(points_evaled%10000==0):
                print(str(points_evaled)+" isovists sorted")
            dist = math.dist([row_a[x_axis_col],row_a[y_axis_col]],[row_b[x_axis_col],row_b[y_axis_col]])
            if(dist<0.1):
                proximal_b_found=True
                print("Proximal found")
                break
        if not proximal_b_found:
                map_a_coord_df.at[index_a,'MapAppearance'] = 'a_only'
        
        #Limit the amount processed during testing
        #if(aloops>loop_limit):
        #    break
            
    for index_b, row_b in map_b_coord_df.iterrows():
        proximal_a_found = False
        
        for index_a, row_a in map_a_coord_df.iterrows():
            points_evaled+=1
            bloops+=1
            if(points_evaled%10000==0):
                print(str(points_evaled)+ " isovists sorted")
            dist = math.dist([row_a[x_axis_col],row_a[y_axis_col]],[row_b[x_axis_col],row_b[y_axis_col]])
            if(dist<0.1):
                proximal_a_found=True
                print("Proximal found")
                break
        if not proximal_a_found:
            #row_a['MapAppearance'] = 'b_only'
            map_b_coord_df.at[index_b,'MapAppearance'] = 'b_only'
        
        
        #Limit the amount processed during testing
        #if(bloops>loop_limit):
        #    break

    return pd.concat([map_a_coord_df, map_b_coord_df], ignore_index=True, axis=0)
    
def find_location_matched_isovists(input_df, map_name_a, map_name_b, x_axis_col, y_axis_col, z_axis_col, row_fraction_to_check = 0.05):
    
    map_a_df =  input_df.loc[(input_df['MapName'] == map_name_a)]
    map_b_df =  input_df.loc[(input_df['MapName'] == map_name_b)]

    coordinate_tuples = list()
    
    loops = 0
    pairs_found = 0
    outer_index_counter = 0

    for index_a, row_a in map_a_df.iterrows():
        
        #Only check every 100th index A row to save on time
        #if(outer_index_counter%100==0):

        #Only check every 2% of rows to save on time
        if(random.random()<row_fraction_to_check):

            for index_b, row_b in map_b_df.iterrows():
                loops+=1
                if(loops%100000==0):
                    print(str(loops)+" pairs checked")

                
                dist = math.dist([row_a[x_axis_col],row_a[y_axis_col],row_a[z_axis_col]],[row_b[x_axis_col],row_b[y_axis_col],row_b[z_axis_col]])
                if(dist<2):
                    pairs_found+=1
                    #print("Co-located pair found: " + row_a['MapName'] + str(row_a['Isovist_ID']) + " and " + row_b['MapName'] +  str(row_b['Isovist_ID']))
                    coordinate_tuples.append([row_a['MapName'], row_a['Isovist_ID'],row_a[x_axis_col],row_a[y_axis_col],row_a[z_axis_col],row_b['MapName'],row_b['Isovist_ID'],row_b[x_axis_col],row_b[y_axis_col],row_b[z_axis_col]])
                    break
            
            #if(pairs_found>100000):
                #print("Found 100 pairs")
                #break
        outer_index_counter+=1
    
    return pd.DataFrame(coordinate_tuples, columns = ["Map_A_Name", "Map_A_Isovist", "Row_A_X","Row_A_Y","Row_A_Z", "Map_B_Name", "Map_B_Isovist","Row_B_X","Row_B_Y","Row_B_Z"])

def generate_arrows_from_matched_isovists(loc_matches_df, pca_data_df, arrow_fraction_to_add = 1):
    
    arrow_counter = 0
    arrow_coords = []
    for index_a, row_a in loc_matches_df.iterrows():
        arrow_counter+=1
        ##Only add every Xth arrow
        #if arrow_counter%arrow_fraction_to_add == 0:
                
        from_pca1 = pca_data_df.loc[(pca_data_df['MapName'] == row_a["Map_A_Name"])&(pca_data_df['Isovist_ID'] == row_a["Map_A_Isovist"]),'PCA 1'].iloc[0]
        from_pca2 = pca_data_df.loc[(pca_data_df['MapName'] == row_a["Map_A_Name"])&(pca_data_df['Isovist_ID'] == row_a["Map_A_Isovist"]),'PCA 2'].iloc[0]
        to_pca1 = pca_data_df.loc[(pca_data_df['MapName'] == row_a["Map_B_Name"])&(pca_data_df['Isovist_ID'] == row_a["Map_B_Isovist"]),'PCA 1'].iloc[0] - from_pca1
        to_pca2 = pca_data_df.loc[(pca_data_df['MapName'] == row_a["Map_B_Name"])&(pca_data_df['Isovist_ID'] == row_a["Map_B_Isovist"]),'PCA 2'].iloc[0] - from_pca2

        arrow_coords.append([from_pca1,from_pca2,to_pca1,to_pca2])

            #print("Arrow added")
    
    return arrow_coords

def find_biggest_isovist_shifts(loc_matches_df, pca_data_df, n_shifts):
    biggest_shifts = []
    for index_a, row_a in loc_matches_df.iterrows():
                
        from_pca1 = pca_data_df.loc[(pca_data_df['MapName'] == row_a["Map_A_Name"])&(pca_data_df['Isovist_ID'] == row_a["Map_A_Isovist"]),'PCA 1'].iloc[0]
        from_pca2 = pca_data_df.loc[(pca_data_df['MapName'] == row_a["Map_A_Name"])&(pca_data_df['Isovist_ID'] == row_a["Map_A_Isovist"]),'PCA 2'].iloc[0]

        #to_pca1 = pca_data_df.loc[(pca_data_df['MapName'] == row_a["Map_B_Name"])&(pca_data_df['Isovist_ID'] == row_a["Map_B_Isovist"]),'PCA 1'].iloc[0] - from_pca1
        #to_pca2 = pca_data_df.loc[(pca_data_df['MapName'] == row_a["Map_B_Name"])&(pca_data_df['Isovist_ID'] == row_a["Map_B_Isovist"]),'PCA 2'].iloc[0] - from_pca2

        to_pca1 = pca_data_df.loc[(pca_data_df['MapName'] == row_a["Map_B_Name"])&(pca_data_df['Isovist_ID'] == row_a["Map_B_Isovist"]),'PCA 1'].iloc[0]
        to_pca2 = pca_data_df.loc[(pca_data_df['MapName'] == row_a["Map_B_Name"])&(pca_data_df['Isovist_ID'] == row_a["Map_B_Isovist"]),'PCA 2'].iloc[0]

        distance = math.dist([from_pca1, from_pca2],[to_pca1, to_pca2])
        
        if(len(biggest_shifts)<n_shifts):
            biggest_shifts.append([index_a, distance, from_pca1, from_pca2, to_pca1, to_pca2])
        else:
            smallest_ind = 0
            smallest_val = 1000
            for index, shift in enumerate(biggest_shifts):
                if shift[1]<smallest_val:
                    smallest_ind = index
                    smallest_val = shift[1]
            if smallest_val < distance:
                biggest_shifts.pop(smallest_ind)
                biggest_shifts.append([index_a, distance, from_pca1, from_pca2, to_pca1, to_pca2])
    
    print(f"Biggest shifts:{biggest_shifts}")

    big_index = []
    for s in biggest_shifts:
        big_index.append(s[0])

    
    biggest_shift_locmatches = loc_matches_df.iloc[big_index]

    print(f"Biggest shift loc matches:{biggest_shift_locmatches.head}")
        

    output_frame = loc_matches_df.iloc[big_index]

    output_frame['PCADistance'] = -1
    output_frame['MapA_PCA1'] = -1
    output_frame['MapA_PCA2'] = -1
    output_frame['MapB_PCA1'] = -1
    output_frame['MapB_PCA2'] = -1

    #print(biggest_shifts)

    for shift in biggest_shifts:
        #print([shift[1:]])
        output_frame.at[shift[0],'PCADistance'] = shift[1]
        output_frame.at[shift[0],'MapA_PCA1'] = shift[2]
        output_frame.at[shift[0],'MapA_PCA2'] = shift[3]
        output_frame.at[shift[0],'MapB_PCA1'] = shift[4]
        output_frame.at[shift[0],'MapB_PCA2'] = shift[5]
                        
    
    return output_frame, biggest_shift_locmatches

def find_most_isolated_pca(pca_data_df_from, pca_data_df_to, metric_df, n_points, same_map = False):
    most_distant = []

    for index_from, row_from in pca_data_df_from.iterrows():

        min_dist = 10000
        index_to_nearest = 0
        from_pca1 = row_from['PCA 1']
        from_pca2 = row_from['PCA 2']
        
        for index_to, row_to in pca_data_df_to.iterrows():
            if(index_from!=index_to or not same_map):
                to_pca1 = row_to['PCA 1']
                to_pca2 = row_to['PCA 2']
                dist = math.dist([from_pca1, from_pca2],[to_pca1, to_pca2])
                if(dist<min_dist):
                    min_dist = dist
                    index_to_nearest = index_to
        

        if(len(most_distant)<n_points):
            most_distant.append([index_from, index_to_nearest, min_dist])
            print(f"adding {index_from},{index_to_nearest},{min_dist} as less than {n_points} found")
        else:
            smallest_stored_ind = -1
            smallest_stored_val = 1000
            for index, distant_data in enumerate(most_distant):
                if distant_data[2]<smallest_stored_val:
                    biggest_ind = index
                    smallest_stored_val = distant_data[2]
            if smallest_stored_val < min_dist:
                
                print(f"adding {index_from},{index_to_nearest},{min_dist} as distance bigger than {most_distant[biggest_ind]} which is currently stored")
                most_distant.pop(biggest_ind)
                most_distant.append([index_from,index_to_nearest, min_dist])
    
    distant_indices = []
    for val in most_distant:
        distant_indices.append(val[0])

    print("Most distant indices")
    print(distant_indices)
    
    
    #distant_locations = pca_data_df.iloc[distant_indices]
    #distant_locations = pca_data_df.index.isin(distant_indices)
    distant_locations =  pca_data_df_from.filter(items = distant_indices, axis=0)

    print(distant_locations.head)
    
    distant_locations['XPos'] = -1
    distant_locations['YPos'] = -1
    distant_locations['ZPos'] = -1

    for index, row in distant_locations.iterrows():
         
        distant_locations.at[index, 'XPos'] = metric_df.loc[(metric_df['MapName'] == row["MapName"])&(metric_df['Isovist_ID'] == row["Isovist_ID"]),'XPos'].iloc[0]
        distant_locations.at[index, 'YPos'] = metric_df.loc[(metric_df['MapName'] == row["MapName"])&(metric_df['Isovist_ID'] == row["Isovist_ID"]),'YPos'].iloc[0]
        distant_locations.at[index, 'ZPos'] = metric_df.loc[(metric_df['MapName'] == row["MapName"])&(metric_df['Isovist_ID'] == row["Isovist_ID"]),'ZPos'].iloc[0]

                
    return distant_locations


def mario_combinedset_analysis(inputfile):

    df_metricdata = pd.read_csv(inputfile)

    #Compile metrics to be used in analysis. Playability excluded as it is used as our fitness heuristic
    full_mets = []
    for metric in marioFunc.enum_MarioMetrics:
        if metric != marioFunc.enum_MarioMetrics.Playability:
            full_mets.append(metric)
    full_mets = []
    for metric in marioFunc.enum_MarioMetrics:
        if metric != marioFunc.enum_MarioMetrics.Playability:
            full_mets.append(metric)
    
    
    #methods.gen_overalldata_only(df_metricdata, full_mets, buck_cnt, 'output/' +out_folder +'/',"Generator","LevelName", True, marioFunc.enum_MarioMetrics.Playability)
    
    methods.gen_selection_criteria_information_for_dataset(df_metricdata, full_mets, buck_cnt, 'output/' +out_folder +'/',"Generator","LevelName",False, marioFunc.enum_MarioGenerators, using_fitness = True, fitness_metric = marioFunc.enum_MarioMetrics.Playability)

def isovist_combined_set_analysis(inputfile, use_fitness = False, fitness_metric = isoF.enum_IsovistMetrics.Diversity):

    full_isovist_df = isoF.create_combined_df_all_isovist_sheets(inputfile)

    print("Isovist data compiled into dataframe")

    methods.gen_selection_criteria_information_for_dataset(full_isovist_df, isoF.get_isovist_metrics_list(use_fitness, fitness_metric), buck_cnt, 'output/' +out_folder +'/',"MapName","Isovist_ID",False, [], use_fitness, fitness_metric)

def apply_pca_to_isovist_data(isovist_df, save_scaled_data = False, saved_data_path = "", save_pca_data = False, pca_data_path = ""):
    #isovist_df = isoF.create_combined_df_all_isovist_sheets(iso_data)
    data_only = isovist_df[["Area","Perimeter","Diversity","var_Radials","mean_Radials","Roundness","Openness", "Clutter","Reachability","Occlusivity","DriftLength","VistaLength","RealPerimeterSize"]].copy()

    
    data_scaled = pd.DataFrame(preprocessing.scale(data_only),columns = data_only.columns)

    np_iso_data = data_scaled.to_numpy()

    pca = PCA(n_components=2)
    projectedValues = pca.fit_transform(np_iso_data)
    varExplained = pca.explained_variance_ratio_ 

    print("Variance explained")
    print(varExplained)
    if(save_pca_data):
        pd.DataFrame(varExplained, index = ['PC-1','PC-2']).to_csv(pca_data_path+"/varianceExplained.csv")

    print("PCA Component Linear Metric Relations")
    lin_met_data = pd.DataFrame(pca.components_,columns=data_only.columns,index = ['PC-1','PC-2'])
    print(lin_met_data)
    if(save_pca_data):
        lin_met_data.to_csv(pca_data_path+"/PCA_Linear_Metric_Data.csv")

    pca_df = pd.DataFrame(projectedValues, columns = ['PCA 1', 'PCA 2'])

    #iso_id = isovist_df['Isovist_ID']
    #pca_df.insert(0,"Isovist_ID", iso_id)
    #mapnames = isovist_df['MapName']
    #pca_df.insert(1,"MapName", mapnames)

    #print("OG Iso DF")
    #print(isovist_df.head)
    #print("New PCA DF - Pre new Cols")
    #print(pca_df.head)

    #print("Isovist df head before being used for assignment to PCA data")
    #print(isovist_df.head)


    pca_df['Isovist_ID'] = isovist_df['Isovist_ID']
    pca_df['MapName'] = isovist_df['MapName']

    if(save_scaled_data):
        data_scaled['Isovist_ID'] = isovist_df['Isovist_ID']
        data_scaled['MapName'] = isovist_df['MapName']
        data_scaled.to_csv(saved_data_path)

    
    #print("PCA df head after assignment from iso data")
    #print(pca_df.head)


    #print("Isovist PCA Head:")
    #print(pca_df.head)

    return pca_df

def apply_tsne_to_isovist_data(isovist_df):
    #isovist_df = isoF.create_combined_df_all_isovist_sheets(iso_data)
    data_only = isovist_df[["Area","Perimeter","Diversity","var_Radials","mean_Radials","Roundness","Openness", "Clutter","Reachability","Occlusivity","DriftLength","VistaLength","RealPerimeterSize"]].copy()

    
    data_scaled = pd.DataFrame(preprocessing.scale(data_only),columns = data_only.columns)

    np_iso_data = data_scaled.to_numpy()

    #pca = PCA(n_components=2)
    #projectedValues = pca.fit_transform(np_iso_data)
    #varExplained = pca.explained_variance_ratio_ 

    
    tsne = TSNE(n_components=2, n_iter=250, random_state=42)
    tsne.fit(np_iso_data)
    projectedValues = tsne.fit_transform(np_iso_data) 
    varExplained = []

    print("Variance explained")
    print(varExplained)

    tsne_df = pd.DataFrame(projectedValues, columns = ['TSNE 1', 'TSNE 2'])

    #iso_id = isovist_df['Isovist_ID']
    #pca_df.insert(0,"Isovist_ID", iso_id)
    #mapnames = isovist_df['MapName']
    #pca_df.insert(1,"MapName", mapnames)

    #print("OG Iso DF")
    #print(isovist_df.head)
    #print("New PCA DF - Pre new Cols")
    #print(pca_df.head)
    


    tsne_df['Isovist_ID'] = isovist_df['Isovist_ID']
    tsne_df['MapName'] = isovist_df['MapName']

    #print("Isovist PCA Head:")
    #print(pca_df.head)

    return tsne_df

def visualise_compressed_data_all_maps(pca_df, vis_folder, feature_1, feature_2):

    pca_scatter_folder = 'output/' +vis_folder

    if not os.path.exists(pca_scatter_folder):
        os.makedirs(pca_scatter_folder)

    #pca_df = pd.read_csv(isovist_pca_data)

    #PCA VISUAL ON OVERALL SET
    scatterGen.generate_era_scatterplot(pca_df, feature_1, feature_2, (pca_scatter_folder +"/PCA_Visual-Scatterplot"), "Full Set PCA", "MapName", True, isoF.get_isovist_sheet_names_list())

    #PCA VISUAL FOR TWO MAPS ONLY
    #twomaps_df = pca_df.loc[(pca_df['MapName'] == 'outputsHybrid2') | (pca_df['MapName'] == 'BaseMap')  | (pca_df['MapName'] == 'Settlement_16')]
    #scatterGen.generate_era_scatterplot(twomaps_df, "PCA 1", "PCA 2", ('output/' +out_folder +"/PCA_Visual-Scatterplot_TwoSheetsOnly"), "MapName", True, ['outputsHybrid2','BaseMap', 'Settlement_16'])

    #Visualise all maps individually
    map_names = isoF.get_isovist_sheet_names_list()
    for name in map_names:
        map_data_only = pca_df.loc[(pca_df['MapName'] == name)]
        scatterGen.generate_era_scatterplot(map_data_only, feature_1, feature_2, (pca_scatter_folder +"/PCA_Plot_"+name), f"{name} Only", "MapName", True, [name])
        
    basemap_df = pca_df.loc[(pca_df['MapName'] == 'BaseMap')]
    scatterGen.generate_era_scatterplot(basemap_df, feature_1, feature_2, (pca_scatter_folder +"/PCA_Visual-Scatterplot_BaseMap"), "Volcano Only", "MapName", True, ['BaseMap'])

    hybrid_df = pca_df.loc[(pca_df['MapName'] == 'HybridMap')]
    scatterGen.generate_era_scatterplot(hybrid_df, feature_1, feature_2, (pca_scatter_folder +"/PCA_Visual-Scatterplot_HybridMap"),"Hybrid Only", "MapName", True, ['HybridMap'])

    #PCA VISUAL FOR Base and Hybrid
    twomaps_df = pca_df.loc[(pca_df['MapName'] == 'BaseMap') | (pca_df['MapName'] == 'HybridMap')]
    scatterGen.generate_era_scatterplot(twomaps_df, feature_1, feature_2, (pca_scatter_folder +"/PCA_Visual-Scatterplot_HybridAndBase"),"Hybrid and Base Scatter", "MapName", True, ['BaseMap', 'HybridMap'])




    #ALL ERA ONLY ON SPECIFIED CLUSTER

def generate_all_era_plots_for_selected_pca_cluster(run_folder, map_a_name, map_b_name, x1, y1, x2, y2):
    full_isovist_df = pd.read_csv(assembled_isovist_data)
    print(f"Iso Head: {full_isovist_df.head}")
    two_map_df = full_isovist_df.loc[(full_isovist_df['MapName'] == map_a_name) | (full_isovist_df['MapName'] == map_b_name)]
    two_map_df['InCluster'] = False
    print(f"Two Map df Head: {two_map_df.head}")

    pca_data = pd.read_csv(isovist_pca_data)
    two_map_pca_df = pca_data.loc[(pca_data['MapName'] == map_a_name) | (pca_data['MapName'] == map_b_name)]

    #print(f"Two Map pca Head: {two_map_pca_df.head}")

    #cluster_only_pca_df = two_map_pca_df.loc[(two_map_pca_df['PCA 1'] >= x1) & (two_map_pca_df['PCA 1'] <= x2)& (two_map_pca_df['PCA 2'] >= y1)& (two_map_pca_df['PCA 2'] <= y2)]
    cluster_only_pca_df = two_map_pca_df
    
    #print(f"Cluster only  pca Head: {cluster_only_pca_df.head}")

    scatterGen.generate_era_scatterplot(cluster_only_pca_df, 'PCA 1', 'PCA 2', (run_folder +"/PCA_ClusterOnlyERAPlot"), "Cluster Only", "MapName", True, [map_a_name, map_b_name])

    #print(cluster_only_pca_df.head)

    #print(two_map_df.head)

    for index_a, row_a in cluster_only_pca_df.iterrows():
        two_map_df.loc[((two_map_df['MapName'] == row_a["MapName"])&(two_map_df['Isovist_ID'] == row_a["Isovist_ID"])),'InCluster'] = True

    full_cluster_data = two_map_df.loc[(two_map_df['InCluster'] == True)]

    met_names = isoF.get_isovist_metric_names_list()

    for met_a in met_names:
        for met_b in met_names:          
            scatterGen.generate_era_scatterplot(full_cluster_data, met_a, met_b, (f"{run_folder}/{met_a},{met_b}_ClusterOnlyERAPlot"),  f"{met_a},{met_b} Scatter", "MapName", True, [map_a_name, map_b_name])

def add_map_id_column_to_isovist_data(input_df, name_list):
    input_df['MapID'] = 0
    #

    #input_df.loc[input_df['MapName'] == 'BaseMap', 'MapID'] = 1
    #input_df.loc[input_df['MapName'] == 'HybridMap', 'MapID'] = 2

    counter =1

    #map_names = isoF.get_isovist_sheet_names_list()

    for name in name_list:
        
        input_df.loc[input_df['MapName'] == name, 'MapID'] = counter
        counter+=1
    return input_df

def calc_buckets(metric_list, bin_count):
    
    #First, we generate the bins
    metric_lower_lim = 100
    metric_upper_lim = -100
    for met in metric_list:
        if(met<metric_lower_lim):
            metric_lower_lim = met
        if(met>metric_upper_lim):
            metric_upper_lim = met
    
    bins = []
    bin_size = (metric_upper_lim-metric_lower_lim)/bin_count
    
    increm = metric_lower_lim
    for i in range(bin_count):
        bins.append([increm, increm+bin_size])
        increm+=bin_size
    return bins

def generate_era_heatmap(input_df, metric1, metric2, plot_title, file_name, userange = False, xmin = None, xmax = None, ymin = None, ymax = None):

    sns.set_palette(palette='Greys_r')
    #, palette=sb.color_palette("pastel")
    hexplot = sb.jointplot(x = metric1,y = metric2,data = input_df,kind = 'hex', marginal_ticks = False, cmap = 'Greys_r')

    hexplot.fig.suptitle(plot_title, fontsize = 30)
    
    if userange:
        #hexplot.ax_marg_x.set_xlim(xmin,xmax)
        #hexplot.ax_marg_y.set_ylim(ymin,ymax)
        plt.xlim = (xmin,xmax) 
        plt.ylim=(ymin,ymax)
    hexplot.ax_marg_x.remove()
    hexplot.ax_marg_y.remove()

    #plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)  # shrink fig so cbar is visible
    # make new ax object for the cbar
    cbar_ax = hexplot.fig.add_axes([.85, .25, .05, .4])  # x, y, width, height
    plt.colorbar(cax=cbar_ax)

    #sb.set_style(rc = {'axes.facecolor':'black','figure.facecolor': 'black'})

    #plt.show()

    fig = hexplot.fig
    fig.savefig(file_name,dpi=300, bbox_inches="tight") 
    
    #plt.savefig(file_name,dpi=300, bbox_inches="tight")
    plt.close()
    return

def find_biggest_isovist_shifts_every_map(isovist_df, pca_df, shift_num, output_loc):

    all_dfs = []
    #Base Map Shifts
    basemap_locmatch_path = output_loc+"/BaseMap_LocMatches"
    if not os.path.exists(basemap_locmatch_path):
        os.makedirs(basemap_locmatch_path)
    for sheet in  isoF.enum_IsovistNormalMapSheets:
        loc_matches = find_location_matched_isovists(isovist_df, 'BaseMap', sheet.name, 'XPos', 'YPos', 'ZPos')
        loc_matches.to_csv(f"{basemap_locmatch_path}/Base_{sheet.name}_MatchedPairs.csv")
        biggest_shifts = find_biggest_isovist_shifts(loc_matches, pca_df, shift_num)
        biggest_shifts['From_Map'] = 'BaseMap'
        biggest_shifts['To_Map'] = sheet.name
        biggest_shifts.to_csv(f"{basemap_locmatch_path}/Base_{sheet.name}_LargetShifts.csv")
        all_dfs.append(biggest_shifts)
        print(f"BaseMap and {sheet.name} loc match processed")

    
    #Hybrid Map Shifts
    hybridmap_locmatch_path = output_loc+"/HybridMap_LocMatches"
    if not os.path.exists(hybridmap_locmatch_path):
        os.makedirs(hybridmap_locmatch_path)
    for sheet in isoF.enum_IsovistHybridMapSheets:
        loc_matches = find_location_matched_isovists(isovist_df, 'HybridMap', sheet.name, 'XPos', 'YPos', 'ZPos')
        loc_matches.to_csv(f"{hybridmap_locmatch_path}/Hybrid_{sheet.name}_MatchedPairs.csv")
        biggest_shifts = find_biggest_isovist_shifts(loc_matches, pca_df, shift_num)
        biggest_shifts['From_Map'] = 'HybridMap'
        biggest_shifts['To_Map'] = sheet.name
        biggest_shifts.to_csv(f"{hybridmap_locmatch_path}/Base_{sheet.name}_LargetShifts.csv")
        all_dfs.append(biggest_shifts)
        print(f"HybridMap and {sheet.name} loc match processed")

    combined_shift_set = pd.concat(all_dfs)#

    combined_shift_set.to_csv(f"{output_loc}/AllMatches.csv")

    return combined_shift_set


def plot_isovist_shifts_and_largest_shifts_for_pair(map1, map2, pca_file_name, isovist_file_name, run_folder, fraction_to_check):
        #FIND AND PLOT ISOVIST SHIFTS AND LARGEST SHIFTS

    #Generate location matches
    #map1 = 'BaseMap'
    #map2 = 'Settlement_6'
    #pca_file = volcano_only_pca_data

    full_isovist_df = pd.read_csv(isovist_file_name)
    loc_matches = find_location_matched_isovists(full_isovist_df, map1, map2, 'XPos', 'YPos', 'ZPos', fraction_to_check)
    loc_matches.to_csv(f"{run_folder}/{map1}_{map2}_MatchedPairs.csv")

    #loc_matches = pd.read_csv(basemap_settlement6_matched_pairs)

    pca_data = pd.read_csv(pca_file_name)

    arrow_coords = generate_arrows_from_matched_isovists(loc_matches, pca_data)
    twomaps_df = pca_data.loc[(pca_data['MapName'] == map1) | (pca_data['MapName'] == map2)]

    #print(f"Arrow coords: {arrow_coords}")
    
    #Finding largest shifts
    biggest_shifts, biggest_shift_locmatches = find_biggest_isovist_shifts(loc_matches, pca_data, 5)

    biggest_shift_locmatches.to_csv(f"{run_folder}/{map1}_{map2}_LargestShifts.csv")

    longest_arrow_coords = generate_arrows_from_matched_isovists(biggest_shift_locmatches, pca_data)
    twomaps_df = pca_data.loc[(pca_data['MapName'] == map1) | (pca_data['MapName'] == map2)]

    #print(f"Longest Arrow coords: {longest_arrow_coords}")
    scatterGen.generate_era_scatterplot(twomaps_df, "PCA 1", "PCA 2", (f"{run_folder}/PCA_{map1}_And_{map2}_ArrowFlow"), f"{map1}-{map2}","MapName", True, [map1, map2], arrow_coords, longest_arrow_coords)


def main():
    print("Start")
    overall_start_time = datetime.now()

    
    run_folder = 'output/' +out_folder
    if not os.path.exists(run_folder):
        os.makedirs(run_folder)


    #FULL RUN OF MARIO METRIC EVALUATION
    #mario_combinedset_analysis(short_mario_inputFile)

    #GENERATING ASSEMBLED ISOVIST DATA
    
    #full_isovist_df = isoF.create_combined_df_all_isovist_sheets(isovist_fulldata)
    #full_isovist_df.to_csv(run_folder+"/full_iso_df.csv")

    #Alternatively, load it
    #full_isovist_df = pd.read_csv(assembled_isovist_data)



    #FULL RUN OF ISOVIST METRIC EVALUATION
    #isovist_combined_set_analysis(full_isovist_df, True , isoF.enum_IsovistMetrics.Diversity)

    
    #DBSCAN 
    #print(basemap_df.head)
    #pcs_only_df= basemap_df[['PCA 1','PCA 2']].copy()
    #print(pcs_only_df.head)
    #db.compute_and_visualise_dbscan(pcs_only_df)

    #GENERATE AND SAVE PCA DATA
    
    """
    full_isovist_df = pd.read_csv(assembled_isovist_data)

    print(full_isovist_df.head)

    pca_df = apply_pca_to_isovist_data(full_isovist_df, True, run_folder+'/isovist_data_scaled.csv')

    pca_df.to_csv(run_folder+'/pca_data_Revised.csv')
    """

    """

    #Alternatively, load it
    pca_df = pd.read_csv(isovist_pca_data)

    #Add Map ID col and Save
    
    all_names = isoF.get_isovist_sheet_names_list()
    all_names.append('BaseMap')
    all_names.append('HybridMap')
    pca_df_with_ids = add_map_id_column_to_isovist_data(pca_df, all_names)
    pca_df_with_ids.to_csv(run_folder+'/PCA_Data_MapIDsAdded.csv')
    """


    #GENERATE AND SAVE PCA DATA FOR INDIVIDUAL MAP SETS 

    #Generate Volcano Map Only
    
    """
    full_isovist_df = pd.read_csv(assembled_isovist_data)

    volcano_names = []
    for sheet in isoF.enum_IsovistNormalMapSheets:
        volcano_names.append(sheet.name)
    volcano_names.append('BaseMap')
    volcano_only_df = full_isovist_df[full_isovist_df['MapName'].isin(volcano_names)]
    volcano_only_df= volcano_only_df.reset_index(drop=True)
    #print("Volcano only isovist data")
    #print(volcano_only_df.head)
    volcano_only_pca_df = apply_pca_to_isovist_data(volcano_only_df, False, "", True, run_folder)
    #print("Volcano only PCA data")
    #print(volcano_only_pca_df.head)
    volcano_only_pca_df = add_map_id_column_to_isovist_data(volcano_only_pca_df, volcano_names)
    volcano_only_pca_df.to_csv(run_folder+'/volcano_only_pcadata.csv')
    """
    


    #Generate Hybrid Only
    """
    full_isovist_df = pd.read_csv(assembled_isovist_data)

    hybrid_names = []
    for sheet in isoF.enum_IsovistHybridMapSheets:
        hybrid_names.append(sheet.name)
    hybrid_names.append('HybridMap')
    hybrid_only_df = full_isovist_df[full_isovist_df['MapName'].isin(hybrid_names)]
    hybrid_only_df = hybrid_only_df.reset_index(drop=True)
    #print("Hybrid only isovist data")
    #print(hybrid_only_df.head)

    hybrid_only_pca_df = apply_pca_to_isovist_data(hybrid_only_df, False, "", True, run_folder)

    #print("Hybrid only PCA data")
    #print(hybrid_only_pca_df.head)
    
    hybrid_only_pca_df = add_map_id_column_to_isovist_data(hybrid_only_pca_df, hybrid_names)

    hybrid_only_pca_df.to_csv(run_folder+'/hybrid_only_pcadata.csv')
    """

    #GENERATE HIGH RES MAP DATA

    """
    highres_df = isoF.create_combined_df_highres_maps()

    print(highres_df.head)
    pca_highres_df = apply_pca_to_isovist_data(highres_df)
    #Readding MapID and coordinates for heatmap generation
    pca_highres_df['MapID'] = highres_df['MapID']
    pca_highres_df['XPos'] = highres_df['XPos']
    pca_highres_df['YPos'] = highres_df['YPos']
    pca_highres_df['ZPos'] = highres_df['ZPos']
    print(pca_highres_df.head)

    pca_highres_df.to_csv(run_folder+'/highres_pca_data.csv')
    """

    #Load highres data 
    #highres_pca = pd.read_csv(highres_pca_data)

    #Normalise and extract PCA1
    """
    pca1_only = highres_pca.drop(['PCA 2'], axis = 1)
    pca1_only['PCA1_Normalised'] = (pca1_only['PCA 1'] - pca1_only['PCA 1'].min()) / (pca1_only['PCA 1'].max() - pca1_only['PCA 1'].min())
    pca1_only = pca1_only.drop(['PCA 1'], axis = 1)

    
    pca1_only.to_csv(run_folder+'/highres_pca1_normalised.csv')
    """

    #scatterGen.generate_era_scatterplot(highres_pca, "PCA 1", "PCA 2", (run_folder +"/HighRes_PCA_Scatter"), "High Res PCA", "MapName", True, ['HighRes_outputs0', 'HighRes_outputs6','HighRes_outputs15' ], [])


    #GENERATE AND SAVE T-SNE DATA
    """
    #tsne_df = apply_tsne_to_isovist_data(full_isovist_df)

    #tsne_df.to_csv(run_folder+'/tsne_data.csv')
    """



    #GENERATE PCA VISUALS
    #visualise_compressed_data_all_maps(pca_df, out_folder, "PCA 1", "PCA 2")

    #GENERATE HEATMAPs OF PCA DATA

    """

    volc_pca_data = pd.read_csv(isovist_pca_data)

    basemap_only_pca = volc_pca_data.loc[(volc_pca_data['MapName'] == 'BaseMap')]
    generate_era_heatmap(basemap_only_pca, 'PCA 1', 'PCA 2', 'Volcano BaseMap', run_folder +'/VolcanoBase-HeatMap', True, -4, 5, -3, 4)
    volc1_only_pca = volc_pca_data.loc[(volc_pca_data['MapName'] == 'Settlement_1')]
    generate_era_heatmap(volc1_only_pca, 'PCA 1', 'PCA 2', 'Volcano + Generator 1', run_folder +'/VolcanoSettlement1-HeatMap', True, -4, 5, -3, 4)
    volc1_only_pca = volc_pca_data.loc[(volc_pca_data['MapName'] == 'Settlement_15')]
    generate_era_heatmap(volc1_only_pca, 'PCA 1', 'PCA 2', 'Volcano + Generator 15', run_folder +'/VolcanoSettlement15-HeatMap', True, -4, 5, -3, 4)

    hybrid_pca_data = pd.read_csv(isovist_pca_data)

    hybrid_only_pca = hybrid_pca_data.loc[(hybrid_pca_data['MapName'] == 'HybridMap')]
    generate_era_heatmap(hybrid_only_pca, 'PCA 1', 'PCA 2', 'Hybrid BaseMap', run_folder +'/HybridBase-HeatMap', True, -4, 5, -2, 3)
    hybrid1_only_pca = hybrid_pca_data.loc[(hybrid_pca_data['MapName'] == 'outputsHybrid1')]
    generate_era_heatmap(hybrid1_only_pca, 'PCA 1', 'PCA 2', 'Hybrid + Generator 1', run_folder +'/HybridSettlement1-HeatMap', True, -4, 5, -2, 3)
    hybrid15_only_pca = hybrid_pca_data.loc[(hybrid_pca_data['MapName'] == 'outputsHybrid15')]
    generate_era_heatmap(hybrid15_only_pca, 'PCA 1', 'PCA 2', 'Hybrid + Generator 15', run_folder +'/HybridSettlement15-HeatMap', True, -4, 5, -2, 3)
    """
    

    #GENERATE TSNE VISUALS
    #visualise_compressed_data_all_maps(tsne_df, out_folder, "TSNE 1", "TSNE 2")

    
    #FIND CO-LOCATED PAIRS
    """

    #Generate location matches
    full_isovist_df = pd.read_csv(assembled_isovist_data)
    loc_matches = find_location_matched_isovists(full_isovist_df, 'BaseMap', 'Settlement_1', 'XPos', 'YPos', 'ZPos')
    loc_matches.to_csv(run_folder+"/Base_Settlement_1_MatchedPairs.csv")

    #Alternatively, load location matches
    #loc_matches = pd.read_csv(test_matched_pairs)

    #pca_data = pd.read_csv(isovist_pca_data)

    #arrow_coords = generate_arrows_from_matched_isovists(loc_matches, pca_df)
    #twomaps_df = pca_df.loc[(pca_df['MapName'] == 'BaseMap') | (pca_df['MapName'] == 'Settlement_1')]
    #scatterGen.generate_era_scatterplot(twomaps_df, "PCA 1", "PCA 2", (run_folder +"/PCA_Base_And_Settlement1_ArrowFlow"),  f"Base and Settlement1 Generative Shift","MapName", True, ['BaseMap', 'Settlement_1'], arrow_coords)
    
    """

    #FIND BIGGEST PCA SHIFTS
    """
    map1 = 'BaseMap'
    map2 = 'Settlement_6'
    pca_file = hybrid_only_pca_data

    volc_1 = 'output/' +out_folder
    if not os.path.exists(run_folder):
        os.makedirs(run_folder)

    plot_isovist_shifts_and_largest_shifts_for_pair(map1, map2, pca_file, assembled_isovist_data, run_folder, .02)
    """

    """
    loc_matches = pd.read_csv(test_matched_pairs)
    pca_data = pd.read_csv(isovist_pca_data)

    biggest_shifts = find_biggest_isovist_shifts(loc_matches, pca_data, 10)

    print(biggest_shifts.head)
    """
    
    #FIND AND PLOT ISOVIST SHIFTS AND LARGEST SHIFTS

    #Generate location matches
    map1 = 'BaseMap'
    map2 = 'Settlement_6'
    pca_file = volcano_only_pca_data

    full_isovist_df = pd.read_csv(assembled_isovist_data)
    #loc_matches = find_location_matched_isovists(full_isovist_df, map1, map2, 'XPos', 'YPos', 'ZPos')
    #loc_matches.to_csv(f"{run_folder}/{map1}_{map2}_MatchedPairs.csv")

    loc_matches = pd.read_csv(basemap_settlement6_matched_pairs)

    pca_data = pd.read_csv(pca_file)

    arrow_coords = generate_arrows_from_matched_isovists(loc_matches, pca_data)
    twomaps_df = pca_data.loc[(pca_data['MapName'] == map1) | (pca_data['MapName'] == map2)]

    #print(f"Arrow coords: {arrow_coords}")
    
    #Finding largest shifts
    biggest_shifts, biggest_shift_locmatches = find_biggest_isovist_shifts(loc_matches, pca_data, 5)

    longest_arrow_coords = generate_arrows_from_matched_isovists(biggest_shift_locmatches, pca_data)
    twomaps_df = pca_data.loc[(pca_data['MapName'] == map1) | (pca_data['MapName'] == map2)]

    biggest_shift_locmatches.to_csv(f"{run_folder}/{map1}_{map2}_LargestShifts.csv")

    #print(f"Longest Arrow coords: {longest_arrow_coords}")
    scatterGen.generate_era_scatterplot(twomaps_df, "PCA 1", "PCA 2", (f"{run_folder}/PCA_{map1}_And_{map2}_ArrowFlow"), f"{map1},{map2} Generative Shift","MapName", True, [map1, map2], arrow_coords, longest_arrow_coords)
    


    ##FIND MOST ISOLATED POINTS IN SINGLE MAP
    """
    full_isovist_df = pd.read_csv(assembled_isovist_data)
    pca_data = pd.read_csv(isovist_pca_data)
    #basemap_only_df = full_isovist_df.loc[(full_isovist_df['MapName'] == 'BaseMap')]
    basemap_only_pca = pca_data.loc[(pca_data['MapName'] == 'BaseMap')]
    basemap_only_pca= basemap_only_pca.iloc[:100]

    print("Pca head pre generation")
    print(basemap_only_pca.head)


    isolated_points = find_most_isolated_pca(basemap_only_pca,basemap_only_pca,full_isovist_df, 5 , True)

    print(isolated_points.head)
    """

    ##FIND MOST ISOLATED POINTS FROM ONE MAP TO ANOTHER
    """
    pca_data = pd.read_csv(isovist_pca_data)
    full_isovist_df = pd.read_csv(assembled_isovist_data)

    basemap_only_pca = pca_data.loc[(pca_data['MapName'] == 'BaseMap')]
    basemap_only_pca= basemap_only_pca.iloc[:100]
    
    settlement16_only_pca = pca_data.loc[(pca_data['MapName'] == 'Settlement_16')]
    settlement16_only_pca= settlement16_only_pca.iloc[:100]

    print("Settlement 16 head:")
    print(settlement16_only_pca.head)

    isolated_settlement16 = find_most_isolated_pca(basemap_only_pca,settlement16_only_pca,full_isovist_df, 5 )

    print(isolated_settlement16.head)
    """


    #GENERATE ALL ERA PLOTS FOR SPECIFIC AREA OF PCA PLOT
    
    
    #generate_all_era_plots_for_selected_pca_cluster(run_folder, 'BaseMap', 'Settlement_1', -4, 4, -4, 4)

    #generate_all_era_plots_for_selected_pca_cluster(run_folder, 'HybridMap', 'outputsHybrid1', -4, 4, -4, 4)


    #SEPERATE PCA DATA INTO GROUPS
    #pca_data = pd.read_csv(isovist_pca_data)
    #sorted_df = sort_datapoints(pca_data, 'BaseMap', 'HybridMap', 'PCA 1', 'PCA 2', -1)
    #print(sorted_df.head)
    #pca_scatter_folder = 'output/' +out_folder
    #if not os.path.exists(pca_scatter_folder):
    #    os.makedirs(pca_scatter_folder)
    #scatterGen.generate_era_scatterplot(sorted_df, "PCA 1", "PCA 2", (pca_scatter_folder +"/PCA_Visual-Scatterplot_HybridAndBase"), "Combined PCA","MapAppearance", True, ['Both', 'a_only','b_only'])
    #scatterGen.generate_era_scatterplot(sorted_df, "PCA 1", "PCA 2", (pca_scatter_folder +"/PCA_Visual-Scatterplot_BaseOnly"), "Base Only PCA","MapAppearance", True, ['a_only'])
    #scatterGen.generate_era_scatterplot(sorted_df, "PCA 1", "PCA 2", (pca_scatter_folder +"/PCA_Visual-Scatterplot_HybridOnly"),"Hybrid Only PCA", "MapAppearance", True, ['b_only'])
    #scatterGen.generate_era_scatterplot(sorted_df, "PCA 1", "PCA 2", (pca_scatter_folder +"/PCA_Visual-Scatterplot_BothOnly"),"Both Only PCA", "MapAppearance", True, ['Both'])
    


    #GENERATE LARGEST ISOVIST SHIFTS FOR EVERY MAP
    """
    full_isovist_df = pd.read_csv(assembled_isovist_data)
    pca_data = pd.read_csv(isovist_pca_data)

    find_biggest_isovist_shifts_every_map(full_isovist_df, pca_data, 5, run_folder)
    """
    


    runtime_seconds=  datetime.now () -overall_start_time
    runtime_minutes = runtime_seconds/60
    print("Total Runtime: " + str(runtime_minutes) + " minutes")

if __name__ == "__main__":
    main()

