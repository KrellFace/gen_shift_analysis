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
import isovistFunc as isoF
import ERAScatterGeneration as scatterGen

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
#highres_pca_data = 'inputdata/highres_pca_data.csv'

basemap_settlement6_matched_pairs = 'inputdata/BaseMap_Settlement_6_MatchedPairs.csv'

hybrid_only_pca_data = 'inputdata/hybrid_only_pcadata.csv'
volcano_only_pca_data = 'inputdata/volcano_only_pcadata.csv'

#CONFIG
buck_cnt = 20
out_folder = 'TestRunV2'


overall_start_time = 0


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

def add_map_id_column_to_isovist_data(input_df, name_list):
    input_df['MapID'] = 0

    counter =1

    #map_names = isoF.get_isovist_sheet_names_list()

    for name in name_list:
        
        input_df.loc[input_df['MapName'] == name, 'MapID'] = counter
        counter+=1
    return input_df


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




    #GENERATE AND SAVE PCA DATA FOR VOLCANO MAPS
    
    
    """
    full_isovist_df = pd.read_csv(assembled_isovist_data)

    volcano_names = []
    for sheet in isoF.enum_IsovistNormalMapSheets:
        volcano_names.append(sheet.name)
    volcano_names.append('BaseMap')
    volcano_only_df = full_isovist_df[full_isovist_df['MapName'].isin(volcano_names)].reset_index(drop=True)

    volcano_only_pca_df = apply_pca_to_isovist_data(volcano_only_df, False, "", True, run_folder)
    volcano_only_pca_df = add_map_id_column_to_isovist_data(volcano_only_pca_df, volcano_names)
    volcano_only_pca_df.to_csv(run_folder+'/volcano_only_pcadata.csv')
    """
    
    #GATHER MATCHED LOCATIONS
    
    map1 = 'BaseMap'
    map2 = 'Settlement_6'
    volc_pca_file = volcano_only_pca_data
    

    #Generate Location Matches

    """
    loc_matches = find_location_matched_isovists(full_isovist_df, map1, map2, 'XPos', 'YPos', 'ZPos')
    loc_matches.to_csv(f"{run_folder}/{map1}_{map2}_MatchedPairs.csv")
    """
    

    #Or Read Location Matches from file 
    
    loc_matches = pd.read_csv(basemap_settlement6_matched_pairs)

    #GENERATE SHIFT ARROWS FROM LOCATION MATCHES

    volc_pca_data = pd.read_csv(volc_pca_file)

    arrow_coords = generate_arrows_from_matched_isovists(loc_matches, volc_pca_data)
    twomaps_df = volc_pca_data.loc[(volc_pca_data['MapName'] == map1) | (volc_pca_data['MapName'] == map2)]

    #FIND LARGEST SHIFTS
    biggest_shifts, biggest_shift_locmatches = find_biggest_isovist_shifts(loc_matches, volc_pca_data, 5)
    biggest_shift_locmatches.to_csv(f"{run_folder}/{map1}_{map2}_LargestShifts.csv")

    #GENERATE SHIFT VISUALS
    longest_arrow_coords = generate_arrows_from_matched_isovists(biggest_shift_locmatches, volc_pca_data)
    scatterGen.generate_era_scatterplot(twomaps_df, "PCA 1", "PCA 2", (f"{run_folder}/PCA_{map1}_And_{map2}_ArrowFlow"), f"{map1},{map2} Generative Shift","MapName", True, [map1, map2], arrow_coords, longest_arrow_coords)
    

    runtime_seconds=  datetime.now () -overall_start_time
    runtime_minutes = runtime_seconds/60
    print("Total Runtime: " + str(runtime_minutes) + " minutes")

if __name__ == "__main__":
    main()

