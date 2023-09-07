
from enum import Enum, auto
import pandas as pd
import os.path

class enum_IsovistHybridMapSheets(Enum):
    
    outputsHybrid1 = auto(),
    outputsHybrid2 = auto(),
    outputsHybrid3 = auto(),
    outputsHybrid4 = auto(),
    outputsHybrid5 = auto(),
    outputsHybrid6 = auto(),
    outputsHybrid7 = auto(),
    outputsHybrid8 = auto(),
    outputsHybrid9 = auto(),
    outputsHybrid10 = auto(),
    outputsHybrid11 = auto(),
    outputsHybrid12 = auto(),
    outputsHybrid13 = auto(),
    outputsHybrid14 = auto(),
    outputsHybrid15 = auto(),
    outputsHybrid16 = auto(),
    outputsHybrid17 = auto(),
    outputsHybrid18 = auto(),
    outputsHybrid19 = auto(),
    outputsHybrid20 = auto(),
    
    
class enum_IsovistNormalMapSheets(Enum):
    Settlement_1 = auto(),
    Settlement_2 = auto(),
    Settlement_3 = auto(),
    Settlement_4 = auto(),
    Settlement_5 = auto(),
    Settlement_6 = auto(),
    Settlement_7 = auto(),
    Settlement_8 = auto(),
    Settlement_9 = auto(),
    Settlement_10 = auto(),
    Settlement_11 = auto(),
    Settlement_12 = auto(),
    Settlement_13 = auto(),
    Settlement_14 = auto(),
    Settlement_15 = auto(),
    Settlement_16 = auto(),
    Settlement_17 = auto(),
    Settlement_18 = auto(),
    Settlement_19 = auto(),
    Settlement_20 = auto()

class enum_IsovistMetrics(Enum):
    Area = auto(),
    Perimeter = auto(),
    Diversity = auto(),
    var_Radials = auto(),
    mean_Radials = auto(),
    Roundness = auto(),
    Openness = auto(),
    Clutter = auto(),
    Reachability = auto(),
    Occlusivity = auto(),
    DriftLength = auto(),
    VistaLength = auto(),
    RealPerimeterSize = auto()


isovist_basemap = 'inputdata/VolcanoBaseMap.csv'
isovist_hybrid_basemap = 'inputdata/HybridBaseMap.csv'


def load_isovist_data_from_sheet(file_path, sheet_name, noHeader = False):

    if os.path.exists(file_path):
        print("Isovist File exists")
    raw_df
    if(noHeader):
        raw_df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    else:
        raw_df = pd.read_excel(file_path, sheet_name=sheet_name)
    renamed_df = trim_and_rename_isovist_columns(raw_df)
    #print(test_df.head)
    #print(renamed_df.head)
    return renamed_df

def trim_and_rename_isovist_columns(input_df):
    #output_df = input_df[['Column4','Column5','Column7','Column8','Column9','Column10','Column11','Column14','Column15','Column16','Column18','Column19','Column20',]].copy()
    #output_df.rename(columns = {'Column4':"Area","Column5":"Perimeter","Column7":"Diversity","Column8":"var_Radials","Column9":"mean_Radials","Column10":"Roundness","Column11":"Openness",
    #                             "Column14":"Clutter","Column15":"Reachability","Column16":"Occlusivity","Column1":"DriftLength","Column19":"VistaLength","Column20":"RealPerimeterSize"})
    output_df = input_df.iloc[: , [0,1,2,3, 4, 6, 7, 8, 9, 10, 13, 14, 15, 17, 18, 19]].copy()
    output_df.columns = ["XPos","YPos", "ZPos", "Area","Perimeter","Diversity","var_Radials","mean_Radials","Roundness","Openness",
                                "Clutter","Reachability","Occlusivity","DriftLength","VistaLength","RealPerimeterSize"]
    return output_df


def create_combined_df_all_isovist_sheets(file_path):
    all_dfs = []
    for sheet in enum_IsovistHybridMapSheets:
        sheet_name = sheet.name.replace("_"," ")
        sheet_df = pd.read_excel(file_path, sheet_name=sheet_name)
        #print("Prerenamed sheet:")
        #print(sheet_df.head)
        sheet_renamedcols_df = trim_and_rename_isovist_columns(sheet_df)
        #Add new column of increasing integers 
        sheet_renamedcols_df.insert(0, 'Isovist_ID', range(0, 0 + len(sheet_renamedcols_df)))

        sheet_renamedcols_df['MapName'] = sheet.name
        all_dfs.append(sheet_renamedcols_df)
        
    for sheet in enum_IsovistNormalMapSheets:
        sheet_name = sheet.name.replace("_"," ")
        sheet_df = pd.read_excel(file_path, sheet_name=sheet_name)
        #print("Prerenamed sheet:")
        #print(sheet_df.head)
        sheet_renamedcols_df = trim_and_rename_isovist_columns(sheet_df)
        #Add new column of increasing integers 
        sheet_renamedcols_df.insert(0, 'Isovist_ID', range(0, 0 + len(sheet_renamedcols_df)))

        sheet_renamedcols_df['MapName'] = sheet.name
        all_dfs.append(sheet_renamedcols_df)
    
    #Add base map
    basemap_df = pd.read_csv(isovist_basemap)
    basemap_renamedcols_df = trim_and_rename_isovist_columns(basemap_df)
    basemap_renamedcols_df.insert(0, 'Isovist_ID', range(0, 0 + len(basemap_renamedcols_df)))
    basemap_renamedcols_df['MapName'] = 'BaseMap'
    all_dfs.append(basemap_renamedcols_df)

    #Add hybrid basemap
    hybrid_basemap_df = pd.read_csv(isovist_hybrid_basemap)
    hybrid_renamedcols_df = trim_and_rename_isovist_columns(hybrid_basemap_df)
    hybrid_renamedcols_df.insert(0, 'Isovist_ID', range(0, 0 + len(hybrid_renamedcols_df)))
    hybrid_renamedcols_df['MapName'] = 'HybridMap'
    all_dfs.append(hybrid_renamedcols_df)

    
    combined_df = pd.concat(all_dfs, axis=0, ignore_index=True)
    return combined_df

def get_isovist_metric_names_list(usingFitness = False, fitnessMetric = enum_IsovistMetrics.Diversity):
    
    full_mets = []
    for metric in enum_IsovistMetrics:
        if(not usingFitness or metric is not fitnessMetric):
            full_mets.append(metric.name)
    return full_mets

def get_isovist_metrics_list(usingFitness = False, fitnessMetric = enum_IsovistMetrics.Diversity):
    
    full_mets = []
    for metric in enum_IsovistMetrics:
        if(not usingFitness or metric is not fitnessMetric):
            full_mets.append(metric)
    return full_mets

def get_isovist_sheets_list():
    
    all_sheets = []
    for sheet in enum_IsovistHybridMapSheets:
        all_sheets.append(sheet)
    for sheet in enum_IsovistNormalMapSheets:
        all_sheets.append(sheet.name)
    return all_sheets

def get_isovist_sheet_names_list():
    
    all_sheets = []
    for sheet in enum_IsovistHybridMapSheets:
        all_sheets.append(sheet.name)
    for sheet in enum_IsovistNormalMapSheets:
        all_sheets.append(sheet.name)
    return all_sheets
