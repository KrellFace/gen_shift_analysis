import enum
from operator import truediv
from scipy.stats import spearmanr
from enum import Enum, auto
from datetime import *
import numpy as np
import pandas as pd
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt
import statistics
import os
import os.path
import marioFunc
import ERAScatterGeneration as genERAscatter


#Generate bucket boundaries for full range of values for a given metric
#Boundaries will be evenly spaced between the minimum and maximum values found
def generate_bucketlist_for_metric(valslist, bucketcount):
    minVal = 9999999999999999999
    maxVal = 0
    for val in valslist:
        if (val>maxVal):
            maxVal = val
        if (val<minVal):
            minVal = val

    #Calculate bucket increments
    diff = maxVal-minVal
    bucketSize = diff/bucketcount

    metricBuckets =[]
    metricBuckets.append(minVal)
    for i in range(1,(bucketcount+1)):
        metricBuckets.append(minVal+(bucketSize*i))
    
    return metricBuckets

def normalise_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

#Create dictionary of coordinates
def generate_coordinate_dict(xysize):
    coorddict = dict()
    for x in range(xysize):
        for y in range(xysize):
            coorddict[tuple([y,x])] = []
    return coorddict

#Add a given set of levels in a dataframe, add them to a dictionary of coordinate locations based on their metric values
def add_levels_to_coord_dict_based_on_metricvals(df, coorddict, metricx, metricy, bucketsx, bucketsy, collection_col_name, artefact_col_name):

    for index, row in df.iterrows():
        #print(row[metricx.name], row[metricy.name])
        xpos = 0
        ypos = 0
        #NB - No need to check the final bucket, as that represents the upper limit of values found
        for x in range(len(bucketsx)-1):
            if(row[metricx.name]>bucketsx[x]):
                xpos = x
        
        for y in range(len(bucketsy)-1):
            if(row[metricy.name]>bucketsy[y]):
                ypos = y
        
        #OLD METHOD
        #coorddict[tuple([ypos, xpos])].append([row[collection_col_name],row[artefact_col_name]])
        coorddict[tuple([ypos, xpos])].append(index)

        #print("Adding level " +row['LevelName']+ "," + row['Generator'] + " to location: " + str(xpos)+","+str(ypos))
    
    return coorddict

def gen_and_place_avg_metric_vals_np_matrix(df, coorddict, bucket_count, metric, collection_col_name, artefact_col_name):
    #print("Starting artefact placement in avg fit grid")
    avgfitnessmatrix = np.zeros((bucket_count, bucket_count))

    total_artefacts_processed = 0

    for x in range(0,bucket_count):
        for y in range(0,bucket_count):

            #print("Processing location: " + str(x) +","+str(y))

            artefacts = coorddict[tuple([y,x])]
            
            if(len(artefacts)>0):


            
                totmetricval = 0
                #OLD METHOD, LOOPING THROUGH INDIVIDUAL ROWS
                #for artefact in artefacts:
                    #levelrow = df.loc[(df[collection_col_name] == artefact[0]) & (df[artefact_col_name] ==artefact[1])]
                    
                    #metricval = levelrow[[metric.name]].values.reshape(-1)
                    #totmetricval+=metricval[0]
                    #total_artefacts_processed+=1

                    #if(total_artefacts_processed%1000==0):
                        #print("Artefacts processed in avg fit calc: " + str(total_artefacts_processed))
                    #print("Artefact with metric val " + str(metricval[0]) +" added")
                
                
                #avgmetricval = totmetricval/len(artefacts)
                
                #Grab only rows with matching indexes
                df_filtered = df.filter(items = artefacts, axis=0)
                #Average metric col value
                avgmetricval = df_filtered.loc[:, metric.name].mean()


                avgfitnessmatrix[y,x] = avgmetricval
                #print("Avg " + metric.name + " val at " + str(x) +"," + str(y) + ": " + str(avgmetricval))

    return avgfitnessmatrix

def buckets_to_colnames(bucketlist):
    bucketnames = []
    for i in range(0, len(bucketlist)-1):
        #bucket = (str(round(bucketlist[i],2))+ " to " + str(round(bucketlist[i+1],2)))
        bucket = str(round(bucketlist[i],2))
        bucketnames.append(bucket)
        #print(bucket)
    return bucketnames


#Generate a heatmap image from a heatmap formatted dataframe
def generate_heatmap_From_df(df, metricx, metricy, showvisual, saveimg, filepath):
    ax = sns.heatmap(df, vmin=0, vmax=1, cbar_kws={'label': 'Avg Fitness'}, cmap = "magma")
    ax.figure.axes[-1].yaxis.label.set_size(18)
    ax.invert_yaxis()
    ax.set_xlabel(metricx.name, fontsize = 20)
    ax.set_ylabel(metricy.name, fontsize = 20)
    
    plt.tight_layout()
    if (showvisual):
        plt.show()
    if(saveimg):
        plt.savefig(filepath+'.png')
    plt.close()

def calc_mutual_correlation(inputdf,metricXvals, metricYvals):
    #metricXvals = inputdf[[metricx.name]].values.reshape(-1)
    #metricYvals = inputdf[[metricy.name]].values.reshape(-1)

    metricXallSame = False
    metricYallSame = False

    #Handling for all metric values being identical, which causes spearmans rho to fail
    if(np.min(metricXvals)==np.max(metricXvals)):
        metricXallSame = True
    if(np.min(metricYvals)==np.max(metricYvals)):
        metricYallSame = True


    #MUTUAL CORRELATION BETWEEN PAIR
    mut_corr = 0
    mut_corr_pval = 0

    if(not metricXallSame and not metricYallSame):
        mut_corr, mut_corr_pval = spearmanr(metricXvals, metricYvals)
        #Instead store 1 - Abs(mut_corr), to give a score from 0 to 1 where 1 is best
        mut_corr = (1-abs(mut_corr))
    return mut_corr, mut_corr_pval


def calc_alt_metric_correlation(inputdf, metricx, metricy, metrics,metricXvals,metricYvals, metricXallSame, metricYallSame):
    #CORRELATION LEVEL WITH ALT BCS:
    tot_othercorr = 0
    tot_othercorr_pval = 0

    for otherbc in metrics:
        if (otherbc!=metricx and otherbc!= metricy):
            
        
            maxcorr = 0
            maxcorrpval = 0
            otherMetricVals = inputdf[[otherbc.name]].values.reshape(-1)

            otherMetricValsAllIdentical = (np.min(otherMetricVals)==np.max(otherMetricVals))

            if not otherMetricValsAllIdentical:
                
                #print(f"Checking alt corr for pair: {metricx},{metricy} with alt metric: {otherbc}")
                metricXAltCorr, metricXaltCorrP = [0,0]


                if not metricXallSame:
                    metricXAltCorr, metricXaltCorrP = spearmanr(metricXvals, otherMetricVals)
                #print('Sp corr for BC sub pair: ' + metric1.name + ", " + otherbc.name + ' : %.3f' % spcorr1 + " with P Value: " + str("{:.2f}".format(pspval)))
                if(abs(metricXAltCorr)>maxcorr):
                    maxcorr = abs(metricXAltCorr)
                    maxcorrpval = metricXaltCorrP
                    
                metricYAltCorr, metricYaltCorrP = [0,0]
                if not metricYallSame:
                    metricYAltCorr, metricYaltCorrP = spearmanr(metricYvals, otherMetricVals)
                
                #print('Sp corr for BC sub pair: ' + metric2.name + ", " + otherbc.name + ' : %.3f' % spcorr2 + " with P Value: " + str("{:.2f}".format(pspval)))
                if(abs(metricYAltCorr)>maxcorr):
                    maxcorr = abs(metricYAltCorr)
                    maxcorrpval=metricYaltCorrP

                #print('Best abs corr found with sub BC: ' + otherbc.name+ " = " + str(maxcorr) )
                
                tot_othercorr+=maxcorr
                tot_othercorr_pval += maxcorrpval
                #else:
                    #print(f"All BC {otherbc} vals for dataset: {dataset_name} identical. Set other corr to 0" )


    avg_alt_corr = tot_othercorr/(len(metrics)-2)
    avg_alt_corr_pval = tot_othercorr_pval/(len(metrics)-2)

    print("Average alt metcor for " + metricx.name+","+metricy.name+":"+str(avg_alt_corr))

    return avg_alt_corr, avg_alt_corr_pval

def calc_and_visualise_fitness_independence(inputdf, metricx, metricy, bucket_count, fitness_metric, collection_col_name, artefact_col_name, output_foldername):

    #print("Starting calc av fitness")

    metricXvals = inputdf[[metricx.name]].values.reshape(-1)
    metricYvals = inputdf[[metricy.name]].values.reshape(-1)

    #Calculate Bucket values
    metricxBuckets = generate_bucketlist_for_metric(metricXvals,bucket_count)
    metricyBuckets = generate_bucketlist_for_metric(metricYvals,bucket_count)

    #print("Bucket list created")

    #Generate a dictionary of coordinates and populate it with pointers to each occupying level
    coord_dict =  generate_coordinate_dict(bucket_count)
    #print("Coord dict created")
    coord_dict =  add_levels_to_coord_dict_based_on_metricvals(inputdf, coord_dict, metricx, metricy, metricxBuckets, metricyBuckets, collection_col_name, artefact_col_name)
    #print("Levels added to coord dict")

    #Generate matrix of the average fitness of levels at each location
   #print("Inputdf head:")
    #print(inputdf.head)


    avgfitmatrix = gen_and_place_avg_metric_vals_np_matrix(inputdf, coord_dict, bucket_count, fitness_metric, collection_col_name, artefact_col_name)   
    #print("Avg fit matrix generated")
    #print("AvgFit marix generated")
    
    #Generate Heatmap
    xcols = buckets_to_colnames(metricxBuckets)
    ycols = buckets_to_colnames(metricyBuckets)
    heatmapdf = pd.DataFrame(avgfitmatrix, index = ycols, columns=xcols)
    #print("Headmap head:")
    #print(heatmapdf.head())
    generate_heatmap_From_df(heatmapdf, metricx, metricy, False, True, (output_foldername + metricx.name+","+metricy.name+"-FitnessHeatmap"))
    
    #Generate Standard Deviation of Fitness:
    avgfitvals = avgfitmatrix.flatten()

    stddev_avg_fit = statistics.stdev(avgfitvals)
    avg_fit = np.average(avgfitvals)


    print("Avg fit for output: " + output_foldername + " and metric pair: "+ metricx.name+","+metricy.name+":"+str(avg_fit))

    
    return avg_fit, stddev_avg_fit


def create_criteria_ranks_csv(gen_data_df, using_fitness, csvname):
    ranks_dict = dict()

    for index, row in gen_data_df.iterrows():
        mut_corr_val = row['Mutual_Correlation']
        mut_corr_rank = 1
        alt_met_val = row['Avg_Alt_Metric_Correlation']
        alt_met_rank = 1
        if(using_fitness):
            avg_fit_val = row['Avg_Fit']
            avg_fit_rank = 1
            fit_stddev_val = row['Avg_Fit_Stddev']
            fit_stddev_rank = 1
        else:
            avg_fit_val = []
            avg_fit_rank = 1
            fit_stddev_val = []
            fit_stddev_rank = 1


        for inner_index, inner_row in gen_data_df.iterrows():
            #Use the Abs value as strong negative correlation is also bad
            if inner_row['Mutual_Correlation']>mut_corr_val:
                mut_corr_rank+=1
            if inner_row['Avg_Alt_Metric_Correlation']>alt_met_val:
                alt_met_rank+=1
            if(using_fitness):
                if inner_row['Avg_Fit']>avg_fit_val:
                    avg_fit_rank+=1
                if inner_row['Avg_Fit_Stddev']<fit_stddev_val:
                    fit_stddev_rank+=1
        if(using_fitness):
            avg_all_ranks = (mut_corr_rank+alt_met_rank+avg_fit_rank)/3
        else:
            avg_all_ranks = (mut_corr_rank+alt_met_rank)/2
        
        if(using_fitness):
            ranks_dict[index] = [mut_corr_rank, alt_met_rank, avg_fit_rank, fit_stddev_rank, avg_all_ranks]
        else:
            ranks_dict[index] = [mut_corr_rank, alt_met_rank,avg_all_ranks]
    
    if(using_fitness):
        rank_data_df = pd.DataFrame.from_dict(ranks_dict, orient='index', columns=['Mut_Corr_Rank', 'Alt_Met_Rank', 'Avg_Fit_Rank', 'Avg_Fit_Stddev_Rank', 'Avg_All_Ranks'])
    else:
        rank_data_df = pd.DataFrame.from_dict(ranks_dict, orient='index', columns=['Mut_Corr_Rank', 'Alt_Met_Rank', 'Avg_All_Ranks'])
    
    rank_data_df.to_csv((csvname+".csv"), index = True)


def gen_scatterplot_for_all_metricpairs(inputdf, metric_list, output_foldername, dataset_name, gen_column_name, generators):
    pairstested = []
    for metricx in metric_list:
        for metricy in metric_list:
            if (metricx!=metricy and [metricx,metricy] not in pairstested):
                
                print("Generating scatterplot for metric pair: " + metricx.name +"," + metricy.name+" for set:" + dataset_name)

                #Generate Scatter Plot for Metric Pair
                genERAscatter.generate_era_scatterplot(inputdf, metricx.name, metricy.name, (output_foldername + metricx.name+","+metricy.name+"-Scatterplot"), gen_column_name, True, generators)

#Main method for calculating metric selection criteria for an input data frame of levels and their metric values
def gen_validationdata_from_metric_df(inputdf, metrics, bucket_count, output_foldername, collection_col_name, artefact_col_name,  dataset_name, using_fitness = True, fitness_metric = marioFunc.enum_MarioMetrics.Playability):

    if not os.path.exists(output_foldername):
        os.makedirs(output_foldername)
    generator_data = dict()

    #Loop through every metric pair
    pairstested = []
    for metricx in metrics:
        for metricy in metrics:
            if (metricx!=metricy and [metricx,metricy] not in pairstested):
                
                print("Evaluating metric pair: " + metricx.name +"," + metricy.name+" for set:" + dataset_name)

                #Generate Scatter Plot for Metric Pair
                #genERAscatter.generate_era_scatterplot(inputdf, metricx.name, metricy.name, (output_foldername + metricx.name+","+metricy.name+"-Scatterplot"), gen_column_name = collection_col_name)
                
                #Extract value list for each metric 
                metricXvals = inputdf[[metricx.name]].values.reshape(-1)
                metricYvals = inputdf[[metricy.name]].values.reshape(-1)

                metricXallSame = False
                metricYallSame = False

                #Handling for all metric values being identical, which causes spearmans rho to fail
                if(np.min(metricXvals)==np.max(metricXvals)):
                    metricXallSame = True
                if(np.min(metricYvals)==np.max(metricYvals)):
                    metricYallSame = True


                #MUTUAL CORRELATION BETWEEN PAIR

                mut_corr,mut_corr_pval = calc_mutual_correlation(inputdf, metricXvals, metricYvals)

                avg_alt_corr,avg_alt_corr_pval= calc_alt_metric_correlation(inputdf, metricx, metricy, metrics,metricXvals,metricYvals, metricXallSame, metricYallSame)
                
                print("For Output: " + output_foldername + " and metric pair: " + metricx.name +"," + metricy.name+ " Avg Alt Metric corr: " + str(avg_alt_corr) + " and avg pval: " + str(avg_alt_corr_pval))

                    
                #AVERAGE FITNESS AND FITNESS HEATMAP
                if(using_fitness):

                    avg_fit,stddev_avg_fit= calc_and_visualise_fitness_independence(inputdf, metricx, metricy, bucket_count, fitness_metric, collection_col_name, artefact_col_name, output_foldername)

                #Add to dictionary of data
                if(using_fitness):
                    generator_data[f"{metricx.name},{metricy.name}"] = [mut_corr, mut_corr_pval, avg_alt_corr, avg_alt_corr_pval, avg_fit, stddev_avg_fit]
                else:
                    generator_data[f"{metricx.name},{metricy.name}"] = [mut_corr, mut_corr_pval, avg_alt_corr, avg_alt_corr_pval]


                #ADD TO PAIRS TESTED
                pairstested.extend(([metricx, metricy],[metricy, metricx]))
        

        #Create Generator data csv
        if(using_fitness):
            gen_data_df = pd.DataFrame.from_dict(generator_data, orient= 'index', columns=['Mutual_Correlation', 'Mutual_Correlation_Pval', 'Avg_Alt_Metric_Correlation', 'Avg_Alt_Met_Pval', 'Avg_Fit', 'Avg_Fit_Stddev'])
        else:
            gen_data_df = pd.DataFrame.from_dict(generator_data, orient= 'index', columns=['Mutual_Correlation', 'Mutual_Correlation_Pval', 'Avg_Alt_Metric_Correlation', 'Avg_Alt_Met_Pval'])
        #gen_data_outputpath = Path(generator_output_path + generator.name+"-data")
        gen_data_df.to_csv((output_foldername + dataset_name+"-data.csv"), index = True)

        #CREATE DATAFRAME OF RANKINGS FOR EACH VALUE
        create_criteria_ranks_csv(gen_data_df, using_fitness, (output_foldername + dataset_name+"-ranks"))
        


#Method for calculating metric pair selection criteria for individual generators, as well as the compiled set
#def gen_data_for_generators_and_overall_set(input_df, generators, metrics, bucket_count, output_folder_name, collection_col_name, artefact_col_name, using_fitness = True, fitness_metric = marioFunc.enum_MarioMetrics.Playability):
    #if not os.path.exists(output_folder_name):
        #os.makedirs(output_folder_name)
    
    #for generator in generators:
        #print("Starting data gathering for generator: " + generator.name)


        #gen_df = input_df.loc[(input_df[collection_col_name] == generator.name)]
        #if(using_fitness):
            #gen_validationdata_from_metric_df(gen_df, metrics, bucket_count, output_folder_name +generator.name+"/",collection_col_name, artefact_col_name, generator.name, True, fitness_metric)
        #else:
            #gen_validationdata_from_metric_df(input_df, metrics, bucket_count, output_folder_name +generator.name+"/",collection_col_name, artefact_col_name, generator.name, False)

    
    #if(using_fitness):
        #gen_validationdata_from_metric_df(input_df, metrics, bucket_count, output_folder_name +"Overall_Data/", collection_col_name, artefact_col_name,  "Overall_Data", True, fitness_metric)
    #else:
        #gen_validationdata_from_metric_df(input_df, metrics, bucket_count, output_folder_name +"Overall_Data/", collection_col_name, artefact_col_name, "Overall_Data", False)


#Method for calculating metric pair selection criteria for overall set only
def gen_selection_criteria_information_for_dataset(input_df, metrics, bucket_count, output_folder_name, collection_col_name, artefact_col_name, eval_individual_generators = False, generators = [], using_fitness = True, fitness_metric = marioFunc.enum_MarioMetrics.Playability):
    if not os.path.exists(output_folder_name):
        os.makedirs(output_folder_name)


    #If using Fitness - Normalize Values Between 0 & 1
    if(using_fitness):
        input_df[fitness_metric.name] = (input_df[fitness_metric.name] - input_df[fitness_metric.name].min()) / (input_df[fitness_metric.name].max() - input_df[fitness_metric.name].min())    


    if(eval_individual_generators):
        for generator in generators:
            print("Starting data gathering for generator: " + generator.name)


            gen_df = input_df.loc[(input_df[collection_col_name] == generator.name)]
            if(using_fitness):
                gen_validationdata_from_metric_df(gen_df, metrics, bucket_count, output_folder_name +generator.name+"/",collection_col_name, artefact_col_name, generator.name, True, fitness_metric)
            else:
                gen_validationdata_from_metric_df(input_df, metrics, bucket_count, output_folder_name +generator.name+"/",collection_col_name, artefact_col_name, generator.name, False)
    
    if(using_fitness):

        gen_validationdata_from_metric_df(input_df, metrics, bucket_count, output_folder_name +"Overall_Data/",  collection_col_name, artefact_col_name, "Overall_Data",  True, fitness_metric )
    else:
        gen_validationdata_from_metric_df(input_df, metrics, bucket_count, output_folder_name +"Overall_Data/",  collection_col_name, artefact_col_name, "Overall_Data", False)
    
    gen_names = []
    if(len(generators)>0):
        for g in generators:
            gen_names.append(g.name)
    
        gen_scatterplot_for_all_metricpairs(input_df, metrics, output_folder_name, "Overall_Data", collection_col_name, gen_names)


