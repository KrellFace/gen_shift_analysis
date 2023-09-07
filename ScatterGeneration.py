from datetime import *
import pandas as pd
import numpy as np
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt

color_dict = dict({
                #Dark Grey
                #0:[87/ 255.0, 87/ 255.0, 87/ 255.0],

                
                #Light Green
                0: [129/ 255.0, 197/ 255.0, 122/ 255.0],
                #Light Blue
                1: [157/ 255.0, 175/ 255.0, 255/ 255.0],
                #Red
                2: [173/ 255.0, 35/ 255.0, 35/ 255.0],
                #Green
                3: [29/ 255.0, 105/ 255.0, 20/ 255.0],
                #Blue
                4: [42/ 255.0, 75/ 255.0, 215/ 255.0],
                #Cyan
                5:[41/ 255.0, 208/ 255.0, 208/ 255.0],
                #Pink
                6: [255/ 255.0, 205/ 255.0, 243/ 255.0],
                #Purple
                7: [129/ 255.0, 38/ 255.0, 192/ 255.0],
                #Orange
                8: [255/ 255.0, 146/ 255.0, 51/ 255.0]})


output_fold = 'output/scatterplots/'

def generate_era_scatterplot(input_df, metric1, metric2, file_name, plot_title, gen_column_name = "Generator", multiple_generators = False, generator_list = [], arrows = [], highlight_arrows = []):

    
    #df_metricdata = pd.read_csv(inputFile)

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    
    ax.set_xlabel(metric1, fontsize = 35)
    ax.set_ylabel(metric2, fontsize = 35)
        
    title = f"{metric1}-{metric2} ERA Scatterplot"

    #Color each generators points differently if we are running for multiple alternatives

    plot_col = 0
    gen_list = []
    if(multiple_generators):
        for generator in generator_list:
            gen_list.append(generator)
            
            if(plot_col>len(color_dict)-1):
                plot_col=0
            else:

                rgb = color_dict[plot_col]
            plot_col+=1 
            #Limit our targets to just current generator
            to_keep = input_df[gen_column_name] == generator

            rand_z = []
            for i in range(len(to_keep.index)):
                rand_z.append(np.random.random())


            ax.scatter(input_df.loc[to_keep, metric1]
                        , input_df.loc[to_keep, metric2]
                        , c = [rgb]
                        , alpha = 0.8
                        , s = .3)
    else:
        rgb = color_dict[3]
        #Limit our targets to just current generator
        ax.scatter(input_df.loc[:, metric1]
                    , input_df.loc[:, metric2]
                    , c = [rgb]
                    , alpha = 0.1
                    , s = 8)

    #If we passed in arrows, plot them here
    for arrow in arrows:
        plt.arrow(arrow[0],arrow[1],arrow[2],arrow[3], color = 'k', width = 0.001, length_includes_head = True, head_width = 0.06, head_length = 0.06)
    
    for arrow in highlight_arrows:
        plt.arrow(arrow[0],arrow[1],arrow[2],arrow[3], color = color_dict[plot_col], width = 0.03, length_includes_head = True, head_width = 0.2, head_length = 0.2)

    #ax.set_xlim(xmin=0.3)
    #ax.set_ylim(ymin=75)
    #ax.set_xlim(-5,7)
    #ax.set_ylim(-4,8)
    ax.spines['left'].set_color("black")
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_color("black")
    ax.spines['bottom'].set_linewidth(0.5)
    lgnd = ax.legend(gen_list)
    lgnd.legendHandles[0]._sizes = [30]
    lgnd.legendHandles[1]._sizes = [30]
    ax.grid(color = "black", linestyle = "dashed", linewidth = 0.1)
    ax.set_facecolor((1.0, 1.0,1.0))
    #plt.savefig(f"{output_fold}{metric1.name},{metric2.name} "+file_name)
    plt.title(plot_title, fontsize = 30)
    plt.savefig(file_name,dpi=300, bbox_inches="tight")
    plt.close()



