from scipy.stats import zscore
import matplotlib.pyplot as plt

import re, random
def list_sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    def tryint(s):
        try:
            return int(s)
        except:
            return s
    def alphanum_key(s):
        """ Turn a string into a list of string and number chunks.
            "z23a" -> ["z", 23, "a"]
        """
        return [ tryint(c) for c in re.split('([0-9]+)', s) ]
    l.sort(key=alphanum_key)
    return l


#--------
import scipy as sp
def get_residual_plot(Y_TRUE, Y_PRED):
    resid = Y_PRED- Y_TRUE
    plt.scatter(Y_TRUE, resid, marker = ".", linewidth = 1, color = "k")
#     plt.hlines( y = 0, xmin = 1, xmax = 7)
    plt.xlabel("TRUE")
    plt.ylabel("residual")
    plt.title(tr_num)
    plt.show()
    
    plt.scatter(Y_TRUE, resid, marker = ".", linewidth = 1, color = "k")
    plt.xlabel("TRUE")
    plt.ylabel("residual")
    plt.title(tr_num)
    plt.show()
    print(abs(resid).max(), abs(resid).mean(), resid.std(), abs(resid).median())
    
    plt.hist(resid, edgecolor ="k" , bins = 20)
    plt.title(tr_num)
    plt.show()
    
    sp.stats.probplot(resid, plot = plt)
    plt.axis("equal")
    plt.show()
    # skewness test
    
    
#---------
import numpy as np
from math import sqrt
from sklearn.metrics import r2_score, max_error, median_absolute_error
def get_RMSE(Y_TRUE, Y_PRED):
    mse = np.average((Y_PRED - Y_TRUE)**2)
    rmse = sqrt(mse)
    return rmse

def get_MAE(Y_TRUE, Y_PRED):
    mae = np.average(abs(Y_PRED - Y_TRUE))
    return mae

def get_max_error(Y_TRUE, Y_PRED):
    max_er = max_error(Y_PRED , Y_TRUE)
    return max_er
def get_uncentered_R_squared(Y_TRUE, Y_PRED):
    SS_tot_uncentered = (Y_PRED**2).sum()
    SS_res = ((Y_PRED - Y_TRUE)**2).sum()

    R_squared_uncentered = 1- SS_res/SS_tot_uncentered
    return R_squared_uncentered

def get_CCC(Y_TRUE, Y_PRED):

    x_bar = np.mean(Y_TRUE)
    y_bar = np.mean(Y_PRED)

    var_x = ((Y_TRUE - x_bar)**2).sum() / len(Y_TRUE)
    var_y = ((Y_PRED - y_bar)**2).sum() / len(Y_PRED)

    covar = ((Y_TRUE - x_bar)*(Y_PRED - y_bar)).sum()/ len(Y_TRUE)

    rho_c= (2 * covar) / (var_x + var_y + (x_bar - y_bar)**2)
    
    return rho_c



def get_evaluation_metrics(Y_TRUE, Y_PRED):
#     print(tr_num, "\n", _target_col)
    rmse_value = get_RMSE(Y_TRUE, Y_PRED)
    
    mae_value = get_MAE(Y_TRUE, Y_PRED)
    abs_max_error_value = get_max_error(Y_TRUE, Y_PRED)
    median_abs_error_value = median_absolute_error(Y_TRUE, Y_PRED)
    
    r2_scipy_value = r2_score(Y_TRUE, Y_PRED)
    
#     correlation_matrix = np.corrcoef(Y_TRUE, Y_PRED)
#     correlation_xy = correlation_matrix[0,1]
#     r2_inverse = correlation_xy**2
    r2_inverse_value = r2_score( Y_PRED, Y_TRUE)
    resid = Y_PRED - Y_TRUE
    r2_uncentered = get_uncentered_R_squared( Y_TRUE, Y_PRED)
    Lin_concor_value = get_CCC(Y_TRUE, Y_PRED)
    
    print("RMSE: ", rmse_value)
    print("MAE: ", mae_value)
    print("R2_scipy: ", r2_scipy_value)
    print("R2_scipy_inverse:", r2_inverse_value)
    print("R2_uncentered: ", r2_uncentered)
    print("Lin Concordance correlation coefficient: ", Lin_concor_value)
    
    print("\nmax resid: ", abs(resid).max())
    print("mean resid: ", abs(resid).mean())
    print("std resid: ", resid.std())
    print("median resid: ", abs(resid).median())
    
    return rmse_value, mae_value, abs_max_error_value, median_abs_error_value, Lin_concor_value, r2_scipy_value, r2_inverse_value


# def get_evalutation_metrics(Y_TRUE, Y_PRED):
# #     print(tr_num, "\n", _target_col)
#     rmse = get_RMSE(Y_TRUE, Y_PRED)
    
#     mae_value = get_MAE(Y_TRUE, Y_PRED)
#     abs_max_error_value = get_max_error(Y_TRUE, Y_PRED)
#     median_abs_error_value = median_absolute_error(Y_TRUE, Y_PRED)
#     r2_scipy_value = r2_score(Y_TRUE, Y_PRED)
    
#     correlation_matrix = np.corrcoef(Y_TRUE, Y_PRED)
#     correlation_xy = correlation_matrix[0,1]
#     r2_cal = correlation_xy**2
    
#     resid = Y_PRED - Y_TRUE
    
#     print("RMSE: ", rmse)
#     print("MAE: ", mae_value)
#     print("R2_scipy: ", r2_scipy_value)
#     print("R2_cal:", r2_cal)
    
#     print("\nmax resid: ", abs(resid).max())
#     print("mean resid: ", abs(resid).mean())
#     print("std resid: ", resid.std())
#     print("median resid: ", abs(resid).median())
    
#     return rmse, mae_value, r2_scipy_value, r2_cal, abs_max_error_value, median_abs_error_value


# =============================================================================
# # Bland Altman Plot by sex
# =============================================================================
def get_Bland_Altman_by_sex(target_df, col_measure1, col_measure2, check_criteria = 1.96, modified = False):
    import statsmodels.api as sm
    import numpy as np
    import matplotlib.pyplot as plt
    
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    target_measure_1 =col_measure1
    target_measure_2 = col_measure2

#     df_target =target_df.copy() 
    ax_ylim = (target_df[target_measure_1] -target_df[target_measure_2]).abs().max()
    ax_xlim_min = np.round(target_df[target_measure_1].min(), 2)
    ax_xlim_max = np.round(target_df[target_measure_1].max(), 2)
    if modified == False:
        #----------------------------------------------------------------
        f, ax = plt.subplots(1,3, figsize = (21,5))   
        f.suptitle("Bland-Altman Plot\n- First Visit 2018 - weak external validation set", fontsize = 20)






        for _i, (_sex, _color, _marker) in enumerate(zip(["F", "M"],colors, ["+", "x"])) :
            df_sub = target_df[target_df["sex"] == _sex]

            # version 1
        #     m1 =df_sub['FVC_MEAS ']
        #     m2 = df_sub['modelpredicted(FVC_MEAS )']
            # version 2
            m2 =df_sub[target_measure_1]
            m1 = df_sub[target_measure_2]    

            sm.graphics.mean_diff_plot(m1, m2, ax = ax[_i],
                                       sd_limit = check_criteria, 
                                    scatter_kwds = {"marker":_marker, 
                                                     "color":_color, 
                                                     "alpha":0.7})
            ax[_i].set_xlabel(f"average({target_measure_1}, {target_measure_1} model estimation)", fontsize = 12)
            ax[_i].set_title(_sex, fontsize = 15)
            ax[_i].set_ylim(-ax_ylim, ax_ylim)
            ax[_i].set_xlim(ax_xlim_min, ax_xlim_max)




            #----------------------------------------------------------------------
            means = np.mean([m1, m2], axis=0)
            diffs = m1 - m2
            mean_diff = np.mean(diffs)
            std_diff = np.std(diffs, axis=0)

            ax[2].scatter(means, diffs, 
                         marker = _marker,
                         color = _color, 
                         alpha = 0.7, label = _sex)
            ax[2].hlines(mean_diff,xmin =ax_xlim_min, xmax = ax_xlim_max,
                        color = _color, 
                        linestyle = "--", linewidth = 1)
            ax[2].hlines(mean_diff + check_criteria *std_diff,xmin =ax_xlim_min, xmax = ax_xlim_max,
                        color = _color,
                         linestyle = "--", linewidth = 1)
            ax[2].hlines(mean_diff - check_criteria *std_diff,xmin =ax_xlim_min, xmax = ax_xlim_max,
                        color = _color,
                        linestyle = "--", linewidth = 1)

        ax[1].set_ylabel("")  
        ax[2].set_ylabel("")

        ax[2].tick_params(labelsize = 13)
        ax[2].legend()
        ax[2].set_xlabel(f"average({target_measure_1}, {target_measure_1} model estimation)", fontsize = 12)
        ax[2].set_ylim(-ax_ylim,ax_ylim)
        ax[2].set_xlim(ax_xlim_min, ax_xlim_max)

        f.tight_layout()   
    elif modified == True:
        f, ax = plt.subplots(1,3, figsize = (21,5))   
#         f.suptitle("Modified Bland-Altman Plot(Redidual)\n- First Visit 2018 - weak external validation set - ", fontsize = 20)


        for _i, (_sex, _color, _marker) in enumerate(zip(["F", "M"],colors, ["+", "x"])) :
            df_sub = target_df[target_df["성별"] == _sex]
            x =df_sub[target_measure_1]
            m2 =df_sub[target_measure_1]
            m1 = df_sub[target_measure_2]    
                  
            diffs = m1 - m2
            mean_diff = np.mean(diffs)
            std_diff = np.std(diffs, axis=0)      
                  
                  
            ax[_i].scatter(x, diffs , marker = _marker, color = _color, alpha = 0.7, s = 20)


            ax[_i].hlines(y =mean_diff ,
                          xmin = ax_xlim_min ,
                          xmax = ax_xlim_max, 
                          linestyle = "solid",
                          color = _color,
                          linewidth = 1)

            ax[_i].hlines(y = mean_diff + check_criteria * std_diff ,
                          xmin = ax_xlim_min ,
                          xmax = ax_xlim_max, 
                          linestyle = "--",
                          color = _color,
                          linewidth = 1)
            ax[_i].hlines(y = mean_diff - check_criteria * std_diff ,
                          xmin = ax_xlim_min ,
                          xmax = ax_xlim_max, 
                          linestyle = "--",
                          color = _color,
                          linewidth = 1)
            
            # Annotate mean line with mean difference.
            ax[_i].annotate('mean diff:\n{}'.format(np.round(mean_diff, 2)),
                        xy=(0.99, 0.5),
                        horizontalalignment='right',
                        verticalalignment='center',
                        fontsize=14,
                        xycoords='axes fraction')
            upper = mean_diff + check_criteria * std_diff
            lower = mean_diff - check_criteria * std_diff
            ax[_i].annotate('-SD{}: {}'.format(check_criteria, np.round(lower, 2)),
                    xy=(0.99, 0.07),
                    horizontalalignment='right',
                    verticalalignment='bottom',
                    fontsize=14,
                    xycoords='axes fraction')
            ax[_i].annotate('+SD{}: {}'.format(check_criteria, np.round(upper, 2)),
                    xy=(0.99, 0.92),
                    horizontalalignment='right',
                    fontsize=14,
                    xycoords='axes fraction')




            ax[_i].set_xlabel(target_measure_1, fontsize = 12)
            ax[_i].set_title(_sex, fontsize = 15)
            ax[_i].set_ylim(-ax_ylim, ax_ylim)
            ax[_i].set_xlim(ax_xlim_min, ax_xlim_max)

            ax[2].scatter(x, diffs , marker = _marker, color = _color, alpha = 0.7, 
                          s = 20,
                         label = _sex)
            ax[2].hlines(y =mean_diff ,
                          xmin = ax_xlim_min ,
                          xmax = ax_xlim_max, 
                          linestyle = "solid",
                          color = _color,
                          linewidth = 1)

            ax[2].hlines(y =  mean_diff + check_criteria * std_diff,
                          xmin = ax_xlim_min ,
                          xmax = ax_xlim_max, 
                          linestyle = "--",
                          color = _color,
                          linewidth = 1)
            ax[2].hlines(y =  mean_diff - check_criteria * std_diff ,
                          xmin = ax_xlim_min ,
                          xmax = ax_xlim_max, 
                          linestyle = "--",
                          color = _color,
                          linewidth = 1)

            ax[_i].tick_params(labelsize = 13)
        ax[2].tick_params(labelsize = 13) 
        ax[2].set_xlabel(target_measure_1, fontsize = 12)
        ax[2].set_ylim(-ax_ylim, ax_ylim)
        ax[2].set_xlim(ax_xlim_min, ax_xlim_max)
        ax[2].legend()

        ax[0].set_ylabel("Difference(estimation  - measured)", fontsize = 15)
        ax[1].set_ylabel("")
        ax[2].set_ylabel("")
        f.tight_layout()
        
    return f, ax
def get_Bland_Altman_by_sex_v2(target_df, col_measure1, col_measure2, check_criteria = 1.96, modified = False):
    import statsmodels.api as sm
    import numpy as np
    import matplotlib.pyplot as plt
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    target_measure_1 =col_measure1
    target_measure_2 = col_measure2
    target_str = target_measure_1.split("_")[0]

    df_target =target_df.copy() 
    ax_ylim = (df_target[target_measure_1] -df_target[target_measure_2]).abs().max()
    ax_xlim_min = np.round(df_target[target_measure_1].min(), 2)
    ax_xlim_max = np.round(df_target[target_measure_1].max(), 2)
#----------------------
    M2_total = df_target[target_measure_1]
    M1_total = df_target[target_measure_2]
    
    diffs_total = M1_total - M2_total 
    mean_diff_total = np.mean(diffs_total)
    std_diff_total = np.std(diffs_total, axis = 0)
    upper_total = mean_diff_total + check_criteria * std_diff_total
    lower_total = mean_diff_total - check_criteria * std_diff_total
    print(mean_diff_total,std_diff_total,upper_total  , lower_total)
#-----------------
    if modified == False:
        #----------------------------------------------------------------
        f, ax = plt.subplots(1,3, figsize = (21,5))   
#         f.suptitle("Bland-Altman Plot\n- First Visit 2018 - weak external validation set", fontsize = 20)


        for _i, (_sex, _color, _marker) in enumerate(zip(["F", "M"],colors, ["+", "x"])) :
            df_sub = target_df[target_df["성별"] == _sex]
            if _sex == "F":
                ax_title = "Female"
            else:
                ax_title = "Male"
            # version 1
        #     m1 =df_sub['FVC_MEAS ']
        #     m2 = df_sub['modelpredicted(FVC_MEAS )']
            # version 2
            m2 =df_sub[target_measure_1]
            m1 = df_sub[target_measure_2]    

            sm.graphics.mean_diff_plot(m1, m2, ax = ax[_i],
                                       sd_limit = check_criteria, 
                                    scatter_kwds = {"marker":_marker, 
                                                     "color":_color, 
                                                     "alpha":0.7})
            ax[_i].set_xlabel(f"Mean {target_str} (L)\n(actual + predicted)/2", fontsize = 13)
            ax[_i].set_title(ax_title, fontsize = 15)
            ax[_i].set_ylim(-ax_ylim, ax_ylim)
            ax[_i].set_xlim(ax_xlim_min, ax_xlim_max)




            #----------------------------------------------------------------------
            means = np.mean([m1, m2], axis=0)
            diffs = m1 - m2
            mean_diff = np.mean(diffs)
            std_diff = np.std(diffs, axis=0)

            ax[2].scatter(means, diffs, 
                         marker = _marker,
                         color = _color, 
                         alpha = 0.7, label = _sex)
            

        # draw line
        ax[2].hlines(mean_diff_total,xmin =ax_xlim_min, xmax = ax_xlim_max,
                    color = "k", 
                    linestyle = "--", linewidth = 1)
        ax[2].hlines(mean_diff_total + check_criteria *std_diff_total,xmin =ax_xlim_min, xmax = ax_xlim_max,
                    color = "k",
                     linestyle = "--", linewidth = 1)
        ax[2].hlines(mean_diff_total - check_criteria *std_diff_total,xmin =ax_xlim_min, xmax = ax_xlim_max,
                    color = "k",
                    linestyle = "--", linewidth = 1)
        
        # anootate 
        ax[2].annotate('mean diff:\n{}'.format(np.round(mean_diff_total, 2)),
                    xy=(0.99, 0.5),
                    horizontalalignment='right',
                    verticalalignment='center',
                    fontsize=14,
                    xycoords='axes fraction')

        ax[2].annotate('-SD{}: {}'.format(check_criteria, np.round(lower_total, 2)),
                xy=(0.99, 0.07),
                horizontalalignment='right',
                verticalalignment='bottom',
                fontsize=14,
                xycoords='axes fraction')
        ax[2].annotate('+SD{}: {}'.format(check_criteria, np.round(upper_total, 2)),
                xy=(0.99, 0.92),
                horizontalalignment='right',
                fontsize=14,
                xycoords='axes fraction')

        
        
        

        ax[1].set_ylabel("")  
        ax[2].set_ylabel("")
        ax[0].set_ylabel("Differences\n(predicted - actual)")

        ax[2].tick_params(labelsize = 13)
        ax[2].legend(loc = "upper left")
        ax[2].set_xlabel(f"Mean {target_str} (L)\n(actual + predicted)/2", fontsize = 13)
        ax[2].set_ylim(-ax_ylim,ax_ylim)
        ax[2].set_xlim(ax_xlim_min, ax_xlim_max)

        f.tight_layout()   
    elif modified == True:
        f, ax = plt.subplots(1,3, figsize = (21,5))   
#         f.suptitle("Modified Bland-Altman Plot(Redidual)\n- First Visit 2018 - weak external validation set - ", fontsize = 20)


        for _i, (_sex, _color, _marker) in enumerate(zip(["F", "M"],colors, ["+", "x"])) :
            df_sub = df_target[df_target["성별"] == _sex]
            x =df_sub[target_measure_1]
            m2 =df_sub[target_measure_1]
            m1 = df_sub[target_measure_2]    
                  
            diffs = m1 - m2
            mean_diff = np.mean(diffs)
            std_diff = np.std(diffs, axis=0)      
             
                  
            ax[_i].scatter(x, diffs , marker = _marker, color = _color, alpha = 0.7, s = 20)


            ax[_i].hlines(y =mean_diff ,
                          xmin = ax_xlim_min ,
                          xmax = ax_xlim_max, 
                          linestyle = "solid",
                          color = _color,
                          linewidth = 1)

            ax[_i].hlines(y = mean_diff + check_criteria * std_diff ,
                          xmin = ax_xlim_min ,
                          xmax = ax_xlim_max, 
                          linestyle = "--",
                          color = _color,
                          linewidth = 1)
            ax[_i].hlines(y = mean_diff - check_criteria * std_diff ,
                          xmin = ax_xlim_min ,
                          xmax = ax_xlim_max, 
                          linestyle = "--",
                          color = _color,
                          linewidth = 1)
            
            # Annotate mean line with mean difference.
            ax[_i].annotate('mean diff:\n{}'.format(np.round(mean_diff, 2)),
                        xy=(0.99, 0.5),
                        horizontalalignment='right',
                        verticalalignment='center',
                        fontsize=14,
                        xycoords='axes fraction')
            upper = mean_diff + check_criteria * std_diff
            lower = mean_diff - check_criteria * std_diff
            ax[_i].annotate('-SD{}: {}'.format(check_criteria, np.round(lower, 2)),
                    xy=(0.99, 0.07),
                    horizontalalignment='right',
                    verticalalignment='bottom',
                    fontsize=14,
                    xycoords='axes fraction')
            ax[_i].annotate('+SD{}: {}'.format(check_criteria, np.round(upper, 2)),
                    xy=(0.99, 0.92),
                    horizontalalignment='right',
                    fontsize=14,
                    xycoords='axes fraction')




            ax[_i].set_xlabel(target_measure_1, fontsize = 12)
            ax[_i].set_title(_sex, fontsize = 15)
            ax[_i].set_ylim(-ax_ylim, ax_ylim)
            ax[_i].set_xlim(ax_xlim_min, ax_xlim_max)
            ax[_i].tick_params(labelsize = 13)


        ax[2].hlines(mean_diff_total,xmin =ax_xlim_min, xmax = ax_xlim_max,
                    color = "k", 
                    linestyle = "--", linewidth = 1)
        ax[2].hlines(mean_diff_total + check_critiera *std_diff_total,xmin =ax_xlim_min, xmax = ax_xlim_max,
                    color = "k",
                     linestyle = "--", linewidth = 1)
        ax[2].hlines(mean_diff_total - check_critiera *std_diff_total,xmin =ax_xlim_min, xmax = ax_xlim_max,
                    color = "k",
                    linestyle = "--", linewidth = 1)
        
        # anootate 
        ax[2].annotate('mean diff:\n{}'.format(np.round(mean_diff_total, 2)),
                    xy=(0.99, 0.5),
                    horizontalalignment='right',
                    verticalalignment='center',
                    fontsize=14,
                    xycoords='axes fraction')
        ax[2].annotate('-SD{}: {}'.format(check_criteria, np.round(lower_total, 2)),
                xy=(0.99, 0.07),
                horizontalalignment='right',
                verticalalignment='bottom',
                fontsize=14,
                xycoords='axes fraction')
        ax[2].annotate('+SD{}: {}'.format(check_criteria, np.round(upper_total, 2)),
                xy=(0.99, 0.92),
                horizontalalignment='right',
                fontsize=14,
                xycoords='axes fraction')


        ax[2].tick_params(labelsize = 13) 
        ax[2].set_xlabel(target_measure_1, fontsize = 12)
        ax[2].set_ylim(-ax_ylim, ax_ylim)
        ax[2].set_xlim(ax_xlim_min, ax_xlim_max)
        ax[2].legend(loc = "upper left")

        ax[0].set_ylabel("Difference(estimation  - measured)", fontsize = 15)
        ax[1].set_ylabel("")
        ax[2].set_ylabel("")
        f.tight_layout()
        
    return f, ax





def density_estimation(m1, m2):
    from scipy.stats import gaussian_kde
    X, Y = np.mgrid[m1.min():m1.max():100j, m2.min():m2.max():100j]                                                     
    positions = np.vstack([X.ravel(), Y.ravel()])                                                       
    values = np.vstack([m1, m2])                                                                        
    kernel = gaussian_kde(values)                                                                 
    Z = np.reshape(kernel(positions).T, X.shape)
    return X, Y, Z


from sklearn.metrics import confusion_matrix




def calculate_accuracy(y_truth, y_pred, class_names):

    
    
    plot_confusion_matrix(confusion_matrix(y_truth, y_pred), classes=class_names, title='Confusion matrix')
    
    #return sen, 1-spec
    return sen, spec

def plot_confusion_matrix(cm, classes_true,classes_pred,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        #print('Confusion matrix, without normalization')
        print('Confusion matrix')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize='x-large')
    plt.colorbar()
    tick_marks = np.arange(len(classes_true))
    #plt.xticks(tick_marks, classes, rotation=45)
    plt.xticks(tick_marks, classes_true, fontsize='large')
    plt.yticks(tick_marks, classes_pred, fontsize='large', rotation=90)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", size='xx-large')

    plt.tight_layout()
    plt.ylabel('True label', fontsize='large')
    plt.xlabel('Predicted label', fontsize='large')
    plt.show()
    
# calculate_accuracy(y,pred, ['never', 'current'])   