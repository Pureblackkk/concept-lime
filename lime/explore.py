import numpy as np
import pandas as pd
import statsmodels.api as sm
from k_lasso import k_lasso
from sklearn.linear_model import LinearRegression
from typing import List
import matplotlib.pyplot as plt
from mpl_axes_aligner import align
import seaborn as sns
import os
import re


DF_PRED_COL_PREFIX = 'model_pred_prompt_'
WEIGHTS_COL_NAME = 'cos_distance_weight_from_all'
IMG_SAVE_DIR = '/home/pureblackkkk/concept-lime/res_img'

def adjusted_weights(weights, is_scale: bool = True):
    adjusted_weights = weights

    if is_scale:
        adjusted_weights = - adjusted_weights ** 2 / (adjusted_weights.mean() ** 2)
        adjusted_weights = np.exp(adjusted_weights)
    
    return adjusted_weights

def get_xy_from_elimate_keywords(
    df,
    keywords: List[str],
    elimate_keywords: List[str] = [],
    investiate_class_num: int = 0,
    without_class: bool = True
):
    if elimate_keywords != None and len(elimate_keywords) > 0:
        for e_keyword in elimate_keywords:
            t = (df[e_keyword] == 1)
            df = df[~t]
        
        # Drop the column
        df = df.drop(columns=elimate_keywords)
    
    # Get x
    x = df[list(set(keywords) - set(elimate_keywords))]

    # Get y
    class_suffix = 'without_class' if without_class else 'with_class'
    y_col_name = f'{DF_PRED_COL_PREFIX}{class_suffix}_{investiate_class_num}'
    y = df[y_col_name]

    # Get weights
    w = df[WEIGHTS_COL_NAME]

    return x, y, w

def run_k_lasso(
    df,
    keywords: list,
    K_num: int,
    investiate_class_num: int = 0,
    without_class: bool = True,
):
    x, y, w = get_xy_from_elimate_keywords(
        df,
        keywords,
        [],
        investiate_class_num,
        without_class,
    )

    adj_w = adjusted_weights(w)

    features = k_lasso(x, y, adj_w.to_numpy(), K_num)
    
    return [keywords[item] for item in features]

def simple_linear_regression(x, y):
    print('Sample size ---', x.shape[0])
    # Run the simple linear regression
    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()
    print(model.summary())

    # Rank the keyword
    coef = model.params
    sorted_coef = coef.sort_values(ascending=True)
    
    # Transfer to dataframe
    sorted_coef_df = pd.DataFrame(sorted_coef).reset_index()
    sorted_coef_df.columns = ['Keyword', 'Coefficient']

    print(sorted_coef_df)
    return sorted_coef_df

def weighted_linear_regression(x, y, w):  
    print('Sample size ----', x.shape[0])

    # Fitting
    model = LinearRegression()
    model.fit(x, y, sample_weight=w)

    # Rank the coefficiant
    print("Intercept:", model.intercept_)
    
    df = pd.DataFrame({
        'Keyword': x.columns.tolist(),
        'Coefficient': model.coef_
    }).sort_values(by='Coefficient')

    print(df.to_string(index=False))

    return df

class PlotDrawer:
    def __init__(
        self,
        # Root folder
        keyword_score_path,
        res_csv_path,
        # sub tag info
        model_type,
        klasso_num,
        investiate_class_num,
    ):
        self.folder_name = self.__get_folder_name(
            keyword_score_path,
            res_csv_path
        )

        self.tag_name = f'expmodel:{model_type}_klasso:{str(klasso_num)}_invesclass:{investiate_class_num}'

    def __get_folder_name(self, keyword_score_path, res_csv_path):
        keyword_name_pattern = r"/keyword/([^/]+)/"
        disturb_name_pattern = r"/data/([^/]+)/"
        model_name_pattern = r"/([^/]+).csv$"

        match_keyword = re.search(keyword_name_pattern, keyword_score_path)
        match_disturb = re.search(disturb_name_pattern, res_csv_path)
        match_model = re.search(model_name_pattern, res_csv_path)

        if not(match_keyword and match_disturb and match_model):
            raise ValueError('Wrong file name for painter')

        folder_name = f'{match_keyword.group(1)}##{match_disturb.group(1)}##{match_model.group(1)}'

        print(folder_name)

        return folder_name
    
    def __save_img(self, name):
        # Create dir if not exists
        save_dir = os.path.join(IMG_SAVE_DIR, self.folder_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        # Create final save path
        save_path = os.path.join(
            save_dir,
            f'{name}_{self.tag_name}.png'
        )

        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    def __draw_score_coefficent_(self, df):
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.barh(df["Keyword"], df["Score"], color='blue', alpha=0.6, label='Score')
        ax1.set_xlabel("Score")
        ax1.set_ylabel("Keyword")

        ax2 = ax1.twiny()
        ax2.barh(df["Keyword"], df["Coefficient"], color='red', alpha=0.6, label='Coefficient')
        ax2.set_xlabel("Coefficient")

        ax1.legend(loc="upper right")
        ax2.legend(loc="upper left")

        align.xaxes(ax1, 0, ax2, 0, 0.5)

        plt.title("Keyword Score and Coefficent")
        self.__save_img('keyword_score_coef')
    
    def __draw_acc_coef(self, df):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        sns.regplot(x=df['Score'], y=df['Acc.'], ax=axes[0], scatter_kws={'color': 'blue'}, line_kws={'color': 'blue'})
        axes[0].set_xlabel("Score")
        axes[0].set_ylabel("Acc.")
        axes[0].set_title("Score & Acc with Linear Fit")

        corr_matrix1 = np.corrcoef(df['Score'], df['Acc.'])
        axes[0].text(1, 0, f'PCCs: {corr_matrix1[0, 1]:.2f}', color='blue', fontsize=12, transform=axes[0].transAxes)

        sns.regplot(x=df['Coefficient'], y=df['Acc.'], ax=axes[1], scatter_kws={'color': 'red'}, line_kws={'color': 'red'})
        axes[1].set_xlabel("Coefficient Value")
        axes[1].set_ylabel("Acc.")
        axes[1].set_title("Coefficient & Acc with Linear Fit")

        corr_matrix2 = np.corrcoef(df['Coefficient'], df['Acc.'])
        axes[1].text(1, 0, f'PCCs: {corr_matrix2[0, 1]:.2f}', color='red', fontsize=12, transform=axes[1].transAxes)

        plt.tight_layout()
        self.__save_img('score&coef_acc')

    def draw_plot(self, df):
        self.__draw_score_coefficent_(df)
        self.__draw_acc_coef(df)


def run_lime(
    keyword_score_path: str,
    res_csv_path: str,
    model_type: str = 'simple_linear_regression',
    draw_plot: bool = True,
    elimate_keywords: List[str] = [],
    klasso_num: int = None,
    investiate_class_num: int = 0,
    without_class: bool = True,
):
    # Read orignal keyword score
    keyword_score_df = pd.read_csv(keyword_score_path)
    keywords = list(keyword_score_df['Keyword'].to_numpy())

    # Read dataframe
    df = pd.read_csv(res_csv_path)

    # Use k-lasso or already set elimate keywords
    eli_keywords = elimate_keywords
    if klasso_num:
        selected_keywords = run_k_lasso(
            df,
            keywords,
            klasso_num,
            investiate_class_num,
            without_class
        )
        print(f'Use klasso num {klasso_num}, selected features: {selected_keywords}')
        eli_keywords = list(set(keywords) - set(selected_keywords))
    
    # Retrieve x, y, and weights
    x, y, w = get_xy_from_elimate_keywords(
        df,
        keywords,
        eli_keywords,
        investiate_class_num,
        without_class
    )
    
    # Run explainable model
    res_df = None

    match model_type:
        case 'simple_linear_regression':
            res_df = simple_linear_regression(x, y)
        case 'weighted_linear_regression':
            res_df = weighted_linear_regression(x, y, w)
        case _:
            raise ValueError('Wrong model type')
    
    # Assemble coef df and original keyword df
    assemble_df = pd.merge(keyword_score_df, res_df, on='Keyword')
    print(assemble_df)

    # Draw plot
    if draw_plot:
        plotDrawer = PlotDrawer(
            keyword_score_path,
            res_csv_path,
            model_type,
            klasso_num,
            investiate_class_num,
        )

        plotDrawer.draw_plot(assemble_df)

if __name__ == '__main__':
    run_lime(
        './keyword/celebA_keyword/celeba_best_model_CelebA_erm_blond.csv',
        './data/disturb_celebA_equal_2000_Feb_7/celeba_best_model_CelebA_erm_blond.csv',
        'weighted_linear_regression',
        True,
        [],
        None,
        0,
    )


