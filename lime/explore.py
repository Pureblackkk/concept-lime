import numpy as np
import pandas as pd
import statsmodels.api as sm
from k_lasso import k_lasso
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestRegressor
from typing import List
import matplotlib.pyplot as plt
from mpl_axes_aligner import align
from collections import defaultdict
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
    without_class: bool = True,
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

    y_col_1 = f'{DF_PRED_COL_PREFIX}{class_suffix}_{0}'
    y_col_2 = f'{DF_PRED_COL_PREFIX}{class_suffix}_{1}'

    cols = df[[y_col_1, y_col_2]].to_numpy()
    y_dummy = np.argmax(cols, axis = 1)

    # Get weights
    w = df[WEIGHTS_COL_NAME]

    return x, y, w, y_dummy

def run_k_lasso(
    df,
    keywords: list,
    K_num: int,
    investiate_class_num: int = 0,
    without_class: bool = True,
):
    x, y, w, _ = get_xy_from_elimate_keywords(
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
    print('Model summary ---', model.summary())

    # Rank the keyword
    coef = model.params
    sorted_coef = coef.sort_values(ascending=True).drop('const')
    
    # Transfer to dataframe
    sorted_coef_df = pd.DataFrame(sorted_coef).reset_index()
    sorted_coef_df.columns = ['Keyword', 'Coefficient']

    print('Sorted coef dataframe ---', sorted_coef_df)
    return sorted_coef_df

def weighted_linear_regression(x, y, w):  
    print('Sample size ----', x.shape[0])

    print(x)

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

def weighted_logistic_regression(x, y, w):
    print('Sample size ----', x.shape[0])

    # Fitting
    model = LogisticRegression()
    model.fit(x, y, sample_weight=w)

    # Rank the coefficiant
    print("Intercept:", model.intercept_)

    df = pd.DataFrame({
        'Keyword': x.columns.tolist(),
        'Coefficient': model.coef_[0]
    }).sort_values(by='Coefficient')

    print(df.to_string(index=False))

    return df

def get_effect_direction_for_decision_tree(model, x, y, feature_names):
    tree = model.tree_

    feature_scores = defaultdict(float)
    feature_weights = defaultdict(float)
    
    n_nodes = tree.node_count
    children_left = tree.children_left
    children_right = tree.children_right
    feature = tree.feature
    impurity = tree.impurity
    n_node_samples = tree.n_node_samples
    value = tree.value
    total_sample = x.shape[0]

    for node_id in range(n_nodes):
        if children_left[node_id] != children_right[node_id]:  # 非叶子节点
            f = feature[node_id]
            left = children_left[node_id]
            right = children_right[node_id]
            
            # 计算MSE减少量
            n_samples = n_node_samples[node_id]
            n_left = n_node_samples[left]
            n_right = n_node_samples[right]
            mse_parent = impurity[node_id]
            mse_left = impurity[left]
            mse_right = impurity[right]
            mse_decrease = mse_parent - (n_left / n_samples) * mse_left - (n_right / n_samples) * mse_right
            
            # 计算左右节点的预测值差异
            m_left = value[left][0, 0]
            m_right = value[right][0, 0]
            delta = m_right - m_left
            
            feature_scores[f] += delta * mse_decrease * (n_samples / total_sample)
            feature_weights[f] += mse_decrease
                
    # 输出每个特征的影响方向和信息增益
    for feature, score in feature_scores.items():
        print(f"Feature: {feature_names[feature]}, Score: {score:.4f}")
    
    for feature, weights in feature_weights.items():
        print(f"Feature: {feature_names[feature]}, Weights: {weights:.4f}")


def decision_tree(x, y, w=None, depth=5):
    reg_tree = DecisionTreeRegressor(max_depth=int(depth))
    reg_tree.fit(x, y, sample_weight=w)

    get_effect_direction_for_decision_tree(
        reg_tree,
        x.to_numpy(),
        y.to_numpy(),
        x.columns.to_list()
    )

    plt.figure(figsize=(10, 6))
    plot_tree(reg_tree, filled=True, rounded=True, feature_names=x.columns)
    plt.title("Weighted Decision Tree Regressor")
    plt.savefig('./test.jpg', dpi=300, bbox_inches='tight')

    df = pd.DataFrame({
        'Keyword': x.columns.tolist(),
        'Coefficient': reg_tree.feature_importances_
    }).sort_values(by='Coefficient')

    return df

def classify_decision_tree(x, y, w=None, depth=5):
    reg_tree = DecisionTreeClassifier(max_depth=int(depth))
    reg_tree.fit(x, y)

    plt.figure(figsize=(10, 6))
    plot_tree(reg_tree, filled=True, rounded=True, feature_names=x.columns)
    plt.title("Weighted Decision Tree Regressor")
    plt.savefig('./test.jpg', dpi=1000, bbox_inches='tight')

    df = pd.DataFrame({
        'Keyword': x.columns.tolist(),
        'Coefficient': reg_tree.feature_importances_
    }).sort_values(by='Coefficient')
    
    return df

def random_forest(x, y, w=None, depth=5):
    rf_regressor = RandomForestRegressor(n_estimators=3, max_depth=int(depth))
    rf_regressor.fit(x, y, sample_weight=w)

    df = pd.DataFrame({
        'Keyword': x.columns.tolist(),
        'Coefficient': rf_regressor.feature_importances_
    }).sort_values(by='Coefficient')
    
    return df

def combined_logistic_descision_tree(x, y, y_dummy, w=None):
    # Currently the depth will be 5
    decision_tree_df = decision_tree(x, y, w, '5')

    logistic_df = weighted_logistic_regression(x, y_dummy, w)

    sign = logistic_df['Coefficient'].apply(lambda x: 1 if x > 0 else -1)

    decision_tree_df['Coefficient'] = decision_tree_df['Coefficient'] * sign

    return decision_tree_df


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
        use_class_as_keywords,
    ):
        self.folder_name = self.__get_folder_name(
            keyword_score_path,
            res_csv_path
        )

        self.tag_name = f'expmodel:{model_type}_klasso:{str(klasso_num)}_hyclasskeywords{str(use_class_as_keywords)}_invesclass:{investiate_class_num}'

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


def preprocess_model_type(model_type: str):
    # Preprocess input model type
    decision_tree_re = r'(weighted_classify_decision_tree|decision_tree|weighted_decision_tree|weighted_random_forest)_(\w+)'

    match = re.match(decision_tree_re, model_type)

    if match:
        return (match.group(1), match.group(2))
    else:
        return (model_type, None)

def run_lime(**kwargs):
    # Deal with params
    keyword_score_path=kwargs.get('keyword_score_path')
    res_csv_path=kwargs.get('res_csv_path')
    model_type=kwargs.get('model_type')
    draw_plot=kwargs.get('draw_plot', True)
    elimate_keywords=kwargs.get('elimate_keywords', [])
    klasso_num=kwargs.get('klasso_num', None)
    investiate_class_num=kwargs.get('investiate_class_num', 0)
    without_class=kwargs.get('without_class', True)
    use_class_as_keywords=kwargs.get('use_class_as_keywords', False)


    # Read orignal keyword score
    keyword_score_df = pd.read_csv(keyword_score_path)

    if use_class_as_keywords:
        keywords = map(lambda x: str(x), list(keyword_score_df['class'].to_numpy()))
    else:
        keywords = list(keyword_score_df['Keyword'].to_numpy())

    # Read dataframe
    df = pd.read_csv(res_csv_path)

    # Clean the dataframe
    df = df[~df['prompt_without_class'].str.contains('\(', regex=True)]

    # Use k-lasso or already set elimate keywords
    eli_keywords = elimate_keywords
    if klasso_num:
        selected_keywords = run_k_lasso(
            df,
            keywords,
            klasso_num,
            investiate_class_num,
            without_class,
        )
        print(f'Use klasso num {klasso_num}, selected features: {selected_keywords}')
        eli_keywords = list(set(keywords) - set(selected_keywords))
    
    # Retrieve x, y, and weights
    x, y, w, y_dummy = get_xy_from_elimate_keywords(
        df,
        keywords,
        eli_keywords,
        investiate_class_num,
        without_class,
    )
    
    # Run explainable model
    res_df = None

    processed_model_type, param = preprocess_model_type(model_type)
    
    match processed_model_type:
        case 'simple_linear_regression':
            res_df = simple_linear_regression(x, y)
        case 'weighted_linear_regression':
            res_df = weighted_linear_regression(x, y, w)
        case 'decision_tree':
            res_df = decision_tree(x, y_dummy, None, param)
        case 'weighted_decision_tree':
            res_df = decision_tree(x, y, w, param)
        case 'weighted_classify_decision_tree':
            res_df = classify_decision_tree(x, y_dummy, w, param)
        case 'weighted_logistic_regression':
            res_df = weighted_logistic_regression(x, y_dummy, w)
        case 'weighted_random_forest':
            res_df = random_forest(x, y, w, param)
        case 'combined_logistic_descision_tree':
            res_df = combined_logistic_descision_tree(x, y, y_dummy, w)
        case _:
            raise ValueError('Wrong model type')
    
    # Assemble coef df and original keyword df
    if use_class_as_keywords:
        res_df = res_df.rename(columns={'Keyword': 'class'})
        res_df['class'] = res_df['class'].astype(int)

    assemble_df = pd.merge(keyword_score_df, res_df, on=('class' if use_class_as_keywords else 'Keyword'))
    print(assemble_df)
    assemble_df.to_csv('./temp_explore.csv', index=False)

    # Draw plot
    if draw_plot:
        plotDrawer = PlotDrawer(
            keyword_score_path,
            res_csv_path,
            model_type,
            klasso_num,
            investiate_class_num,
            use_class_as_keywords,
        )

        plotDrawer.draw_plot(assemble_df)

def reverted_image_classifed_res(src_dir):
    '''
    Print reverted image classificaiton result summary
    '''
    names = os.listdir(src_dir)
    classified_res = [name.split('#')[0] for name in names]
    total_num = len(classified_res)
    zero_count = classified_res.count('0')
    one_count = classified_res.count('1')

    print(f'zero num: {zero_count}, raito: {zero_count / total_num} ；；；；； one num: {one_count}, ratio: {one_count / total_num}')
    

if __name__ == '__main__':
    run_lime(
        keyword_score_path='./keyword/waterbird_gpt_cluster/waterbird_best_model_Waterbirds_erm_waterbird.csv',
        res_csv_path='./data/disturb_waterbird_concept_class_1000_Feb_14/waterbird_best_model_Waterbirds_erm_waterbird.csv',
        model_type='weighted_decision_tree_5',
        draw_plot=True,
        elimate_keywords=[],
        klasso_num=None,
        investiate_class_num=0,
        use_class_as_keywords=False,
    )

    # reverted_image_classifed_res(
    #     '/home/pureblackkkk/my_volume/concept-lime-data/revert_waterbird_concept_class_1000_Feb_14/gradacm_waterbird_best_model_Waterbirds_erm_waterbird_without_class'
    # )

