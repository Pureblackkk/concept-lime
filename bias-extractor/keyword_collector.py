import os
import pandas as pd
from hier_clustering import HierCluster

class KeywordCollector:
    def __init__(
        self,
        keyword_dir: str = None,
        save_dir: str = None,
    ):
        self.keyword_dir = keyword_dir
        self.save_dir = save_dir
        self.cluster = HierCluster()
    
    def __load_keyword_dir_from_src(self):
        if self.keyword_dir is None:
            raise ValueError('Keyword file directory location must be provided')
        
        file_list = os.listdir(self.keyword_dir)
        return file_list

    def __save_single_keyword_df(self, name_and_dataframe: (str, pd.DataFrame)):
        name, df = name_and_dataframe
        df.to_csv(os.path.join(self.save_dir, name), index=False)

    def save_keyword(self, with_cluster: bool = False):
        src_file_list = self.__load_keyword_dir_from_src()

        # Find if save directory exists
        # If not, then create one
        if not os.path.exists(self.save_dir):
            print(f"Directory to save: {self.save_dir} is not existing and will be created automatically")
            os.makedirs(self.save_dir, exist_ok=True)

        # Read each csv, then keep the supurious ones and save it
        # If has specificity, then only those < -0.1 could be selected
        for keyword_file in src_file_list:
            df = pd.read_csv(os.path.join(self.keyword_dir, keyword_file))
            df_supurious = df[(df['Bias'] == 'S') | (df['Bias'] == 'M')].copy()

            # Rule out sepcificity
            if 'Sepcificity' in df.columns:
                df_supurious = df_supurious[df_supurious['Sepcificity'] < 0]
            
            print(df_supurious)
            # Do clustering based on concept
            if with_cluster:
                # TODO: Change the clustering function
                df_supurious['class'] = self.cluster.clustering_in_test(df_supurious['Keyword'].tolist())

            # Save dataframe
            self.__save_single_keyword_df((keyword_file, df_supurious))

        print(f"All supurious keyword are written into {self.save_dir}")
