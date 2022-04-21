from dataclasses import dataclass
import fiftyone as fo
import fiftyone.zoo as foz 
import os, json
import numpy as np
import pandas as pd
from fiftyone.core.dataset import get_default_dataset_dir, Dataset


###############################################
############                    ################
#########        UNFINISHED        ##############
#########        UNFINISHED        ############## 
#########        UNFINISHED        ##############
#########        UNFINISHED        ##############
#########        UNFINISHED        ##############
############                    ################
###############################################


@dataclass
class COCO:
    dir: str
    classmap: dict
    
    def get_train_test_val_datadirs(self, fiftyone_default_dataset):
        """
        Assuming a fiftyone dataset object is passed in,
        this method will return the folders associated with those directories
        ARGS:
            fiftyone_default_dataset -> fiftyone.core.dataset.Dataset
                Dataset singleton instance object thing. I'm still wrapping my head around it to be honest.
        RETURNS:
            _train_dir -> str
                path to train directory (not data folder directly, but the json file location)
            _test_dir -> str
                path to test directory (not data folder directly, but the json file location)
            _val_dir -> str
                path to validation directory (not data folder directly, but the json file location)            
        """
        
        datadir = get_default_dataset_dir(fiftyone_default_dataset)
        origin_cwd = globals()['_dh'][0]
        
        os.chdir(datadir)

        _train_dir = f'{os.getcwd()}/train'
        _test_dir = f'{os.getcwd()}/test'
        _val_dir = f'{os.getcwd()}/validation'
        
        os.chdir(origin_cwd)
        
        return _train_dir, _test_dir, _val_dir



    # TODO docstring
    @staticmethod
    def convert_to_df_with_bbox(json_file, path, classmap):
        json_file = json.loads(json_file)
        
        filename_map = {}
        classes = classmap.keys()
        
        image_info = pd.DataFrame(json_file['annotations'])
        dim_info = pd.DataFrame(json_file['images'])

        image_info = image_info[image_info['category_id'].isin(classes)].replace(to_replace = {'category_id': classmap})
        
        image_info = image_info.sort_values('image_id')
        dim_info = dim_info.sort_values('file_name')
        list_of_filenames = dim_info['file_name'].to_list()

        
        for filename in list_of_filenames:
            imgid = filename.strip('0')
            imgid = imgid.strip('.jpg')
            filename_map[int(imgid)] = filename
        image_info = image_info.replace(to_replace = {'image_id': filename_map})

        merged_df = pd.merge_ordered(image_info, dim_info, fill_method = 'ffill',left_on = 'image_id', right_on = 'file_name' )
        merged_df = merged_df.drop(['date_captured','image_id','flickr_url', 'license','coco_url','segmentation','area','iscrowd', 'id_x', 'id_y'], axis = 'columns')
        return merged_df
        
        #Todo beautify, docstring
    @staticmethod
    def scale_bbox(df_with_bbox):
        height = df_with_bbox['height'].values
        width  = df_with_bbox['width'].values
        # convert from Column = bbox : value = [x,y,obj_width, obj_height] to Columns = [x, y, obj_width, obj_height], value = _ | _ | _ | _:
        df2 = pd.DataFrame(df_with_bbox['bbox'].to_list(), columns = ['x','y','obj_width','obj_height'])
        df2['file_name'] = df_with_bbox['file_name']
        df2['category_id'] = df_with_bbox['category_id']
        df2['x'] = df2['x'].div(width, axis = 0)
        df2['y'] = df2['y'].div(height, axis = 0)
        df2['obj_width'] = df2['obj_width'].div(width, axis = 0)
        df2['obj_height'] = df2['obj_height'].div(height, axis = 0)
        df2 = df2.reindex(columns = ['file_name','category_id','x','y','obj_width','obj_height'])
        return df2

    #TODO docstring, beautify
    @staticmethod
    def save_to_yolo(df, path):
        """ This method written presuming path is the path to the json file above the data directory.
            Thus we sink into data directory and implant the txt annotations there"""
        assert df.columns in ['category_id','x','y','obj_width','obj_height'], 'Method not designed to be called directly.\n please use within the context of the class '
        image_folder = f'{path}/data'
        os.chdir(image_folder)
        previous_filename = ''
        file_contents =''
        for (index, filename), row in df.iterrows():
            
            if index == 0:
                previous_filename = filename
                file_contents = f"{int(row['category_id'])} {row['x']} {row['y']} {row['obj_width']} {row['obj_height']}\n"
            
            elif filename == previous_filename:
                file_contents = file_contents + f"{int(row['category_id'])} {row['x']} {row['y']} {row['obj_width']} {row['obj_height']}\n"        
            
            else:     
                
                with open(f'{previous_filename[:-3]}txt', 'w+') as file:
                    file.write(file_contents)        
                    file.close()
                
                file_contents = f"{int(row['category_id'])} {row['x']} {row['y']} {row['obj_width']} {row['obj_height']}\n"
            
            previous_filename = filename
            
        pass


train_dir, test_dir, val_dir = get_train_test_val_datadirs('coco-2017')

print(f'train: {train_dir}\ntest: {test_dir}\nval: {val_dir}')
df = pd.DataFrame()
for path in [train_dir, val_dir]:
    _f = f'{path}/labels.json'

    json_file_contents = ''

    with open(_f, 'r') as file:
        json_file_contents = file.read()#
        df = convert_to_df_with_bbox(json_file_contents, path, classmap = {1:0})
        df = scale_bbox(df)
        df = pd.pivot_table(df, index = [df.index.to_series(), 'file_name'])#, values = ['x','y','obj_width','obj_height']))
        df_to_yolo(df, path)
        file.close()

text = df.to_numpy()