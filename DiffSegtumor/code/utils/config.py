class Config:
    def __init__(self,task):

        if "brats_2d"in task:
            self.task = "brats_2d"
            self.save_dir = '../../Brats/data_prep/merged_nii'
            self.modalities = ['flair', 't1', 't1ce', 't2', 'seg']  # Modalities to process

            self.patch_size = (192, 192) 
            self.num_cls = 4
            self.num_channels = 4
            self.n_filters = 32
            self.early_stop_patience = 50
            self.batch_size = 32                
            self.num_workers = 2
        elif "brats_diff"in task:
            self.task = "brats_diff"
            self.save_dir = '/media/ssd2/zhiwei/Brats/merged_nii/'
            self.modalities = ['flair', 't1', 't1ce', 't2', 'seg']  # Modalities to process

            self.patch_size = (192, 192) 
            self.num_cls = 4
            self.num_channels = 4
            self.n_filters = 32
            self.early_stop_patience = 50
            self.batch_size = 32                
            self.num_workers = 2


        else:
            raise NameError("Please provide correct task name, see config.py")


