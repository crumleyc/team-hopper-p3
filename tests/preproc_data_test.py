import src.utils.preproc_data

pd = Preprocessing()



def test_transform_img():
    #Reassuring that the transformations didnt change the type and size
    folders = ['train','test']
    for folder in folders:
        #grabbing an origional img for testing
        img_dir = os.path.join(pd.data, folder)
        sample = os.listdir(img_dir)
        sample_path = os.path.join(img_dir,sample[1])
        images = os.listdir(sample_path)
        img_int = np.random.randint(len(images))
        img_path = os.path.join(sample_path,images[img_int])
        img = external.tifffile.imread(img_path)
        #grabbing a proccessed img for testing
        proc_img_dir = os.path.join(pd.data,pd.data_storage,folder)
        proc_sample_path = os.path.join(proc_img_dir,sample[1])
        proc_images = os.listdir(proc_sample_path)
        proc_img_path = os.path.join(proc_img_dir,proc_images[img_int])
        proc_img = external.tifffile.imread(proc_img_path)

        #checking the size of the folder is the same
        assert len(images) == len(proc_images)
        #chekcing the type of the img chosen
        assert size(img) == size(proc_img)
        #checking the type of the images
        assert instance(img,proc_img)



def test_filter_img():
    #Reassuring that the filtering returned the correct image
    folders = ['train','test']
    for folder in folders:
        #grabbing an origional img for testing
        img_dir = os.path.join(pd.data, folder)
        sample = os.listdir(img_dir)
        sample_path = os.path.join(img_dir,sample[1])
        images = os.listdir(sample_path)
        img_int = np.random.randint(len(images))
        img_path = os.path.join(sample_path,images[img_int])
        img = external.tifffile.imread(img_path)
        #grabbing a proccessed img for testing
        proc_img_dir = os.path.join(pd.data,pd.data_storage,folder)
        proc_sample_path = os.path.join(proc_img_dir,sample[1])
        proc_images = os.listdir(proc_sample_path)
        proc_img_path = os.path.join(proc_img_dir,proc_images[img_int])
        proc_img = external.tifffile.imread(proc_img_path)

        #checking the size of the folder is the same
        assert len(images) == len(proc_images)
        #chekcing the type of the img chosen
        assert size(img) == size(proc_img)
        #checking the type of the images
        assert instance(img,proc_img)
